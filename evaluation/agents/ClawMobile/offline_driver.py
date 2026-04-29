"""
ClawMobile 离线评估 - Python 层 OfflineDriver
=============================================

这个文件实现了一个离线版的 DeviceDriver，用于替换 mobilerun 的 AndroidDriver。

使用方法：
1. 将此文件放到任意位置
2. 运行 run_offline_mobilerun.py 进行离线评估

黑盒保证：
- 不修改任何 mobilerun 内部代码
- 通过 driver 参数注入给 MobileAgent
- MobileAgent 完全不知道在离线运行
"""

from __future__ import annotations
import asyncio
import json
import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
from mobilerun.tools.driver.base import DeviceDriver, DeviceDisconnectedError


class OfflineDriver(DeviceDriver):
    """
    离线设备驱动 - 从本地数据集读取 UI 状态，记录动作而不执行
    """

    platform = "Android"
    supported = {
        "tap", "swipe", "input_text", "press_button", "start_app",
        "screenshot", "get_ui_tree", "get_date", "get_apps", "list_packages",
    }
    supported_buttons = {"back", "home", "enter"}

    _BUTTON_KEYCODES = {
        "back": 4,
        "home": 3,
        "enter": 66,
    }

    def __init__(
        self,
        dataset_path: str,
        output_path: str,
        testcase_index: int = 0,
    ) -> None:
        """
        初始化离线驱动

        Args:
            dataset_path: 数据集 JSON 文件路径
            output_path: 结果输出路径
            testcase_index: 当前测试用例索引
        """
        self._dataset_path = dataset_path
        self._output_path = output_path
        self._testcase_index = testcase_index

        # 加载数据集
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self._dataset = json.load(f)

        # 解析 checkpoints
        self._checkpoints = self._parse_checkpoints()
        self._checkpoint_index = 0

        # 记录的动作序列
        self._recorded_actions: List[Dict[str, Any]] = []

        # 当前 UI 状态
        self._current_ui_tree: Dict[str, Any] = {}
        self._current_screenshot: bytes = b""

        # 设备信息（模拟）
        self._screen_width = 1080
        self._screen_height = 2400

        self._connected = True  # 模拟已连接

        # 控制每个 step 只记录一个动作（优先级：type > scroll/swipe/drag > click）
        self._step_actions: List[Dict[str, Any]] = []  # 收集当前 step 的所有动作
        self._step_recorded = False  # 是否已选择并记录动作

        print(f"[OfflineDriver] 已加载 {len(self._checkpoints)} 个 checkpoint")

    def _parse_checkpoints(self) -> List[Dict[str, Any]]:
        """解析数据集中的所有 checkpoint"""
        all_checkpoints = []

        testcases = self._dataset.get("testcases", [])
        if not testcases:
            return all_checkpoints

        # 获取当前测试用例
        if self._testcase_index >= len(testcases):
            return all_checkpoints

        tc = testcases[self._testcase_index]

        # 解析 clean trace
        if tc.get("clean", {}).get("trace"):
            for cp in tc["clean"]["trace"]:
                all_checkpoints.append({
                    **cp,
                    "testcase_id": tc.get("testcase_id", "unknown"),
                    "type": "clean"
                })

        # 解析 noise traces
        if tc.get("noise"):
            for noise in tc["noise"]:
                if noise.get("trace"):
                    for cp in noise["trace"]:
                        all_checkpoints.append({
                            **cp,
                            "testcase_id": tc.get("testcase_id", "unknown"),
                            "type": noise.get("type", "unknown")
                        })

        return all_checkpoints

    def get_clean_trace_length(self) -> int:
        """获取 clean trace 的长度（用于设置 max_steps）"""
        testcases = self._dataset.get("testcases", [])
        if self._testcase_index >= len(testcases):
            return 0
        tc = testcases[self._testcase_index]
        clean_trace = tc.get("clean", {}).get("trace", [])
        return len(clean_trace)

    def set_testcase_index(self, index: int) -> None:
        """设置当前测试用例索引"""
        self._testcase_index = index
        self._checkpoints = self._parse_checkpoints()
        self._checkpoint_index = 0
        self._recorded_actions = []

    def has_checkpoint(self) -> bool:
        """
        检查是否还有未处理的checkpoint

        Returns:
            是否还有更多checkpoint
        """
        return self._checkpoint_index < len(self._checkpoints)

    def advance_checkpoint(self) -> bool:
        """
        推进到下一个 checkpoint

        Returns:
            是否还有更多 checkpoint
        """
        self._checkpoint_index += 1
        if self._checkpoint_index >= len(self._checkpoints):
            self._save_results()
            return False
        return True

    def _save_results(self) -> None:
        """保存记录的结果"""
        os.makedirs(os.path.dirname(self._output_path), exist_ok=True)

        result = {
            "testcase_id": self._checkpoints[0].get("testcase_id", "unknown") if self._checkpoints else "unknown",
            "actions": self._recorded_actions,
            "total_steps": len(self._recorded_actions),
        }

        with open(self._output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"[OfflineDriver] 结果已保存: {self._output_path}")

    def _get_current_checkpoint(self) -> Optional[Dict[str, Any]]:
        """获取当前 checkpoint"""
        if self._checkpoint_index < len(self._checkpoints):
            return self._checkpoints[self._checkpoint_index]
        return None

    def _resolve_path(self, path: str) -> str:
        """解析相对路径，基于数据集文件所在目录"""
        if os.path.isabs(path):
            return path
        # 相对于数据集文件的目录
        dataset_dir = os.path.dirname(os.path.abspath(self._dataset_path))
        return os.path.normpath(os.path.join(dataset_dir, path))

    def get_recorded_actions(self) -> List[Dict[str, Any]]:
        """获取记录的动作序列"""
        return self._recorded_actions

    def _record_action(self, action_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        收集动作，在 step 结束时选择最有意义的一个记录

        优先级：type (input) > scroll/swipe/drag > click > 其他
        """
        # 如果已经选择了动作，跳过后续动作
        if self._step_recorded:
            print(f"[OfflineDriver] 跳过动作: {action_type} (当前 step 已记录)")
            return {"ok": True, "skipped": True}

        # 收集动作
        self._step_actions.append({
            "type": action_type,
            "params": params,
            "checkpoint_index": self._checkpoint_index,
        })
        print(f"[OfflineDriver] 收集动作: {action_type}")

        return {"ok": True, "collected": True}

    def _select_best_action(self) -> Optional[Dict[str, Any]]:
        """从收集的动作中选择最有意义的一个"""
        if not self._step_actions:
            return None

        # 优先级：type > scroll/swipe/drag > click > key > open_app
        priority_order = ["type", "scroll", "swipe", "drag", "click", "key", "open_app"]

        for action_type in priority_order:
            for action in self._step_actions:
                if action["type"] == action_type:
                    return action

        # 返回第一个
        return self._step_actions[0]

    def _commit_action(self, action: Dict[str, Any]) -> None:
        """提交选中的动作到记录列表"""
        action_type = action["type"]
        params = action["params"]

        # 映射到标准格式
        mapped = self._map_action(action_type, params)

        record = {
            "step": len(self._recorded_actions) + 1,
            "original_type": action_type,
            "original_params": params,
            "mapped": mapped,
            "checkpoint_index": action["checkpoint_index"],
        }
        self._recorded_actions.append(record)
        print(f"[OfflineDriver] 记录动作: {action_type} -> {mapped['action']} @ step {record['step']}")

    def _map_action(self, action_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """将 agent 动作映射为标准格式"""

        if action_type == "click":
            # click: {"action": "click", "bbox": [x1,y1,x2,y2]}
            x, y = params.get("x", 0), params.get("y", 0)
            # 假设点击区域为 10x10 的方框
            bbox = [x - 5, y - 5, x + 5, y + 5]
            return {"action": "click", "bbox": bbox}

        elif action_type == "scroll":
            # scroll: {"action": "scroll", "bbox": [x1,y1,x2,y2], "input_value": "up/down/left/right"}
            x1, y1 = params.get("x1", 0), params.get("y1", 0)
            x2, y2 = params.get("x2", 0), params.get("y2", 0)

            # 判断方向
            dx = x2 - x1
            dy = y2 - y1

            if abs(dy) > abs(dx):
                direction = "up" if dy < 0 else "down"
            else:
                direction = "left" if dx < 0 else "right"

            bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            return {"action": "scroll", "bbox": bbox, "input_value": direction}

        elif action_type == "type":
            # input: {"action": "input", "bbox": [x1,y1,x2,y2], "input_value": "content"}
            # input 动作没有特定坐标，使用屏幕中心
            text = params.get("text", "")
            bbox = [0, 0, 0, 0]  # input 通常没有 bbox
            return {"action": "input", "bbox": bbox, "input_value": text}

        elif action_type == "key":
            # 按键动作，映射为 wait
            button = params.get("button", "")
            return {"action": "wait"}

        elif action_type == "open_app":
            # 打开应用，映射为 wait
            return {"action": "wait"}

        elif action_type == "drag":
            # drag 类似 scroll
            x1, y1 = params.get("x1", 0), params.get("y1", 0)
            x2, y2 = params.get("x2", 0), params.get("y2", 0)
            dx = x2 - x1
            dy = y2 - y1
            if abs(dy) > abs(dx):
                direction = "up" if dy < 0 else "down"
            else:
                direction = "left" if dx < 0 else "right"
            bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            return {"action": "scroll", "bbox": bbox, "input_value": direction}

        else:
            # 未知动作，映射为 wait
            return {"action": "wait"}

    def get_mapped_actions(self) -> List[Dict[str, Any]]:
        """获取映射后的标准动作列表"""
        return [action.get("mapped", {"action": "wait"}) for action in self._recorded_actions]

    def finalize(self) -> None:
        """处理最后收集的动作（agent 结束时调用）"""
        if self._step_actions and not self._step_recorded:
            best_action = self._select_best_action()
            if best_action:
                self._commit_action(best_action)
                self._step_recorded = True
        print(f"[OfflineDriver] finalize: 共记录 {len(self._recorded_actions)} 个动作")

    def _parse_xml_to_tree(self, xml_str: str) -> Dict[str, Any]:
        """将 XML 字符串解析为字典树结构"""
        import re

        def parse_bounds_to_screen(bounds_str: str) -> Dict[str, int]:
            """将 bounds 字符串 '[x1,y1][x2,y2]' 转换为 boundsInScreen 字典"""
            match = re.match(r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]', bounds_str)
            if match:
                return {
                    "left": int(match.group(1)),
                    "top": int(match.group(2)),
                    "right": int(match.group(3)),
                    "bottom": int(match.group(4))
                }
            return {"left": 0, "top": 0, "right": 0, "bottom": 0}

        # XML 属性名 -> formatter 期望的属性名映射
        ATTR_MAP = {
            'class': 'className',
            'resource-id': 'resourceId',
            'content-desc': 'contentDescription',
            'checkable': 'isCheckable',
            'checked': 'isChecked',
            'clickable': 'isClickable',
            'enabled': 'isEnabled',
            'focusable': 'isFocusable',
            'focused': 'isFocused',
            'scrollable': 'isScrollable',
            'long-clickable': 'isLongClickable',
            'password': 'isPassword',
            'selected': 'isSelected',
            'displayed': 'isDisplayed',
        }

        try:
            root = ET.fromstring(xml_str)

            def node_to_dict(node: ET.Element) -> Dict[str, Any]:
                result = {}

                for key, value in node.attrib.items():
                    # 转换属性名
                    new_key = ATTR_MAP.get(key, key)
                    # 布尔值转换
                    if value.lower() in ('true', 'false'):
                        result[new_key] = value.lower() == 'true'
                    else:
                        result[new_key] = value

                # 关键：将 bounds 转换为 boundsInScreen 格式
                if 'bounds' in result:
                    result['boundsInScreen'] = parse_bounds_to_screen(result['bounds'])

                children = [node_to_dict(child) for child in node]
                if children:
                    result["children"] = children
                return result

            tree = node_to_dict(root)

            # 如果根节点没有 bounds，根据 width/height 添加
            if 'bounds' not in tree:
                width = int(tree.get('width', self._screen_width))
                height = int(tree.get('height', self._screen_height))
                tree['bounds'] = f'[0,0][{width},{height}]'
                tree['boundsInScreen'] = {"left": 0, "top": 0, "right": width, "bottom": height}

            # 更新屏幕尺寸（用于 device_context）
            if 'width' in tree:
                self._screen_width = int(tree['width'])
            if 'height' in tree:
                self._screen_height = int(tree['height'])

            return tree
        except Exception as e:
            print(f"[OfflineDriver] XML 解析失败: {e}")
            return {}

    # -- lifecycle -----------------------------------------------------------

    async def connect(self) -> None:
        """模拟连接"""
        self._connected = True
        print("[OfflineDriver] 已连接（模拟）")

    async def ensure_connected(self) -> None:
        """确保已连接"""
        if not self._connected:
            await self.connect()

    # -- input actions -------------------------------------------------------

    async def tap(self, x: int, y: int) -> None:
        """记录 tap 动作"""
        self._record_action("click", {"x": x, "y": y})
        # 不在这里推进 checkpoint，在 get_ui_tree 中统一推进

    async def swipe(
        self, x1: int, y1: int, x2: int, y2: int, duration_ms: float = 1000
    ) -> None:
        """记录 swipe 动作"""
        self._record_action("scroll", {
            "x1": x1, "y1": y1, "x2": x2, "y2": y2, "duration_ms": duration_ms
        })
        # 不在这里推进 checkpoint

    async def input_text(self, text: str, clear: bool = False) -> bool:
        """记录 input 动作"""
        self._record_action("type", {"text": text, "clear": clear})
        # 不在这里推进 checkpoint
        return True

    async def press_button(self, button: str) -> None:
        """记录按键动作"""
        button_lower = button.lower()
        if button_lower not in self.supported_buttons:
            raise ValueError(f"Button '{button}' not supported")
        self._record_action("key", {"button": button_lower})
        # 不在这里推进 checkpoint

    async def drag(self, x1: int, y1: int, x2: int, y2: int, duration: float = 3.0) -> None:
        """记录 drag 动作"""
        self._record_action("drag", {
            "x1": x1, "y1": y1, "x2": x2, "y2": y2, "duration": duration
        })
        # 不在这里推进 checkpoint

    # -- app management ------------------------------------------------------

    async def start_app(self, package: str, activity: Optional[str] = None) -> str:
        """记录启动应用"""
        self._record_action("open_app", {"package": package, "activity": activity})
        return f"App started: {package}"

    async def install_app(self, path: str, **kwargs) -> str:
        """安装应用（离线不支持）"""
        return "App install skipped (offline mode)"

    async def get_apps(self, include_system: bool = True) -> List[Dict[str, str]]:
        """获取应用列表"""
        return [{"package": "com.ss.android.ugc.aweme", "label": "抖音"}]

    async def list_packages(self, include_system: bool = False) -> List[str]:
        """获取包列表"""
        return ["com.ss.android.ugc.aweme"]

    # -- state / observation -------------------------------------------------

    async def screenshot(self, hide_overlay: bool = True) -> bytes:
        """返回当前 checkpoint 的截图"""
        checkpoint = self._get_current_checkpoint()
        if checkpoint:
            screenshot_path = checkpoint.get("screenshot_path", "")
            if screenshot_path:
                full_path = self._resolve_path(screenshot_path)
                if os.path.exists(full_path):
                    with open(full_path, 'rb') as f:
                        return f.read()

        # 返回空 PNG
        return b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'

    async def get_ui_tree(self) -> Dict[str, Any]:
        """
        返回当前 checkpoint 的 UI 树

        必须返回包含以下 key 的字典：
        - a11y_tree: accessibility 树（字典或原始数据）
        - phone_state: 手机状态
        - device_context: 设备上下文（包含 screen_bounds）
        - raw_xml: 原始 XML 字符串
        """
        # 处理上一步收集的动作：选择最有意义的一个记录
        if self._step_actions and not self._step_recorded:
            best_action = self._select_best_action()
            if best_action:
                self._commit_action(best_action)
                self._step_recorded = True
        elif not self._step_actions:
            # 【新增】没有成功执行任何动作（如 open_app 失败），记录空动作
            self._recorded_actions.append({
                "step": len(self._recorded_actions) + 1,
                "original_type": "none",
                "original_params": {},
                "mapped": {"action": "wait"},
                "checkpoint_index": self._checkpoint_index,
            })
            print(f"[OfflineDriver] 记录打开app动作映射为wait @ step {len(self._recorded_actions)}")


        # 推进 checkpoint
        self.advance_checkpoint()

        # 重置下一步的收集状态
        self._step_actions = []
        self._step_recorded = False

        checkpoint = self._get_current_checkpoint()
        print(f"[OfflineDriver] get_ui_tree: checkpoint_index={self._checkpoint_index}, has_checkpoint={checkpoint is not None}")

        # 读取 XML 文件
        xml_content = ""
        if checkpoint:
            xml_path = checkpoint.get("xml_path", "")
            print(f"[OfflineDriver] xml_path: {xml_path}")
            if xml_path:
                full_path = self._resolve_path(xml_path)
                print(f"[OfflineDriver] resolved path: {full_path}")
                print(f"[OfflineDriver] file exists: {os.path.exists(full_path)}")
                if os.path.exists(full_path):
                    with open(full_path, 'r', encoding='utf-8') as f:
                        xml_content = f.read()
                    print(f"[OfflineDriver] 读取 XML 成功: {len(xml_content)} bytes")

        # 解析 XML 为字典树
        a11y_tree = {}
        if xml_content:
            a11y_tree = self._parse_xml_to_tree(xml_content)
            print(f"[OfflineDriver] a11y_tree 解析完成, keys: {list(a11y_tree.keys())[:5]}")

        # 返回与 Portal 格式一致的结构
        return {
            "a11y_tree": a11y_tree,
            "phone_state": {
                "foreground_app": "com.ss.android.ugc.aweme",
                "orientation": "portrait",
            },
            "device_context": {
                "screen_bounds": {
                    "width": self._screen_width,
                    "height": self._screen_height,
                },
                "device_id": "offline_device",
            },
            "raw_xml": xml_content,
        }

    async def get_date(self) -> str:
        """返回当前日期时间"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
