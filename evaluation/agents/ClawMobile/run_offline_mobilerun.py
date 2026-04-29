#!/usr/bin/env python3
"""
ClawMobile 离线评估运行脚本
===========================

完全遵循原来运行方式，只替换底层的 Driver。

输出格式:
{
    "agent_name": "ClawMobile",
    "results": [
        {
            "app_tested": "douyin",
            "task_1": [
                {
                    "type": "clean",
                    "trace_output": [
                        {"action": "click", "bbox": [x1,y1,x2,y2]},
                        {"action": "scroll", "bbox": [x1,y1,x2,y2], "input_value": "up"}
                    ]
                },
                {
                    "type": "VE",
                    "trace_output": [...]
                }
            ]
        }
    ]
}
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_dataset(path: str) -> Dict[str, Any]:
    """加载数据集"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_results_append(output_path: str, app_name: str, task_id: str,
                        trace_type: str, trace_output: List[Dict]) -> None:
    """
    追加结果到文件（不覆盖已有 task）
    每条 trace 完成时追加，bbox 压缩在一行
    """
    import re

    # 创建结果目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 读取现有结果
    results = {"agent_name": "ClawMobile", "results": []}
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
        except:
            pass

    # 查找或创建 app 条目
    app_entry = None
    for entry in results["results"]:
        if entry.get("app_tested") == app_name:
            app_entry = entry
            break

    if app_entry is None:
        app_entry = {"app_tested": app_name}
        results["results"].append(app_entry)

    # 创建 trace 条目
    trace_entry = {
        "type": trace_type,
        "trace_output": trace_output
    }

    # 追加到对应的 task（如果 task 不存在才创建）
    task_key = f"task_{task_id}"
    if task_key not in app_entry:
        app_entry[task_key] = []
    app_entry[task_key].append(trace_entry)

    # 保存（bbox 在一行输出，手动格式化，indent=4）
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('{\n')
        f.write(f'    "agent_name": "{results["agent_name"]}",\n')
        f.write('    "results": [\n')

        for i, app in enumerate(results["results"]):
            f.write('        {\n')
            f.write(f'            "app_tested": "{app.get("app_tested", "")}"')

            # 遍历所有 task_x 字段
            task_keys = sorted([k for k in app.keys() if k.startswith("task_")])
            for task_key in task_keys:
                traces = app[task_key]
                f.write(',\n')
                f.write(f'            "{task_key}": [\n')

                for j, trace in enumerate(traces):
                    f.write('                {\n')
                    f.write(f'                    "type": "{trace.get("type", "")}",\n')
                    f.write('                    "trace_output": [')

                    trace_output = trace.get("trace_output", [])
                    for k, action in enumerate(trace_output):
                        # 每个动作一行，bbox 压缩
                        # 将 bbox 数组格式化为紧凑形式
                        action_copy = dict(action)
                        if 'bbox' in action_copy:
                            bbox = action_copy['bbox']
                            # 创建紧凑的 JSON 字符串
                            action_str = json.dumps(action_copy, ensure_ascii=False)
                            # 替换 bbox 部分为紧凑格式
                            bbox_str = json.dumps(bbox)
                            action_str = re.sub(
                                r'"bbox":\s*\[[\d\.\,\s]+\]',
                                f'"bbox": {bbox_str}',
                                action_str
                            )
                        else:
                            action_str = json.dumps(action_copy, ensure_ascii=False)

                        if k < len(trace_output) - 1:
                            f.write(f'\n                        {action_str},')
                        else:
                            f.write(f'\n                        {action_str}')

                    f.write('\n                    ]')
                    if j < len(traces) - 1:
                        f.write('\n                },\n')
                    else:
                        f.write('\n                }\n')

                f.write('            ]')

            f.write('\n        }')
            if i < len(results["results"]) - 1:
                f.write(',')
            f.write('\n')

        f.write('    ]\n')
        f.write('}\n')

    print(f"📝 结果已追加: task_{task_id} [{trace_type}]")


class SingleTraceDriver:
    """
    单个 trace 的驱动包装器
    用于分别处理 clean trace 和每个 noise trace
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
            testcase_index: int,
            trace_type: str,
            trace_index: int = 0,
    ) -> None:
        """
        Args:
            dataset_path: 数据集路径
            testcase_index: 测试用例索引
            trace_type: "clean" 或 noise 类型（如 "VE"）
            trace_index: noise trace 的索引（仅用于 noise）
        """
        self._dataset_path = dataset_path
        self._testcase_index = testcase_index
        self._trace_type = trace_type
        self._trace_index = trace_index

        # 加载数据集
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self._dataset = json.load(f)

        # 解析当前 trace 的 checkpoints
        self._checkpoints = self._parse_trace_checkpoints()
        self._checkpoint_index = 0

        # 记录的动作序列
        self._recorded_actions: List[Dict[str, Any]] = []

        # 设备信息（模拟）
        self._screen_width = 1080
        self._screen_height = 2400

        self._connected = True

        # 控制每个 step 只记录一个动作（优先级：type > scroll/swipe/drag > click）
        self._step_actions: List[Dict[str, Any]] = []  # 收集当前 step 的所有动作
        self._step_recorded = False  # 是否已选择并记录动作

        print(f"[OfflineDriver] 已加载 {len(self._checkpoints)} 个 checkpoint")

    def _parse_trace_checkpoints(self) -> List[Dict[str, Any]]:
        """解析当前 trace 的 checkpoints"""
        testcases = self._dataset.get("testcases", [])
        if self._testcase_index >= len(testcases):
            return []

        tc = testcases[self._testcase_index]

        if self._trace_type == "clean":
            # 返回 clean trace 的 checkpoints
            clean_trace = tc.get("clean", {}).get("trace", [])
            return [{**cp, "testcase_id": tc.get("testcase_id", "unknown"), "type": "clean"}
                    for cp in clean_trace]
        else:
            # 返回指定 noise trace 的 checkpoints
            noise_traces = tc.get("noise", [])
            if self._trace_index >= len(noise_traces):
                return []
            noise = noise_traces[self._trace_index]
            noise_trace = noise.get("trace", [])
            return [{**cp, "testcase_id": tc.get("testcase_id", "unknown"), "type": self._trace_type}
                    for cp in noise_trace]

    def _resolve_path(self, path: str) -> str:
        """解析相对路径，基于数据集文件所在目录"""
        if os.path.isabs(path):
            return path
        dataset_dir = os.path.dirname(os.path.abspath(self._dataset_path))
        return os.path.normpath(os.path.join(dataset_dir, path))

    def _get_current_checkpoint(self) -> Optional[Dict[str, Any]]:
        """获取当前 checkpoint"""
        if self._checkpoint_index < len(self._checkpoints):
            return self._checkpoints[self._checkpoint_index]
        return None

    def advance_checkpoint(self) -> bool:
        """推进到下一个 checkpoint"""
        self._checkpoint_index += 1
        return self._checkpoint_index < len(self._checkpoints)

    def get_recorded_actions(self) -> List[Dict[str, Any]]:
        """获取记录的动作序列"""
        return self._recorded_actions

    def _record_action(self, action_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """收集动作，在 step 结束时选择最有意义的一个记录（优先级：type > scroll > click）"""
        if self._step_recorded:
            print(f"[OfflineDriver] 跳过动作: {action_type} (当前 step 已记录)")
            return {"ok": True, "skipped": True}

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
        return self._step_actions[0]

    def _commit_action(self, action: Dict[str, Any]) -> None:
        """提交选中的动作到记录列表"""
        action_type = action["type"]
        params = action["params"]
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
        """将原始动作映射为标准格式"""
        if action_type == "click":
            x, y = params.get("x", 0), params.get("y", 0)
            return {"action": "click", "bbox": [x - 5, y - 5, x + 5, y + 5]}
        elif action_type in ["scroll", "swipe", "drag"]:
            x1, y1 = params.get("x1", 0), params.get("y1", 0)
            x2, y2 = params.get("x2", 0), params.get("y2", 0)
            dx, dy = x2 - x1, y2 - y1
            direction = "down"
            if abs(dx) > abs(dy):
                direction = "right" if dx > 0 else "left"
            else:
                direction = "down" if dy > 0 else "up"
            return {"action": "scroll", "bbox": [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)],
                    "input_value": direction}
        elif action_type == "type":
            return {"action": "input", "bbox": [0, 0, 0, 0], "input_value": params.get("text", "")}
        elif action_type == "key":
            return {"action": "wait"}
        elif action_type == "open_app":
            return {"action": "wait"}
        else:
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
        import xml.etree.ElementTree as ET

        def parse_bounds_to_screen(bounds_str: str) -> Dict[str, int]:
            match = re.match(r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]', bounds_str)
            if match:
                return {
                    "left": int(match.group(1)),
                    "top": int(match.group(2)),
                    "right": int(match.group(3)),
                    "bottom": int(match.group(4))
                }
            return {"left": 0, "top": 0, "right": 0, "bottom": 0}

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
                    new_key = ATTR_MAP.get(key, key)
                    if value.lower() in ('true', 'false'):
                        result[new_key] = value.lower() == 'true'
                    else:
                        result[new_key] = value
                if 'bounds' in result:
                    result['boundsInScreen'] = parse_bounds_to_screen(result['bounds'])
                children = [node_to_dict(child) for child in node]
                if children:
                    result["children"] = children
                return result

            tree = node_to_dict(root)
            if 'bounds' not in tree:
                width = int(tree.get('width', self._screen_width))
                height = int(tree.get('height', self._screen_height))
                tree['bounds'] = f'[0,0][{width},{height}]'
                tree['boundsInScreen'] = {"left": 0, "top": 0, "right": width, "bottom": height}
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
        self._connected = True
        print("[OfflineDriver] 已连接（模拟）")

    async def ensure_connected(self) -> None:
        if not self._connected:
            await self.connect()

    # -- input actions -------------------------------------------------------

    async def tap(self, x: int, y: int) -> None:
        self._record_action("click", {"x": x, "y": y})
        # 不在这里推进 checkpoint，在 get_ui_tree 中统一推进

    async def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: float = 1000) -> None:
        self._record_action("scroll", {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "duration_ms": duration_ms})
        # 不在这里推进 checkpoint

    async def input_text(self, text: str, clear: bool = False) -> bool:
        self._record_action("type", {"text": text, "clear": clear})
        # 不在这里推进 checkpoint
        return True

    async def press_button(self, button: str) -> None:
        button_lower = button.lower()
        if button_lower not in self.supported_buttons:
            raise ValueError(f"Button '{button}' not supported")
        self._record_action("key", {"button": button_lower})
        # 不在这里推进 checkpoint

    async def drag(self, x1: int, y1: int, x2: int, y2: int, duration: float = 3.0) -> None:
        self._record_action("drag", {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "duration": duration})
        # 不在这里推进 checkpoint

    # -- app management ------------------------------------------------------

    async def start_app(self, package: str, activity: Optional[str] = None) -> str:
        self._record_action("open_app", {"package": package, "activity": activity})
        return f"App started: {package}"

    async def install_app(self, path: str, **kwargs) -> str:
        return "App install skipped (offline mode)"

    async def get_apps(self, include_system: bool = True) -> List[Dict[str, str]]:
        return [{"package": "com.ss.android.ugc.aweme", "label": "抖音"}]

    async def list_packages(self, include_system: bool = False) -> List[str]:
        return ["com.ss.android.ugc.aweme"]

    # -- state / observation -------------------------------------------------

    async def screenshot(self, hide_overlay: bool = True) -> bytes:
        checkpoint = self._get_current_checkpoint()
        if checkpoint:
            screenshot_path = checkpoint.get("screenshot_path", "")
            if screenshot_path:
                full_path = self._resolve_path(screenshot_path)
                if os.path.exists(full_path):
                    with open(full_path, 'rb') as f:
                        return f.read()
        return b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'

    async def get_ui_tree(self) -> Dict[str, Any]:
        # 处理上一步收集的动作：选择最有意义的一个记录
        if self._step_actions and not self._step_recorded:
            best_action = self._select_best_action()
            if best_action:
                self._commit_action(best_action)
                self._step_recorded = True
            # 推进 checkpoint（每个 step 只推进一次）
            self.advance_checkpoint()

        # 重置下一步的收集状态
        self._step_actions = []
        self._step_recorded = False

        checkpoint = self._get_current_checkpoint()
        print(
            f"[OfflineDriver] get_ui_tree: checkpoint_index={self._checkpoint_index}, has_checkpoint={checkpoint is not None}")

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

        a11y_tree = {}
        if xml_content:
            a11y_tree = self._parse_xml_to_tree(xml_content)
            print(f"[OfflineDriver] a11y_tree 解析完成, keys: {list(a11y_tree.keys())[:5]}")

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
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


async def run_single_trace(
        goal: str,
        driver: SingleTraceDriver,
        model: str,
        api_key: str,
        api_base: str,
        max_steps: int = 30,
        use_reasoning: bool = True,
) -> Dict[str, Any]:
    """运行单个 trace"""
    from mobilerun import MobileAgent, MobileConfig, AgentConfig
    from mobilerun.tools.ui.provider import AndroidStateProvider
    from mobilerun.tools.filters import ConciseFilter
    from mobilerun.tools.formatters import IndexedFormatter
    from llama_index.llms.openai_like import OpenAILike

    llm = OpenAILike(
        model=model,
        api_key=api_key,
        api_base=api_base,
        is_chat_model=True,
    )

    tree_filter = ConciseFilter()
    tree_formatter = IndexedFormatter()
    state_provider = AndroidStateProvider(
        driver=driver,
        tree_filter=tree_filter,
        tree_formatter=tree_formatter,
    )

    config = MobileConfig(
        agent=AgentConfig(
            name="mobilerun",
            max_steps=max_steps,
        ),
    )

    agent = MobileAgent(
        goal=goal,
        llms=llm,
        driver=driver,
        state_provider=state_provider,
        config=config,
        timeout=max_steps * 60,
    )

    # 保存driver引用
    driver._shared_state = agent.shared_state

    # 直接修改实例的finished属性为可控制的property
    # 先获取原始值
    original_finished_value = agent.shared_state.finished

    # 创建一个闭包来保存当前finished值
    finished_holder = {'value': original_finished_value}

    def get_finished(self):
        return finished_holder['value']

    def set_finished(self, value):
        if value and driver.has_checkpoint():
            print(
                f"[OfflineDriver] 阻止agent提前结束，继续执行 (checkpoint {driver._checkpoint_index}/{len(driver._checkpoints)})")
            return
        finished_holder['value'] = value

    # 用type动态创建一个新类
    import types

    # 获取原始类
    OriginalStateClass = agent.shared_state.__class__

    # 创建新类继承原始类
    class ControlledState(OriginalStateClass):
        pass

    # 设置property
    ControlledState.finished = property(
        lambda self: finished_holder['value'],
        lambda self, value: set_finished(self, value)
    )

    # 修改实例的类
    agent.shared_state.__class__ = ControlledState

    try:
        result = await agent.run()

        # 恢复原始类
        agent.shared_state.__class__ = OriginalStateClass

        # 处理最后收集的动作
        driver.finalize()
        return {
            "success": getattr(result, "success", False),
            "reason": getattr(result, "reason", ""),
            "steps": getattr(result, "steps", 0),
            "mapped_actions": driver.get_mapped_actions(),
        }
    except Exception as e:
        import traceback
        # 恢复原始类
        agent.shared_state.__class__ = OriginalStateClass
        # 即使出错也要处理收集的动作
        driver.finalize()
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "mapped_actions": driver.get_mapped_actions(),
        }


async def main():
    parser = argparse.ArgumentParser(description="ClawMobile 离线评估")
    parser.add_argument("--dataset", required=True, help="数据集路径")
    parser.add_argument("--output", required=True, help="结果输出路径")
    parser.add_argument("--api_key", help="OpenAI API 密钥")
    parser.add_argument("--api_base", help="API 基础 URL")
    parser.add_argument("--model", default="gpt-4o", help="模型名称")
    parser.add_argument("--max_steps", type=int, default=None, help="最大步数")
    parser.add_argument("--reasoning", action="store_true", help="使用 reasoning 模式")
    parser.add_argument("--limit", type=int, help="限制测试用例数量")

    args = parser.parse_args()

    # 设置环境变量
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    if args.api_base:
        os.environ["OPENAI_API_BASE"] = args.api_base

    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ 请设置 OPENAI_API_KEY 环境变量或使用 --api_key 参数")
        sys.exit(1)

    # 加载数据集
    print(f"📂 加载数据集: {args.dataset}")
    dataset = load_dataset(args.dataset)

    testcases = dataset.get("testcases", [])
    if not testcases:
        print("❌ 数据集中没有测试用例")
        sys.exit(1)

    app_name = dataset.get("app", "unknown")
    print(f"📱 App: {app_name}")
    print(f"📋 测试用例数: {len(testcases)}")

    if args.limit:
        testcases = testcases[:args.limit]
        print(f"⚡ 限制为前 {args.limit} 个测试用例")

    # 运行每个测试用例
    for i, tc in enumerate(testcases):
        # if i < 10 :
        #     continue
        testcase_id = str(tc.get("testcase_id", f"{i + 1}"))
        goal = tc.get("testcase_desc", tc.get("goal", ""))

        if not goal:
            print(f"⚠️ 测试用例 {testcase_id} 没有描述，跳过")
            continue

        print(f"\n{'=' * 60}")
        print(f"📍 测试用例 {i + 1}/{len(testcases)}: {testcase_id}")
        print(f"🎯 任务: {goal[:80]}...")
        print(f"{'=' * 60}")

        # 获取 clean trace
        clean_trace = tc.get("clean", {}).get("trace", [])
        clean_length = len(clean_trace)

        # 获取 noise traces
        noise_traces = tc.get("noise", [])
        noise_types = [n.get("type", "unknown") for n in noise_traces]

        print(f"📏 Clean trace 长度: {clean_length}")
        print(f"📊 Noise traces: {len(noise_traces)} 个 ({noise_types})")

        # -- 1. 运行 clean trace --
        print(f"\n🔹 运行 clean trace (长度: {clean_length})")

        driver = SingleTraceDriver(
            dataset_path=args.dataset,
            testcase_index=i,
            trace_type="clean",
        )

        actual_max_steps = args.max_steps if args.max_steps else clean_length

        result = await run_single_trace(
            goal=goal,
            driver=driver,
            model=args.model,
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            api_base=os.environ.get("OPENAI_API_BASE", ""),
            max_steps=actual_max_steps,
            use_reasoning=args.reasoning,
        )

        mapped_actions = result.get("mapped_actions", [])

        # 追加 clean trace 结果
        save_results_append(
            output_path=args.output,
            app_name=app_name,
            task_id=testcase_id,
            trace_type="clean",
            trace_output=mapped_actions
        )

        print(f"📝 结果已追加: task_{testcase_id} [clean]")
        print(f"✅ clean: {len(mapped_actions)} 个动作")

        # -- 2. 运行所有 noise traces --
        if noise_traces:
            print(f"\n🔹 运行 {len(noise_traces)} 个 noise traces")

            for j, noise in enumerate(noise_traces):
                noise_type = noise.get("type", f"noise_{j}")
                noise_trace = noise.get("trace", [])
                noise_length = len(noise_trace)

                print(f"\n📍 Noise {j + 1}/{len(noise_traces)}: type={noise_type}, 长度={noise_length}")

                driver = SingleTraceDriver(
                    dataset_path=args.dataset,
                    testcase_index=i,
                    trace_type=noise_type,
                    trace_index=j,
                )

                actual_max_steps = args.max_steps if args.max_steps else noise_length

                result = await run_single_trace(
                    goal=goal,
                    driver=driver,
                    model=args.model,
                    api_key=os.environ.get("OPENAI_API_KEY", ""),
                    api_base=os.environ.get("OPENAI_API_BASE", ""),
                    max_steps=actual_max_steps,
                    use_reasoning=args.reasoning,
                )

                mapped_actions = result.get("mapped_actions", [])

                # 追加 noise trace 结果
                save_results_append(
                    output_path=args.output,
                    app_name=app_name,
                    task_id=testcase_id,
                    trace_type=noise_type,
                    trace_output=mapped_actions
                )

                print(f"📝 结果已追加: task_{testcase_id} [{noise_type}]")
                print(f"✅ {noise_type}: {len(mapped_actions)} 个动作")

        print(f"\n{'=' * 60}")
        print(f"✅ 测试用例 {testcase_id} 完成")
        print(f"{'=' * 60}")

    print(f"\n{'=' * 60}")
    print(f"📊 评估完成")
    print(f"{'=' * 60}")
    print(f"📄 结果已保存: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
