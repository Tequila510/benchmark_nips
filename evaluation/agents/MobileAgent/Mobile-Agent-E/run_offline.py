#!/usr/bin/env python3
"""
Mobile-Agent-E 离线评估执行器
严格遵循源代码 inference_agent_E.py 流程，仅替换 ADB 为离线数据集读取

用法:
    python run_offline_executor_e.py --dataset checkpoint_douyin.json --output results.json \
        --api_url "https://api.openai.com/v1/chat/completions" \
        --api_key YOUR_KEY \
        --model gpt-4o \
        --qwen_api YOUR_QWEN_KEY

依赖:
    - modelscope: OCR (DBNet), GroundingDINO
    - dashscope: Qwen-VL API
"""

import os
import sys
import json
import re
import time
import copy
import shutil
import argparse
import concurrent.futures

# 添加 Mobile-Agent-E 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入 Mobile-Agent-E 核心模块（保持原样）
from MobileAgentE.api import inference_chat
from MobileAgentE.text_localization import ocr
from MobileAgentE.icon_localization import det
from MobileAgentE.agents import (
    InfoPool, Manager, Operator, Notetaker, ActionReflector,
    ExperienceReflectorShortCut, ExperienceReflectorTips,
    INIT_SHORTCUTS, ATOMIC_ACTION_SIGNITURES
)
from MobileAgentE.agents import add_response, add_response_two_image, extract_json_object

# 导入模型加载模块
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope import snapshot_download

import dashscope
from dashscope import MultiModalConversation
from PIL import Image


# ============================ 配置类 ============================
class Config:
    """运行配置"""
    def __init__(self, args):
        self.api_url = args.api_url
        self.token = args.api_key
        self.model = args.model
        self.qwen_api = args.qwen_api
        self.caption_model = args.caption_model
        self.caption_call_method = "api"  # 使用 API 方式
        self.temperature = args.temperature
        self.max_itr = args.max_itr
        self.max_consecutive_failures = args.max_consecutive_failures
        self.max_repetitive_actions = args.max_repetitive_actions
        self.enable_evolution = args.enable_evolution
        self.enable_experience_retriever = args.enable_experience_retriever


# ============================ 感知模块（保持原版逻辑） ============================
def get_all_files_in_folder(folder_path):
    """获取文件夹中所有文件"""
    file_list = []
    for file_name in os.listdir(folder_path):
        file_list.append(file_name)
    return file_list


def crop(image, box, i, temp_file):
    """裁剪图片"""
    image = Image.open(image)
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    if x1 >= x2 - 10 or y1 >= y2 - 10:
        return
    cropped_image = image.crop((x1, y1, x2, y2))
    # 处理 RGBA 格式
    if cropped_image.mode == 'RGBA':
        cropped_image = cropped_image.convert('RGB')
    cropped_image.save(os.path.join(temp_file, f"{i}.jpg"))


def process_image(image, query, qwen_api, caption_model):
    """使用 Qwen-VL API 处理图片"""
    dashscope.api_key = qwen_api
    image = "file://" + image
    messages = [{
        'role': 'user',
        'content': [
            {'image': image},
            {'text': query},
        ]
    }]
    response = MultiModalConversation.call(model=caption_model, messages=messages)
    try:
        response = response['output']['choices'][0]['message']['content'][0]["text"]
    except:
        response = "This is an icon."
    return response


def generate_api(images, query, qwen_api, caption_model):
    """并行处理多张图片"""
    icon_map = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_image, image, query, qwen_api, caption_model): i for i, image in enumerate(images)}
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            response = future.result()
            icon_map[i + 1] = response
    return icon_map


def merge_text_blocks(text_list, coordinates_list):
    """合并相邻文本块 - 从源代码完全复制"""
    merged_text_blocks = []
    merged_coordinates = []
    sorted_indices = sorted(range(len(coordinates_list)), key=lambda k: (coordinates_list[k][1], coordinates_list[k][0]))
    sorted_text_list = [text_list[i] for i in sorted_indices]
    sorted_coordinates_list = [coordinates_list[i] for i in sorted_indices]
    num_blocks = len(sorted_text_list)
    merge = [False] * num_blocks
    for i in range(num_blocks):
        if merge[i]:
            continue
        anchor = i
        group_text = [sorted_text_list[anchor]]
        group_coordinates = [sorted_coordinates_list[anchor]]
        for j in range(i + 1, num_blocks):
            if merge[j]:
                continue
            if abs(sorted_coordinates_list[anchor][0] - sorted_coordinates_list[j][0]) < 10 and \
                    sorted_coordinates_list[j][1] - sorted_coordinates_list[anchor][3] >= -10 and \
                    sorted_coordinates_list[j][1] - sorted_coordinates_list[anchor][3] < 30 and \
                    abs(sorted_coordinates_list[anchor][3] - sorted_coordinates_list[anchor][1] -
                        (sorted_coordinates_list[j][3] - sorted_coordinates_list[j][1])) < 10:
                group_text.append(sorted_text_list[j])
                group_coordinates.append(sorted_coordinates_list[j])
                merge[anchor] = True
                anchor = j
        merge[anchor] = True
        merged_text = "\n".join(group_text)
        min_x1 = min(group_coordinates, key=lambda x: x[0])[0]
        min_y1 = min(group_coordinates, key=lambda x: x[1])[1]
        max_x2 = max(group_coordinates, key=lambda x: x[2])[2]
        max_y2 = max(group_coordinates, key=lambda x: x[3])[3]
        merged_text_blocks.append(merged_text)
        merged_coordinates.append([min_x1, min_y1, max_x2, max_y2])
    return merged_text_blocks, merged_coordinates


def get_perception_infos_offline(screenshot_file, ocr_detection, ocr_recognition, groundingdino_model, config, temp_file):
    """
    离线版感知信息获取
    保持原版流程：OCR -> GroundingDINO -> 图标描述
    """
    # 确保 temp_file 是绝对路径
    temp_file = os.path.abspath(temp_file)

    # 获取图片尺寸
    width, height = Image.open(screenshot_file).size

    # 1. OCR识别文字（使用原版 DBNet OCR）
    text, coordinates = ocr(screenshot_file, ocr_detection, ocr_recognition)
    text, coordinates = merge_text_blocks(text, coordinates)

    # 构建感知信息
    perception_infos = []
    for i in range(len(coordinates)):
        perception_info = {"text": "text: " + text[i], "coordinates": coordinates[i]}
        perception_infos.append(perception_info)

    # 2. GroundingDINO 图标检测
    coordinates = det(screenshot_file, "icon", groundingdino_model)
    for i in range(len(coordinates)):
        perception_info = {"text": "icon", "coordinates": coordinates[i]}
        perception_infos.append(perception_info)

    # 3. 裁剪图标并生成描述
    image_box = []
    image_id = []
    for i in range(len(perception_infos)):
        if perception_infos[i]['text'] == 'icon':
            image_box.append(perception_infos[i]['coordinates'])
            image_id.append(i)

    for i in range(len(image_box)):
        crop(screenshot_file, image_box[i], image_id[i], temp_file)

    images = get_all_files_in_folder(temp_file)
    if len(images) > 0:
        images = sorted(images, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        image_id = [int(image.split('/')[-1].split('.')[0]) for image in images]

        icon_map = {}
        prompt = 'This image is an icon from a phone screen. Please briefly describe the shape and color of this icon in one sentence.'

        # 使用 API 生成图标描述
        for i in range(len(images)):
            images[i] = os.path.join(temp_file, images[i])

        icon_map = generate_api(images, prompt, config.qwen_api, config.caption_model)

        for i, j in zip(image_id, range(1, len(image_id) + 1)):
            if icon_map.get(j):
                # 过滤太大的图标
                image_path = os.path.join(temp_file, f"{i}.jpg")
                if os.path.exists(image_path):
                    icon_img = Image.open(image_path)
                    icon_width, icon_height = icon_img.size
                    if icon_height > 0.8 * height or icon_width * icon_height > 0.2 * width * height:
                        perception_infos[i]['text'] = "icon: None"
                    else:
                        perception_infos[i]['text'] = "icon: " + icon_map[j]

    # 4. 转换坐标为中心点（与源代码一致）
    for i in range(len(perception_infos)):
        perception_infos[i]['coordinates'] = [
            int((perception_infos[i]['coordinates'][0] + perception_infos[i]['coordinates'][2]) / 2),
            int((perception_infos[i]['coordinates'][1] + perception_infos[i]['coordinates'][3]) / 2)
        ]

    return perception_infos, width, height


# ============================ 动作解析 ============================
def parse_action(action_obj):
    """
    解析动作对象
    返回格式: {"action": str, "bbox": [x,y,x,y] or None, "input_value": str or None}

    Mobile-Agent-E 动作格式:
        - Tap(x, y) -> click
        - Swipe(x1, y1, x2, y2) -> scroll
        - Type(text) -> input
        - Open_App(app_name) -> open_app
        - Switch_App(app_name) -> switch_app
        - Back -> back
        - Home -> home
        - Enter -> enter
        - Wait -> wait
        - Shortcut -> 展开为多个原子动作
    """
    result = {
        "action": "wait",
        "bbox": None,
        "input_value": None
    }

    if action_obj is None:
        return result

    action_name = action_obj.get("name", "")
    arguments = action_obj.get("arguments", {})

    # Tap
    if action_name == "Tap":
        result["action"] = "click"
        x = arguments.get("x")
        y = arguments.get("y")
        if x is not None and y is not None:
            result["bbox"] = [int(x), int(y), int(x), int(y)]
        return result

    # Swipe
    if action_name == "Swipe":
        result["action"] = "scroll"
        x1 = arguments.get("x1")
        y1 = arguments.get("y1")
        x2 = arguments.get("x2")
        y2 = arguments.get("y2")
        if all(v is not None for v in [x1, y1, x2, y2]):
            result["bbox"] = [int(x1), int(y1), int(x2), int(y2)]
            # 判断滑动方向
            dx = x2 - x1
            dy = y2 - y1
            if abs(dy) > abs(dx):
                result["input_value"] = "up" if dy < 0 else "down"
            else:
                result["input_value"] = "left" if dx < 0 else "right"
        return result

    # Type
    if action_name == "Type":
        result["action"] = "input"
        result["input_value"] = arguments.get("text", "")
        return result

    # Open_App -> wait（不在GT动作范围内）
    if action_name == "Open_App":
        result["action"] = "wait"
        return result

    # Switch_App -> wait（不在GT动作范围内）
    if action_name == "Switch_App":
        result["action"] = "wait"
        return result

    # Back -> wait（不在GT动作范围内）
    if action_name == "Back":
        result["action"] = "wait"
        return result

    # Home -> wait（不在GT动作范围内）
    if action_name == "Home":
        result["action"] = "wait"
        return result

    # Enter -> wait（不在GT动作范围内）
    if action_name == "Enter":
        result["action"] = "wait"
        return result

    # Wait
    if action_name == "Wait":
        result["action"] = "wait"
        return result

    # Shortcut -> wait（不在GT动作范围内）
    if action_name not in ATOMIC_ACTION_SIGNITURES:
        result["action"] = "wait"
        return result

    return result


# ============================ JSON 格式化 ============================
def format_json_output(data):
    """格式化 JSON：4空格缩进，bbox 单行显示"""
    json_str = json.dumps(data, ensure_ascii=False, indent=4)
    # 将 bbox 数组压缩为单行
    pattern = r'"bbox":\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'

    def replace_bbox(match):
        return f'"bbox": [{match.group(1)}, {match.group(2)}, {match.group(3)}, {match.group(4)}]'

    json_str = re.sub(pattern, replace_bbox, json_str)
    return json_str


# ============================ 数据集加载 ============================
def load_dataset(dataset_path):
    """加载数据集，返回数据和数据集目录"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    dataset_dir = os.path.dirname(os.path.abspath(dataset_path))
    return data, dataset_dir


def resolve_screenshot_path(screenshot_path, dataset_dir):
    """将相对路径转换为绝对路径"""
    if os.path.isabs(screenshot_path):
        return screenshot_path
    if screenshot_path.startswith('./'):
        screenshot_path = screenshot_path[2:]
    return os.path.join(dataset_dir, screenshot_path)


# ============================ 离线版 Operator ============================
class OfflineOperator(Operator):
    """离线版 Operator，继承原始 Operator 的 prompt 生成逻辑，但不执行 ADB 操作"""

    def __init__(self, adb_path=None):
        # 不调用父类的 __init__，因为它会初始化 ADB 相关的东西
        self.adb_path = adb_path

    def execute_offline(self, action_str, info_pool):
        """
        离线执行动作（不实际执行，只返回解析后的动作对象）
        返回: (action_object, num_atomic_actions, error_message)

        Mobile-Agent-E 动作格式（JSON 格式）:
            - {"name":"Tap", "arguments":{"x":385, "y":107}}
            - {"name":"Swipe", "arguments":{"x1":100, "y1":500, "x2":100, "y2":200}}
            - {"name":"Type", "arguments":{"text":"冰淇淋"}}
            - {"name":"Back", "arguments":null}
        """
        action_obj = {"name": "", "arguments": {}}

        if not action_str:
            return action_obj, 0, "Empty action string"

        action_str = action_str.strip()

        # 优先尝试解析 JSON 格式（VLM 输出的标准格式）
        json_obj = extract_json_object(action_str, json_type="dict")
        if json_obj is not None:
            action_obj["name"] = json_obj.get("name", "")
            action_obj["arguments"] = json_obj.get("arguments") or {}
            return action_obj, 1, None

        # 兼容旧版位置参数格式（备用解析）
        if '(' in action_str:
            # 有参数的动作
            idx = action_str.index('(')
            name_part = action_str[:idx].strip()
            args_part = action_str[idx + 1:].rstrip(')')
            action_obj["name"] = name_part

            # 解析位置参数
            if args_part:
                # 处理字符串参数（带引号）
                if '"' in args_part or "'" in args_part:
                    # 字符串参数: Type("冰淇淋")
                    match = re.search(r'["\']([^"\']+)["\']', args_part)
                    if match:
                        # 根据动作类型确定参数名
                        if name_part in ["Type"]:
                            action_obj["arguments"]["text"] = match.group(1)
                        elif name_part in ["Open_App", "Switch_App"]:
                            action_obj["arguments"]["app_name"] = match.group(1)
                else:
                    # 数值参数: Tap(385, 107) 或 Swipe(100, 500, 100, 200)
                    nums = [int(x.strip()) for x in args_part.split(',') if x.strip().isdigit() or x.strip().lstrip('-').isdigit()]
                    if name_part == "Tap" and len(nums) >= 2:
                        action_obj["arguments"]["x"] = nums[0]
                        action_obj["arguments"]["y"] = nums[1]
                    elif name_part == "Swipe" and len(nums) >= 4:
                        action_obj["arguments"]["x1"] = nums[0]
                        action_obj["arguments"]["y1"] = nums[1]
                        action_obj["arguments"]["x2"] = nums[2]
                        action_obj["arguments"]["y2"] = nums[3]
        else:
            # 无参数的动作
            action_obj["name"] = action_str

        return action_obj, 1, None


# ============================ API 调用封装 ============================
def get_reasoning_model_api_response(chat, config):
    """调用推理模型 API"""
    return inference_chat(chat, config.model, config.api_url, config.token, temperature=config.temperature)


# ============================ 初始 Tips ============================
INIT_TIPS = """0. Do not add any payment information. If you are asked to sign in, ignore it or sign in as a guest if possible. Close any pop-up windows when opening an app.
1. By default, no APPs are opened in the background.
2. Screenshots may show partial text in text boxes from your previous input; this does not count as an error.
3. When creating new Notes, you do not need to enter a title unless the user specifically requests it.
"""


# ============================ 主流程 ============================
def run_offline_trace(instruction, screenshot_paths, ocr_detection, ocr_recognition, groundingdino_model, config, temp_file):
    """
    执行单条离线 trace
    严格遵循源代码 inference_agent_E.py 的流程
    """
    # 确保 temp_file 是绝对路径
    temp_file = os.path.abspath(temp_file)

    results = []

    # 初始化信息池
    info_pool = InfoPool(
        instruction=instruction,
        shortcuts=copy.deepcopy(INIT_SHORTCUTS),
        tips=INIT_TIPS,
        future_tasks=[],
        err_to_manager_thresh=2
    )

    # 初始化智能体
    manager = Manager()
    offline_operator = OfflineOperator()
    notetaker = Notetaker()
    action_reflector = ActionReflector()

    # 如果启用进化模块
    if config.enable_evolution:
        exp_reflector_shortcuts = ExperienceReflectorShortCut()
        exp_reflector_tips = ExperienceReflectorTips()

    perception_infos = None
    keyboard = False
    keyboard_height_limit = None

    for step_idx, screenshot_file in enumerate(screenshot_paths):
        print(f"      Step {step_idx + 1}/{len(screenshot_paths)}", end=" ")

        # 获取感知信息
        perception_infos, width, height = get_perception_infos_offline(
            screenshot_file, ocr_detection, ocr_recognition, groundingdino_model, config, temp_file
        )
        shutil.rmtree(temp_file)
        os.mkdir(temp_file)

        # 检测键盘状态
        keyboard = False
        keyboard_height_limit = 0.9 * height
        for perception_info in perception_infos:
            if perception_info['coordinates'][1] < keyboard_height_limit:
                continue
            if 'ADB Keyboard' in perception_info['text']:
                keyboard = True
                break

        info_pool.width = width
        info_pool.height = height
        info_pool.perception_infos_pre = copy.deepcopy(perception_infos)
        info_pool.keyboard_pre = keyboard

        # === Manager: 高层规划 ===
        info_pool.error_flag_plan = False
        if len(info_pool.action_outcomes) >= 2:
            latest_outcomes = info_pool.action_outcomes[-2:]
            if all(o in ["B", "C"] for o in latest_outcomes):
                info_pool.error_flag_plan = True

        info_pool.prev_subgoal = info_pool.current_subgoal

        prompt_planning = manager.get_prompt(info_pool)
        chat_planning = manager.init_chat()
        chat_planning = add_response("user", prompt_planning, chat_planning, image=screenshot_file)
        output_planning = get_reasoning_model_api_response(chat_planning, config)
        parsed_result_planning = manager.parse_response(output_planning)

        info_pool.plan = parsed_result_planning['plan']
        info_pool.current_subgoal = parsed_result_planning['current_subgoal']

        print(f"-> Subgoal: {info_pool.current_subgoal[:50]}")

        # 检查是否完成
        if "Finished" in info_pool.current_subgoal.strip():
            # 经验反思（如果启用）
            if config.enable_evolution and len(info_pool.action_outcomes) > 0:
                # 更新 shortcuts 和 tips
                pass

            # 解析最后一条动作
            if info_pool.last_action:
                parsed = parse_action(info_pool.last_action)
                results.append(parsed)

            # 打印完成
            print(f"      -> Finish")
            break

        # === Operator: 动作决策 ===
        prompt_action = offline_operator.get_prompt(info_pool)
        chat_action = offline_operator.init_chat()
        chat_action = add_response("user", prompt_action, chat_action, image=screenshot_file)
        output_action = get_reasoning_model_api_response(chat_action, config)
        parsed_result_action = offline_operator.parse_response(output_action)

        action_thought = parsed_result_action['thought']
        action_str = parsed_result_action['action']
        action_description = parsed_result_action['description']

        # DEBUG: 打印原始动作字符串
        print(f" [DEBUG] action_str: {action_str[:100] if action_str else 'None'}")

        # 执行动作（离线）
        action_obj, num_atomic_actions, error_message = offline_operator.execute_offline(action_str, info_pool)

        info_pool.last_action = action_obj
        info_pool.last_summary = action_description
        info_pool.last_action_thought = action_thought

        # DEBUG: 打印解析后的动作对象
        print(f" [DEBUG] action_obj: {action_obj}")

        # 解析动作并记录
        parsed = parse_action(action_obj)
        results.append(parsed)

        # 打印动作
        action_type = parsed["action"]
        bbox_str = str(parsed["bbox"]) if parsed["bbox"] else ""
        input_str = f' "{parsed["input_value"]}"' if parsed["input_value"] else ""
        print(f"      -> {action_type.capitalize()} {bbox_str}{input_str}")

        # === Action Reflector: 验证动作结果 ===
        # 保存上一张截图
        last_screenshot_file = screenshot_file

        # 获取下一张截图的感知信息（用于 reflection）
        if step_idx < len(screenshot_paths) - 1:
            next_screenshot = screenshot_paths[step_idx + 1]
            next_perception_infos, _, _ = get_perception_infos_offline(
                next_screenshot, ocr_detection, ocr_recognition, groundingdino_model, config, temp_file
            )
            shutil.rmtree(temp_file)
            os.mkdir(temp_file)

            next_keyboard = False
            for perception_info in next_perception_infos:
                if perception_info['coordinates'][1] < keyboard_height_limit:
                    continue
                if 'ADB Keyboard' in perception_info['text']:
                    next_keyboard = True
                    break

            info_pool.perception_infos_post = next_perception_infos
            info_pool.keyboard_post = next_keyboard

            # Action Reflection
            prompt_action_reflect = action_reflector.get_prompt(info_pool)
            chat_action_reflect = action_reflector.init_chat()
            chat_action_reflect = add_response_two_image("user", prompt_action_reflect, chat_action_reflect, [last_screenshot_file, next_screenshot])
            output_action_reflect = get_reasoning_model_api_response(chat_action_reflect, config)
            parsed_result_action_reflect = action_reflector.parse_response(output_action_reflect)

            outcome = parsed_result_action_reflect['outcome']
            error_description = parsed_result_action_reflect['error_description']
            progress_status = parsed_result_action_reflect['progress_status']

            # 判断动作结果
            if "A" in outcome:
                action_outcome = "A"
            elif "B" in outcome:
                action_outcome = "B"
            elif "C" in outcome:
                action_outcome = "C"
            else:
                action_outcome = "A"  # 默认成功

            info_pool.action_history.append(action_obj)
            info_pool.summary_history.append(action_description)
            info_pool.action_outcomes.append(action_outcome)
            info_pool.error_descriptions.append(error_description)
            info_pool.progress_status_history.append(progress_status)
            info_pool.progress_status = progress_status

            # === Notetaker: 记录重要内容 ===
            if action_outcome == "A":
                prompt_note = notetaker.get_prompt(info_pool)
                chat_note = notetaker.init_chat()
                chat_note = add_response("user", prompt_note, chat_note, image=next_screenshot)
                output_note = get_reasoning_model_api_response(chat_note, config)
                parsed_result_note = notetaker.parse_response(output_note)
                info_pool.important_notes = parsed_result_note['important_notes']

        # 检查终止条件（离线模式下禁用提前终止）
        # 离线评估应按数据集截图数量完整执行，不受 agent 决策影响
        # if len(info_pool.action_outcomes) >= config.max_consecutive_failures:
        #     last_k_outcomes = info_pool.action_outcomes[-config.max_consecutive_failures:]
        #     if all(o in ["B", "C"] for o in last_k_outcomes):
        #         print(f"\n    Consecutive failures reached. Stopping...")
        #         break

        # if len(info_pool.action_history) >= config.max_repetitive_actions:
        #     # 检查重复动作
        #     last_k_actions = info_pool.action_history[-config.max_repetitive_actions:]
        #     action_keys = [a.get("name", "") for a in last_k_actions]
        #     if len(set(action_keys)) == 1 and action_keys[0] not in ["Swipe", "Back"]:
        #         print(f"\n    Repetitive actions reached. Stopping...")
        #         break

        if step_idx + 1 >= config.max_itr:
            print(f"\n    Max iterations reached. Stopping...")
            break

    return results


def main():
    parser = argparse.ArgumentParser(description="Mobile-Agent-E 离线评估")
    parser.add_argument("--dataset", required=True, help="数据集 JSON 文件")
    parser.add_argument("--output", required=True, help="输出 JSON 文件")
    parser.add_argument("--api_url", required=True, help="VLM API URL")
    parser.add_argument("--api_key", required=True, help="API Key")
    parser.add_argument("--model", required=True, help="推理模型名称")
    parser.add_argument("--qwen_api", required=True, help="Qwen API Key")
    parser.add_argument("--caption_model", default="qwen-vl-plus", help="图标描述模型")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    parser.add_argument("--max_itr", type=int, default=40, help="最大迭代次数")
    parser.add_argument("--max_consecutive_failures", type=int, default=3, help="最大连续失败次数,已无效")
    parser.add_argument("--max_repetitive_actions", type=int, default=3, help="最大重复动作次数,已无效")
    parser.add_argument("--enable_evolution", action="store_true", help="启用自我进化模块")
    parser.add_argument("--enable_experience_retriever", action="store_true", help="启用经验检索")
    args = parser.parse_args()

    config = Config(args)

    # 加载数据集
    data, dataset_dir = load_dataset(args.dataset)
    app_name = data.get("app", "unknown")
    testcases = data.get("testcases", [])

    print("=" * 60)
    print(f"App: {app_name}")
    print(f"Model: {config.model}")
    print(f"Testcases: {len(testcases)}")
    print(f"Evolution: {config.enable_evolution}")
    print("=" * 60)

    # 加载模型
    print("\nLoading models...")

    # GroundingDINO
    groundingdino_dir = snapshot_download('AI-ModelScope/GroundingDINO', revision='v1.0.0')
    groundingdino_model = pipeline('grounding-dino-task', model=groundingdino_dir)

    # OCR（DBNet，不需要 TensorFlow）
    ocr_detection = pipeline(Tasks.ocr_detection, model='iic/cv_resnet18_ocr-detection-db-line-level_damo')
    ocr_recognition = pipeline(Tasks.ocr_recognition, model='iic/cv_convnextTiny_ocr-recognition-document_damo')

    print("Models loaded!")

    # 创建临时目录
    temp_file = f"temp_{config.model}"
    if os.path.exists(temp_file):
        shutil.rmtree(temp_file)
    os.mkdir(temp_file)

    # 创建输出目录
    output_path = os.path.abspath(args.output)
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")

    # 加载已有结果（支持累积追加）
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            all_results = existing_results
            print(f"Loaded existing results: {len(all_results['results'])} apps already tested")
        except Exception as e:
            print(f"Failed to load existing results, starting fresh: {e}")
            all_results = {
                "agent_name": "Mobile-Agent-E",
                "results": []
            }
    else:
        all_results = {
            "agent_name": "Mobile-Agent-E",
            "results": []
        }

    # 查找或创建当前 app 的结果对象
    app_result = None
    for r in all_results["results"]:
        if r.get("app_tested") == app_name:
            app_result = r
            print(f"Found existing results for {app_name}, will append tasks")
            break

    if app_result is None:
        app_result = {"app_tested": app_name}
        all_results["results"].append(app_result)
        print(f"Created new entry for {app_name}")

    # 处理每个 testcase
    for testcase in testcases:
        # if testcase["testcase_id"] <= 3:
        #     continue
        testcase_id = testcase["testcase_id"]
        testcase_desc = testcase["testcase_desc"]

        task_key = f"task_{testcase_id}"

        # 检查是否已存在该 task 的结果（跳过已完成的）
        if task_key in app_result and len(app_result[task_key]) > 0:
            print(f"\nTestcase {testcase_id}: Already completed, skipping...")
            continue

        print(f"\nTestcase {testcase_id}: {testcase_desc[:50]}...")

        task_results = []

        # 处理 clean trace
        if "clean" in testcase:
            print(f"   Clean trace: {len(testcase['clean']['trace'])} steps")
            screenshots = [resolve_screenshot_path(cp["screenshot_path"], dataset_dir) for cp in testcase["clean"]["trace"]]
            trace_results = run_offline_trace(
                testcase_desc, screenshots, ocr_detection, ocr_recognition, groundingdino_model, config, temp_file
            )
            task_results.append({
                "type": "clean",
                "trace_output": trace_results
            })

        # 处理 noise traces
        if "noise" in testcase:
            for noise_item in testcase["noise"]:
                noise_type = noise_item["type"]
                print(f"   Noise trace ({noise_type}): {len(noise_item['trace'])} steps")
                screenshots = [resolve_screenshot_path(cp["screenshot_path"], dataset_dir) for cp in noise_item["trace"]]
                trace_results = run_offline_trace(
                    testcase_desc, screenshots, ocr_detection, ocr_recognition, groundingdino_model, config, temp_file
                )
                task_results.append({
                    "type": noise_type,
                    "trace_output": trace_results
                })

        # 保存结果（追加到 app 对象中）
        app_result[task_key] = task_results

        # 每个 testcase 结束立即保存
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(format_json_output(all_results))
        print(f"   ✓ Saved to {output_path}")

    # 清理
    shutil.rmtree(temp_file)

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {args.output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
