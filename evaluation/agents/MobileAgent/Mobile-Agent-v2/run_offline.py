#!/usr/bin/env python3
"""
Mobile-Agent-v2 离线评估执行器
严格遵循源代码 run.py 流程，仅替换 ADB 为离线数据集读取

用法:
python run_offline_executor.py --dataset checkpoint_douyin.json --output results.json \
    --api_url "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions" \
    --api_key YOUR_KEY \
    --model qwen-vl-max \
    --qwen_api YOUR_QWEN_KEY

依赖:
- modelscope: OCR, GroundingDINO (需要安装 tf-keras==2.15.0 解决兼容性问题)
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

# 添加 Mobile-Agent-v2 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入 Mobile-Agent-v2 核心模块（保持原样）
from MobileAgent.api import inference_chat
from MobileAgent.text_localization import ocr
from MobileAgent.icon_localization import det
from MobileAgent.prompt import get_action_prompt, get_reflect_prompt, get_memory_prompt, get_process_prompt
from MobileAgent.chat import init_action_chat, init_reflect_chat, init_memory_chat, add_response, add_response_two_image

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
        self.reflection_switch = args.use_reflection
        self.memory_switch = args.use_memory
        self.add_info = args.add_info


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
    cropped_image.save(f"{temp_file}/{i}.jpg")


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
        futures = {executor.submit(process_image, image, query, qwen_api, caption_model): i for i, image in
                   enumerate(images)}
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            response = future.result()
            icon_map[i + 1] = response
    return icon_map


def merge_text_blocks(text_list, coordinates_list):
    """合并相邻文本块 - 从源代码完全复制"""
    merged_text_blocks = []
    merged_coordinates = []
    sorted_indices = sorted(range(len(coordinates_list)),
                            key=lambda k: (coordinates_list[k][1], coordinates_list[k][0]))
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
                    abs(sorted_coordinates_list[anchor][3] - sorted_coordinates_list[anchor][1] - \
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


def get_perception_infos_offline(screenshot_file, ocr_detection, ocr_recognition, groundingdino_model, config,
                                 temp_file):
    """
    离线版感知信息获取
    保持原版流程：OCR -> GroundingDINO -> 图标描述
    """
    # 获取图片尺寸
    width, height = Image.open(screenshot_file).size

    # 1. OCR识别文字（使用原版 modelscope OCR）
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
def parse_action(action, width=720, height=2400):
    """
    解析动作字符串
    返回格式: {"action": str, "bbox": [x,y,x,y] or None, "input_value": str or None}

    bbox格式说明：
    - Tap: [x, y, x, y] (中心点坐标重复)
    - Swipe: [x1, y1, x2, y2]
    """
    result = {
        "action": "wait",
        "bbox": None,
        "input_value": None
    }

    action = action.strip()

    # Stop
    if "Stop" in action:
        result["action"] = "finish"
        return result

    # Home
    if "Home" in action:
        result["action"] = "click"
        # Home键位置（底部中间）
        home_x = int(width * 0.5)
        home_y = int(height * 0.95)
        result["bbox"] = [home_x, home_y, home_x, home_y]
        return result

    # Back
    if "Back" in action:
        result["action"] = "click"
        # Back键位置（底部左侧）
        back_x = int(width * 0.1)
        back_y = int(height * 0.95)
        result["bbox"] = [back_x, back_y, back_x, back_y]
        return result

    # Open app
    if "Open app" in action:
        result["action"] = "click"
        result["bbox"] = None  # 无法确定具体位置
        return result

    # Tap(x, y)
    if "Tap" in action:
        match = re.search(r'Tap\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)', action)
        if match:
            x, y = int(match.group(1)), int(match.group(2))
            result["action"] = "click"
            # 直接使用 [x, y, x, y]，不扩展
            result["bbox"] = [x, y, x, y]
        return result

    # Swipe(x1, y1), (x2, y2)
    if "Swipe" in action:
        match = re.search(r'Swipe\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)', action)
        if match:
            x1, y1 = int(match.group(1)), int(match.group(2))
            x2, y2 = int(match.group(3)), int(match.group(4))
            result["action"] = "scroll"
            result["bbox"] = [x1, y1, x2, y2]

            # 判断滑动方向
            dx = x2 - x1
            dy = y2 - y1

            if abs(dy) > abs(dx):
                # 垂直方向滑动
                if dy < 0:
                    result["input_value"] = "up"  # 手指从下往上滑，页面向上滚动
                else:
                    result["input_value"] = "down"  # 手指从上往下滑，页面向下滚动
            else:
                # 水平方向滑动
                if dx < 0:
                    result["input_value"] = "left"  # 手指从右往左滑，页面向左滚动
                else:
                    result["input_value"] = "right"  # 手指从左往右滑，页面向右滚动
        return result

    # Type(text)
    if "Type" in action:
        if "(text)" not in action:
            text = action.split("(")[-1].split(")")[0]
        else:
            text = action.split(' "')[-1].split('"')[0]
        result["action"] = "input"
        result["input_value"] = text
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


# ============================ 主流程 ============================
def run_offline_trace(instruction, screenshot_paths, ocr_detection, ocr_recognition, groundingdino_model, config,
                      temp_file):
    """
    执行单条离线 trace
    严格遵循源代码 run.py 的流程
    """
    results = []

    # 状态变量（与源代码一致）
    thought_history = []
    summary_history = []
    action_history = []
    summary = ""
    action = ""
    completed_requirements = ""
    memory = ""
    insight = ""
    error_flag = False

    for step_idx, screenshot_file in enumerate(screenshot_paths):
        print(f"  Step {step_idx + 1}/{len(screenshot_paths)}", end=" ")

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

        # Action Agent
        prompt_action = get_action_prompt(
            instruction, perception_infos, width, height, keyboard,
            summary_history, action_history, summary, action,
            config.add_info, error_flag, completed_requirements, memory
        )
        chat_action = init_action_chat()
        chat_action = add_response("user", prompt_action, chat_action, screenshot_file)
        output_action = inference_chat(chat_action, config.model, config.api_url, config.token)

        # 解析输出
        thought = output_action.split("### Thought ###")[-1].split("### Action ###")[0].replace("\n", " ").replace(":",
                                                                                                                   " ").replace(
            "  ", " ").strip()
        summary = output_action.split("### Operation ###")[-1].replace("\n", " ").replace("  ", " ").strip()
        action = output_action.split("### Action ###")[-1].split("### Operation ###")[0].replace("\n", " ").replace(
            "  ", " ").strip()
        chat_action = add_response("assistant", output_action, chat_action)

        print(f"-> {action[:50]}")

        # Memory Agent
        if config.memory_switch:
            prompt_memory = get_memory_prompt(insight)
            chat_action = add_response("user", prompt_memory, chat_action)
            output_memory = inference_chat(chat_action, config.model, config.api_url, config.token)
            chat_action = add_response("assistant", output_memory, chat_action)
            output_memory = output_memory.split("### Important content ###")[-1].split("\n\n")[0].strip() + "\n"
            if "None" not in output_memory and output_memory not in memory:
                memory += output_memory

        # 解析动作并记录结果
        parsed = parse_action(action, width, height)
        results.append(parsed)

        # 如果是 Stop，结束
        if "Stop" in action:
            break

        # 保存上一状态
        last_perception_infos = copy.deepcopy(perception_infos)
        last_keyboard = keyboard

        # Reflection Agent（需要下一张截图）
        if config.reflection_switch and step_idx < len(screenshot_paths) - 1:
            # 获取下一张截图的感知信息
            next_screenshot = screenshot_paths[step_idx + 1]
            next_perception_infos, _, _ = get_perception_infos_offline(
                next_screenshot, ocr_detection, ocr_recognition, groundingdino_model, config, temp_file
            )
            shutil.rmtree(temp_file)
            os.mkdir(temp_file)

            # 检测键盘状态
            next_keyboard = False
            for perception_info in next_perception_infos:
                if perception_info['coordinates'][1] < keyboard_height_limit:
                    continue
                if 'ADB Keyboard' in perception_info['text']:
                    next_keyboard = True
                    break

            # Reflection
            prompt_reflect = get_reflect_prompt(
                instruction, last_perception_infos, next_perception_infos,
                width, height, last_keyboard, next_keyboard,
                summary, action, config.add_info
            )
            chat_reflect = init_reflect_chat()
            chat_reflect = add_response_two_image("user", prompt_reflect, chat_reflect,
                                                  [screenshot_file, next_screenshot])
            output_reflect = inference_chat(chat_reflect, config.model, config.api_url, config.token)
            reflect = output_reflect.split("### Answer ###")[-1].replace("\n", " ").strip()

            if 'A' in reflect:
                thought_history.append(thought)
                summary_history.append(summary)
                action_history.append(action)

                # Planning Agent
                prompt_planning = get_process_prompt(
                    instruction, thought_history, summary_history, action_history,
                    completed_requirements, config.add_info
                )
                chat_planning = init_memory_chat()
                chat_planning = add_response("user", prompt_planning, chat_planning)
                output_planning = inference_chat(chat_planning, config.model, config.api_url, config.token)
                completed_requirements = output_planning.split("### Completed contents ###")[-1].replace("\n",
                                                                                                         " ").strip()

                error_flag = False
            elif 'B' in reflect or 'C' in reflect:
                error_flag = True
        else:
            thought_history.append(thought)
            summary_history.append(summary)
            action_history.append(action)

            # Planning Agent
            prompt_planning = get_process_prompt(
                instruction, thought_history, summary_history, action_history,
                completed_requirements, config.add_info
            )
            chat_planning = init_memory_chat()
            chat_planning = add_response("user", prompt_planning, chat_planning)
            output_planning = inference_chat(chat_planning, config.model, config.api_url, config.token)
            completed_requirements = output_planning.split("### Completed contents ###")[-1].replace("\n", " ").strip()

    return results


def main():
    parser = argparse.ArgumentParser(description="Mobile-Agent-v2 离线评估")
    parser.add_argument("--dataset", required=True, help="数据集 JSON 文件")
    parser.add_argument("--output", required=True, help="输出 JSON 文件")
    parser.add_argument("--api_url", required=True, help="VLM API URL")
    parser.add_argument("--api_key", required=True, help="API Key")
    parser.add_argument("--model", default="qwen-vl-max", help="模型名称")
    parser.add_argument("--qwen_api", required=True, help="Qwen API Key")
    parser.add_argument("--caption_model", default="qwen-vl-max", help="图标描述模型")
    parser.add_argument("--use_reflection", action="store_true", help="启用 Reflection")
    parser.add_argument("--use_memory", action="store_true", help="启用 Memory")
    parser.add_argument("--add_info",
                        default="If you want to tap an icon of an app, use the action \"Open app\". If you want to exit an app, use the action \"Home\"",
                        help="额外提示")

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
    print(f"Reflection: {config.reflection_switch}")
    print(f"Memory: {config.memory_switch}")
    print("=" * 60)

    # 加载模型
    print("\nLoading models...")

    # GroundingDINO
    groundingdino_dir = snapshot_download('AI-ModelScope/GroundingDINO', revision='v1.0.0')
    groundingdino_model = pipeline('grounding-dino-task', model=groundingdino_dir)

    # OCR（modelscope 原版）
    ocr_detection = pipeline(Tasks.ocr_detection, model='damo/cv_resnet18_ocr-detection-line-level_damo')
    ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-document_damo')

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
                "agent_name": "Mobile-Agent-v2",
                "results": []
            }
    else:
        all_results = {
            "agent_name": "Mobile-Agent-v2",
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
        print(f"\nTestcase {testcase_id}: {testcase_desc[:50]}...")

        task_results = []

        # 处理 clean trace
        if "clean" in testcase:
            print(f"  Clean trace: {len(testcase['clean']['trace'])} steps")
            screenshots = [resolve_screenshot_path(cp["screenshot_path"], dataset_dir) for cp in
                           testcase["clean"]["trace"]]
            trace_results = run_offline_trace(
                testcase_desc, screenshots, ocr_detection, ocr_recognition,
                groundingdino_model, config, temp_file
            )
            task_results.append({
                "type": "clean",
                "trace_output": trace_results
            })

        # 处理 noise traces
        if "noise" in testcase:
            for noise_item in testcase["noise"]:
                noise_type = noise_item["type"]
                print(f"  Noise trace ({noise_type}): {len(noise_item['trace'])} steps")
                screenshots = [resolve_screenshot_path(cp["screenshot_path"], dataset_dir) for cp in
                               noise_item["trace"]]
                trace_results = run_offline_trace(
                    testcase_desc, screenshots, ocr_detection, ocr_recognition,
                    groundingdino_model, config, temp_file
                )
                task_results.append({
                    "type": noise_type,
                    "trace_output": trace_results
                })

        # 保存结果（追加到 app 对象中）
        app_result[f"task_{testcase_id}"] = task_results

        # 每个 testcase 结束立即保存
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(format_json_output(all_results))

        print(f"  ✓ Saved to {output_path}")

    # 清理
    shutil.rmtree(temp_file)

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {args.output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
