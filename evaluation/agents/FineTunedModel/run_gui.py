import os
import io
import json
import re
import time
import base64
import argparse
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
import requests
from prompt import system_prompt

load_dotenv()

COMPRESSED_WIDTH = 576
COMPRESSED_HEIGHT = 1280
# MODEL_NAME = "bytedance/ui-tars-1.5-7b"
# MODEL_NAME = "autoglm-phone"

# Openrouter UI-TARS 7B
OPENROUTER_API_base = os.environ.get("OPENROUTER_API_base")
OPENROUTER_API_key = os.environ.get("OPENROUTER_API_key")
# ==================================

# Zhipu autoGLM-Phone-9B
ZHIPU_API_base = os.environ.get("ZHIPU_API_base")
ZHIPU_API_key = os.environ.get("ZHIPU_API_key")
# ==================================

# ============================================
# GUI-Owl-7B (阿里云 DashScope)
# ============================================
DASHSCOPE_API_KEY = os.environ.get("Aliyun_API_KEY_QWEN")
GUI_OWL_API_URL = "https://dashscope.aliyuncs.com/api/v2/apps/gui-owl/gui_agent_server"

# 阿里云OSS配置（数据集已上传到OSS，直接拼接URL）
OSS_URL_PREFIX = os.environ.get("OSS_URL_PREFIX")
# GUI-Owl坐标转换配置
# GUI-Owl 使用 0-1000 归一化坐标系（官方文档确认）
# 参考: https://deepwiki.com/X-PLUG/MobileAgent/1.1-gui-owl-foundation-models
# "Grounding: 0-1000 Coordinate Mapping"
# "outputs coordinates in a normalized 0–1000 grid"
GUI_OWL_IMAGE_WIDTH = int(os.environ.get("GUI_OWL_IMAGE_WIDTH", "1000"))  # GUI-Owl内部坐标系宽度
GUI_OWL_IMAGE_HEIGHT = int(os.environ.get("GUI_OWL_IMAGE_HEIGHT", "1000"))  # GUI-Owl内部坐标系高度


def convert_gui_owl_coords(x: int, y: int, original_width: int, original_height: int) -> tuple:
    """
    将GUI-Owl返回的 0-1000 归一化坐标转换为原图绝对像素坐标

    GUI-Owl 官方文档确认使用 0-1000 归一化坐标系
    转换公式:
        actual_x = gui_owl_x * (original_width / 1000)
        actual_y = gui_owl_y * (original_height / 1000)

    Args:
        x, y: GUI-Owl返回的归一化坐标 (0-1000)
        original_width, original_height: 原图分辨率

    Returns:
        tuple: 转换后的绝对像素坐标 (x, y)
    """
    # 计算转换比例
    scale_x = original_width / GUI_OWL_IMAGE_WIDTH
    scale_y = original_height / GUI_OWL_IMAGE_HEIGHT

    converted_x = int(x * scale_x)
    converted_y = int(y * scale_y)

    print(
        f"  [GUI-Owl] 坐标转换: ({x}, {y}) -> ({converted_x}, {converted_y}) [原图: {original_width}x{original_height}]")

    return converted_x, converted_y

def get_image_url(image_path: str) -> str:
    """
    将本地图片路径转换为OSS公网URL

    Args:
        image_path: 本地图片路径，如 "../../../data/douyin-20/xxx.png"

    Returns:
        str: OSS公网URL，如 "https://data-benchmark.oss-cn-beijing.aliyuncs.com/douyin-20/xxx.png"
    """
    # 标准化路径分隔符
    normalized_path = image_path.replace("\\", "/")

    # 找到 "data/" 的位置，提取其后的路径
    # 例如: "../../../data/douyin-20/xxx.png" -> "douyin-20/xxx.png"
    marker = "/data/"
    idx = normalized_path.find(marker)

    if idx != -1:
        # 找到了 /data/，取其后的内容
        oss_path = normalized_path[idx + len(marker):]
    else:
        # 兼容没有data前缀的情况
        oss_path = normalized_path.lstrip("./")

    # 拼接OSS URL
    return f"{OSS_URL_PREFIX.rstrip('/')}/{oss_path}"


def encode_image(image_path: str) -> tuple:
    """
    将图片压缩到 576x1280 后转 base64
    返回: (base64_string, original_width, original_height)
    """
    with Image.open(image_path) as img:
        original_width, original_height = img.size
        img_resized = img.resize((COMPRESSED_WIDTH, COMPRESSED_HEIGHT), Image.LANCZOS)
        buffer = io.BytesIO()
        img_resized.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8"), original_width, original_height


def convert_coords(x: int, y: int, original_width: int, original_height: int) -> tuple:
    """
    将压缩图片上的坐标转换回原始尺寸
    1. 归一化：x / COMPRESSED_WIDTH, y / COMPRESSED_HEIGHT
    2. 乘回原尺寸：* original_width, * original_height
    """
    normalized_x = x / COMPRESSED_WIDTH
    normalized_y = y / COMPRESSED_HEIGHT
    original_x = int(normalized_x * original_width)
    original_y = int(normalized_y * original_height)
    return original_x, original_y


# ============================================
# GUI-Owl-7B 动作解析
# ============================================
def parse_action_GUI_OWL(response_data: dict, original_width: int, original_height: int) -> dict:
    """
    解析GUI-Owl API返回的动作

    GUI-Owl返回格式：
    {
        "Explanation": "Click on the xxx.",
        "Operation": "Click (530, 66, 530, 66)",
        "Thought": "..."
    }

    Operation格式：
    - Click (x1, y1, x2, y2)  -> click
    - Swipe (x1, y1, x2, y2)  -> scroll
    - Type "text"  -> input
    - Done  -> finish
    """
    result = {
        "action": "unknown",
        "bbox": [],
        "input_value": ""
    }

    try:
        # 提取data字段
        data = response_data.get("output", [{}])[0].get("content", [{}])[0].get("data", {})

        # 打印原始返回便于调试
        print(f"\n  [GUI-Owl] Raw data: {json.dumps(data, ensure_ascii=False)[:500]}")

        # 提取Operation字段
        operation = data.get("Operation", "") or data.get("operation", "")
        explanation = data.get("Explanation", "") or data.get("explanation", "")

        if explanation:
            print(f"  [GUI-Owl] Explanation: {explanation[:100]}")

        if not operation:
            print("  [GUI-Owl] 未找到Operation字段")
            result["action"] = "wait"
            return result

        print(f"  [GUI-Owl] Operation: {operation}")

        # 解析坐标 - 匹配 (x1, y1, x2, y2) 或 (x1, y1)
        coord_pattern = r'\((\d+)\s*,\s*(\d+)(?:\s*,\s*(\d+)\s*,\s*(\d+))?\)'
        coord_match = re.search(coord_pattern, operation)

        coords = []
        if coord_match:
            coords = [int(g) for g in coord_match.groups() if g is not None]

        # 解析动作类型
        operation_lower = operation.lower()

        # Click/Tap动作
        if "click" in operation_lower or "tap" in operation_lower:
            result["action"] = "click"
            if len(coords) >= 2:
                if len(coords) >= 4:
                    # bbox格式 (x1, y1, x2, y2)，取中心点
                    cx = (coords[0] + coords[2]) // 2
                    cy = (coords[1] + coords[3]) // 2
                    # 坐标转换
                    cx, cy = convert_gui_owl_coords(cx, cy, original_width, original_height)
                    result["bbox"] = [cx, cy, cx, cy]
                else:
                    # 单点格式 (x, y)
                    x, y = convert_gui_owl_coords(coords[0], coords[1], original_width, original_height)
                    result["bbox"] = [x, y, x, y]
            return result

        # Swipe/Scroll动作
        if "swipe" in operation_lower or "scroll" in operation_lower:
            result["action"] = "scroll"
            if len(coords) >= 4:
                x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]

                # 坐标转换
                x1, y1 = convert_gui_owl_coords(x1, y1, original_width, original_height)

                # 根据滑动方向确定scroll方向
                dx = x2 - x1
                dy = y2 - y1

                if abs(dx) > abs(dy):
                    result["input_value"] = "right" if dx > 0 else "left"
                else:
                    result["input_value"] = "down" if dy > 0 else "up"

                result["bbox"] = [x1, y1, x1, y1]
            return result

        # Type/Input动作
        if "type" in operation_lower or "input" in operation_lower:
            result["action"] = "input"
            # 尝试提取输入文本
            # 格式1: Type (text) - 文本在括号内
            # 格式2: Type "text" - 文本在引号内
            type_match = re.search(r'[Tt]ype\s*\(([^)]+)\)', operation)
            if type_match:
                result["input_value"] = type_match.group(1).strip()
            else:
                # 尝试引号格式
                text_match = re.search(r'["\'](.+?)["\']', operation)
                if text_match:
                    result["input_value"] = text_match.group(1)

            if len(coords) >= 2:
                x, y = convert_gui_owl_coords(coords[0], coords[1], original_width, original_height)
                result["bbox"] = [x, y, x, y]
            return result

        # Done/Finish
        if "done" in operation_lower or "finish" in operation_lower or "complete" in operation_lower:
            result["action"] = "finish"
            return result

        # Back - 返回上一页
        if "back" in operation_lower:
            result["action"] = "back"
            return result

        # Wait - 等待
        if "wait" in operation_lower:
            result["action"] = "wait"
            return result

        # 未知动作
        print(f"  [GUI-Owl] 未知操作: {operation}")
        result["action"] = "wait"
        return result

    except Exception as e:
        print(f"  [GUI-Owl] 解析错误: {e}")
        result["action"] = "parse_error"
        result["input_value"] = str(e)
        return result


# ============================================
# GUI-Owl-7B API调用
# ============================================
def call_model_GUI_OWL(task: str, image_path: str, session_id: str = "") -> dict:
    """
    调用阿里云GUI-Owl-7B模型

    Args:
        task: 任务描述
        image_path: 图片路径
        session_id: 会话ID（用于多轮对话，首次调用传空字符串）

    Returns:
        dict: 解析后的动作结果
    """
    if not DASHSCOPE_API_KEY:
        print(f"    [Error] DASHSCOPE_API_KEY 未配置")
        return {"action": "config_error", "bbox": "", "input_value": "DASHSCOPE_API_KEY not set"}

    # 获取原始图片尺寸
    with Image.open(image_path) as img:
        original_width, original_height = img.size

    # 获取图片URL（GUI-Owl需要HTTP URL，不支持base64）
    image_url = get_image_url(image_path)

    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }

    # 构建请求体（Mobile端链路）
    payload = {
        "app_id": "gui-owl",
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "data",
                        "data": {
                            "messages": [
                                {"image": image_url},
                                {"instruction": task},
                                {"session_id": session_id},
                                {"device_type": "mobile"},
                                {"pipeline_type": "agent"},
                                {"model_name": "pre-gui_owl_7b"},
                                {"thought_language": "chinese"},
                                {"param_list": [{"add_info": ""}]}
                            ]
                        }
                    }
                ]
            }
        ]
    }

    try:
        start_time = time.time()
        response = requests.post(GUI_OWL_API_URL, headers=headers, json=payload, timeout=60)
        elapsed = time.time() - start_time

        if response.status_code != 200:
            print(f"    [Error] API返回错误: {response.status_code} - {response.text}")
            return {"action": "api_error", "bbox": "",
                    "input_value": f"HTTP {response.status_code}: {response.text[:200]}"}

        response_data = response.json()

        # 检查业务状态码
        output = response_data.get("output", [{}])[0]
        if output.get("code") != "200":
            print(f"    [Error] 业务错误: {output.get('message', 'Unknown error')}")
            return {"action": "api_error", "bbox": "", "input_value": output.get("message", "Unknown error")}

        print(f"  [GUI-Owl] 耗时: {elapsed:.2f}s")

        # 解析动作
        return parse_action_GUI_OWL(response_data, original_width, original_height)

    except requests.exceptions.Timeout:
        print(f"    [Error] API超时")
        return {"action": "api_error", "bbox": "", "input_value": "API timeout"}
    except Exception as e:
        print(f"    [Error] API调用失败: {e}")
        return {"action": "api_error", "bbox": "", "input_value": str(e)}


# UI-TARS-1.5-7B process action mapping
def parse_action_UI_TARS(response_text: str, original_width: int, original_height: int) -> dict:
    result = {
        "action": "unknown",
        "bbox": [],
        "input_value": ""
    }
    print(f"\n{response_text}")
    # 提取 Action: 后面的内容
    action_match = re.search(r'Action:\s*(.+?)(?:\n|$)', response_text)
    if not action_match:
        result["action"] = "wait"
        return result

    action_text = action_match.group(1).strip()

    # 匹配坐标 - 兼容多种格式
    points = []

    # 格式1: <point>x y</point>
    point_matches = re.findall(r'<point>(\d+)\s+(\d+)</point>', action_text)
    points.extend(point_matches)

    # 格式2: point='(x,y)' 或 point='(x, y)'
    point_matches = re.findall(r"point\s*=\s*['\"]?\s*\((\d+)\s*,\s*(\d+)\)\s*['\"]?", action_text)
    if not points:
        points.extend(point_matches)

    # 格式3: start_box='(x,y)'
    start_box_matches = re.findall(r"start_box\s*=\s*['\"]?\s*\((\d+)\s*,\s*(\d+)\)\s*['\"]?", action_text)
    if not points:
        points.extend(start_box_matches)

    # 格式4: end_box='(x,y)'
    end_box_matches = re.findall(r"end_box\s*=\s*['\"]?\s*\((\d+)\s*,\s*(\d+)\)\s*['\"]?", action_text)

    # scroll 动作：提取方向
    if action_text.startswith("scroll"):
        result["action"] = "scroll"
        direction_match = re.search(r"direction\s*=\s*['\"](up|down|left|right)['\"]", action_text)
        if direction_match:
            result["input_value"] = direction_match.group(1)
        if points:
            x, y = int(points[0][0]), int(points[0][1])
            orig_x, orig_y = convert_coords(x, y, original_width, original_height)
            result["bbox"] = [orig_x, orig_y, orig_x, orig_y]
        return result

    # drag 动作：计算方向
    if action_text.startswith("drag"):
        result["action"] = "scroll"
        # drag 可能有 start_box 和 end_box
        if start_box_matches and end_box_matches:
            start_x, start_y = int(start_box_matches[0][0]), int(start_box_matches[0][1])
            end_x, end_y = int(end_box_matches[0][0]), int(end_box_matches[0][1])

            dx = end_x - start_x
            dy = end_y - start_y

            if abs(dx) > abs(dy):
                result["input_value"] = "right" if dx > 0 else "left"
            else:
                result["input_value"] = "down" if dy > 0 else "up"

            orig_x, orig_y = convert_coords(start_x, start_y, original_width, original_height)
            result["bbox"] = [orig_x, orig_y, orig_x, orig_y]
        elif len(points) >= 2:
            start_x, start_y = int(points[0][0]), int(points[0][1])
            end_x, end_y = int(points[1][0]), int(points[1][1])

            dx = end_x - start_x
            dy = end_y - start_y

            if abs(dx) > abs(dy):
                result["input_value"] = "right" if dx > 0 else "left"
            else:
                result["input_value"] = "down" if dy > 0 else "up"

            orig_x, orig_y = convert_coords(start_x, start_y, original_width, original_height)
            result["bbox"] = [orig_x, orig_y, orig_x, orig_y]
        return result

    # click 和 long_press 映射为 click
    if action_text.startswith("click") or action_text.startswith("long_press"):
        result["action"] = "click"
        if points:
            x, y = int(points[0][0]), int(points[0][1])
            orig_x, orig_y = convert_coords(x, y, original_width, original_height)
            result["bbox"] = [orig_x, orig_y, orig_x, orig_y]
        return result

    # type 映射为 input
    if action_text.startswith("type"):
        result["action"] = "input"
        content_match = re.search(r"content\s*=\s*['\"](.+?)['\"]", action_text, re.DOTALL)
        if content_match:
            result["input_value"] = content_match.group(1).replace("\\n", "\n")
        if points:
            x, y = int(points[0][0]), int(points[0][1])
            orig_x, orig_y = convert_coords(x, y, original_width, original_height)
            result["bbox"] = [orig_x, orig_y, orig_x, orig_y]
        return result

    # wait 映射为 wait
    if action_text.startswith("wait"):
        result["action"] = "wait"
        return result

    # finished 映射为 finish
    if action_text.startswith("finished"):
        result["action"] = "finish"
        return result

    # 其他全部映射为 wait
    result["action"] = "wait"
    return result


def call_model_UI_TARS(task: str, image_path: str) -> dict:
    """调用模型并解析输出"""
    image_base64, original_width, original_height = encode_image(image_path)
    client = OpenAI(
        base_url=OPENROUTER_API_base,
        api_key=OPENROUTER_API_key,
    )
    try:
        response = client.chat.completions.create(
            model="bytedance/ui-tars-1.5-7b",
            messages=[
                {"role": "system", "content": system_prompt.format(instruction=task)},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                        },
                        {"type": "text",
                         "text": "In the current state, in order to advance the task and achieve the goal, please output the next action."}
                    ],
                }
            ],
            max_tokens=2000,
            temperature=0.0
        )

        content = response.choices[0].message.content
        return parse_action_UI_TARS(content, original_width, original_height)

    except Exception as e:
        print(f"    [Error] API调用失败: {e}")
        return {"action": "api_error", "bbox": "", "input_value": str(e)}


def run_trace(trace: list, task_desc: str, dataset_dir: str, model: str) -> list:
    """运行一条 trace"""
    trace_output = []

    # GUI-Owl需要维护session_id用于多轮对话
    gui_owl_session_id = ""

    for i, checkpoint in enumerate(trace):
        checkpoint_id = checkpoint["checkpoint_id"]
        screenshot_path = os.path.join(dataset_dir, checkpoint["screenshot_path"].lstrip("./"))

        print(f"    Checkpoint {checkpoint_id} ({i + 1}/{len(trace)})...", end=" ")

        if not os.path.exists(screenshot_path):
            print(f"截图不存在")
            trace_output.append(
                {"action": "file_not_found", "bbox": "", "input_value": f"screenshot: {screenshot_path}"})
            continue

        start_time = time.time()
        result = {}
        if model == "ui-tars-1.5-7b":
            result = call_model_UI_TARS(task_desc, screenshot_path)
        elif model == "gui-owl-7b":
            result = call_model_GUI_OWL(task_desc, screenshot_path, gui_owl_session_id)
        # elif model == "autoglm-phone-9b":
        #     result = call_model_autoGLM(task_desc, screenshot_path)
        elapsed = time.time() - start_time

        print(f"action:{result['action']} bbox:{result['bbox']} input_value:{result['input_value']} ({elapsed:.2f}s)")
        trace_output.append(result)

        if result["action"] == "finish":
            print(f"    任务完成")
            break

    return trace_output


def load_dataset(path: str) -> dict:
    """加载数据集"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_results(path: str, model: str) -> dict:
    """加载已有结果（追加模式）"""
    if not os.path.exists(path):
        return {"agent_name": model, "results": []}

    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {"agent_name": model, "results": []}


def save_results(results: dict, path: str):
    """保存结果"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def main():
    parser = argparse.ArgumentParser(description="GUI Agent Benchmark Runner")
    parser.add_argument("--dataset", required=True, help="数据集 JSON 文件路径")
    parser.add_argument("--output", required=True, help="输出 JSON 文件路径")
    parser.add_argument("--model", required=True,
                        choices=["ui-tars-1.5-7b", "autoglm-phone-9b", "gui-owl-7b"],
                        help="model")
    args = parser.parse_args()
    model = args.model
    print("=" * 60)
    print("GUI Agent Benchmark Runner")
    print(f"Model: {model}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print("=" * 60)

    # 加载数据集
    print("\n加载数据集...")
    dataset = load_dataset(args.dataset)
    app_name = dataset["app"]
    testcases = dataset["testcases"]
    print(f"  App: {app_name}")
    print(f"  测试用例数: {len(testcases)}")

    dataset_dir = os.path.dirname(args.dataset)

    # 加载已有结果
    results = load_results(args.output, model)

    # 统计已完成
    completed_tasks = set()
    for r in results["results"]:
        for key in r:
            if key.startswith("task_"):
                completed_tasks.add((r.get("app_tested", ""), key))
    print(f"  已完成任务: {len(completed_tasks)}")

    # 遍历测试用例
    for testcase in testcases:
        testcase_id = testcase["testcase_id"]
        testcase_desc = testcase["testcase_desc"]
        task_key = f"task_{testcase_id}"

        print(f"\n{'=' * 60}")
        print(f"Testcase {testcase_id}: {testcase_desc[:50]}...")

        if (app_name, task_key) in completed_tasks:
            print(f"  [跳过] 已完成")
            continue

        task_result = {"app_tested": app_name, task_key: []}

        # Clean trace
        print(f"\n  [Clean Trace]")
        clean_trace = testcase.get("clean", {}).get("trace", [])
        if clean_trace:
            clean_output = run_trace(clean_trace, testcase_desc, dataset_dir, model)
            task_result[task_key].append({"type": "clean", "trace_output": clean_output})

        # Noise traces
        noise_list = testcase.get("noise", [])
        for noise in noise_list:
            noise_type = noise.get("type", "unknown")
            noise_trace = noise.get("trace", [])

            print(f"\n  [Noise Trace: {noise_type}]")
            if noise_trace:
                noise_output = run_trace(noise_trace, testcase_desc, dataset_dir, model)
                task_result[task_key].append({"type": noise_type, "trace_output": noise_output})

        # 追加结果
        app_found = False
        for r in results["results"]:
            if r.get("app_tested") == app_name:
                r[task_key] = task_result[task_key]
                app_found = True
                break

        if not app_found:
            results["results"].append(task_result)

        # 保存
        save_results(results, args.output)
        print(f"\n  [已保存] {args.output}")

    print("\n" + "=" * 60)
    print("Benchmark 完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
