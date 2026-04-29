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
        if model == "ui-tars-1.5-7b" :
            result = call_model_UI_TARS(task_desc, screenshot_path)
        # elif model == "autoglm-phone-9b" :
        #     result = call_model_autoGLM(task_desc, screenshot_path)
        # else :
        #     result = call_model_GUI_OWL(task_desc, screenshot_path)
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
