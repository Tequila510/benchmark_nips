#!/usr/bin/env python3
"""
修改版 task_executor.py - 离线模式
从静态数据集读取截图/XML，只预测不执行，输出评估格式

用法:
python task_executor_offline.py --dataset 数据集.json --output 结果.json
"""

import argparse
import json
import os
import re
import sys
import time
import tempfile

import prompts
from config import load_config
from and_controller import traverse_tree
from model import parse_explore_rsp, OpenAIModel, QwenModel
from utils import print_with_color, draw_bbox_multi, encode_image
import requests
import cv2
import base64


def format_bbox_single_line(json_str):
    """将bbox数组格式化为单行"""
    pattern = r'"bbox":\s*\[\s*([\d\.\-\s,]+?)\s*\]'

    def replace_bbox(match):
        numbers = match.group(1)
        numbers = re.sub(r'\s+', '', numbers)
        return f'"bbox": [{numbers}]'

    return re.sub(pattern, replace_bbox, json_str)


def map_action(act_name: str) -> str:
    """
    映射AppAgent动作到Ground Truth动作

    AppAgent: tap, text, long_press, swipe, FINISH, grid, ERROR
    Ground Truth: click, input, scroll, wait, finish
    """
    mapping = {
        "tap": "click",
        "long_press": "click",
        "text": "input",
        "swipe": "scroll",
        "FINISH": "finish",
        "grid": "wait",
        "ERROR": "wait"
    }
    return mapping.get(act_name, "wait")


def process_trace(trace: list, task_desc: str, mllm, configs: dict, dataset_dir: str) -> list:
    """
    处理单条trace，返回预测结果列表

    Args:
        dataset_dir: 数据集JSON所在的目录，用于解析相对路径
    """
    predictions = []
    last_act = "None"

    for checkpoint in trace:
        checkpoint_id = checkpoint["checkpoint_id"]
        screenshot_path = checkpoint["screenshot_path"]
        xml_path = checkpoint["xml_path"]

        # ===== 路径修正：如果是相对路径，基于数据集目录解析 =====
        if not os.path.isabs(screenshot_path):
            screenshot_path = os.path.join(dataset_dir, screenshot_path)
        if not os.path.isabs(xml_path):
            xml_path = os.path.join(dataset_dir, xml_path)

        print_with_color(f"  Checkpoint {checkpoint_id}/{len(trace)}", "yellow")

        # ===== 核心修改1：从静态文件读取 =====
        # 检查文件是否存在
        if not os.path.exists(screenshot_path):
            print_with_color(f"    ERROR: Screenshot not found: {screenshot_path}", "red")
            predictions.append({"action": "wait", "bbox": None})
            continue

        if not os.path.exists(xml_path):
            print_with_color(f"    ERROR: XML not found: {xml_path}", "red")
            predictions.append({"action": "wait", "bbox": None})
            continue

        # 解析XML获取UI元素
        clickable_list = []
        focusable_list = []
        traverse_tree(xml_path, clickable_list, "clickable", True)
        traverse_tree(xml_path, focusable_list, "focusable", True)

        elem_list = clickable_list.copy()
        for elem in focusable_list:
            bbox = elem.bbox
            center = (bbox[0][0] + bbox[1][0]) // 2, (bbox[0][1] + bbox[1][1]) // 2
            close = False
            for e in clickable_list:
                bbox = e.bbox
                center_ = (bbox[0][0] + bbox[1][0]) // 2, (bbox[0][1] + bbox[1][1]) // 2
                dist = (abs(center[0] - center_[0]) ** 2 + abs(center[1] - center_[1]) ** 2) ** 0.5
                if dist <= configs["MIN_DIST"]:
                    close = True
                    break
            if not close:
                elem_list.append(elem)

        # 标注截图（保存到临时目录）
        labeled_path = os.path.join(tempfile.gettempdir(), f"labeled_{checkpoint_id}.png")
        draw_bbox_multi(screenshot_path, labeled_path, elem_list, dark_mode=configs["DARK_MODE"])

        # 构建prompt（不使用UI文档）
        prompt = re.sub(r"<ui_document>", "", prompts.task_template)
        prompt = re.sub(r"<task_description>", task_desc, prompt)
        prompt = re.sub(r"<last_act>", last_act, prompt)

        # ===== 核心修改2：调用模型预测 =====
        print_with_color("    Thinking...", "yellow")

        # 调试：直接调用API查看完整响应
        try:
            # 打印prompt信息
            print_with_color(f"    [DEBUG] Prompt length: {len(prompt)} chars", "yellow")

            # ===== 压缩图片 =====
            img = cv2.imread(labeled_path)
            if img is None:
                print_with_color(f"    [DEBUG] Failed to read image: {labeled_path}", "red")
                predictions.append({"action": "wait", "bbox": None})
                continue

            # 压缩：缩小尺寸和质量
            height, width = img.shape[:2]
            max_size = 1280  # 最大边长

            if max(height, width) > max_size:
                scale = max_size / max(height, width)
                img = cv2.resize(img, (int(width * scale), int(height * scale)))
                print_with_color(f"    [DEBUG] Resized: {width}x{height} -> {img.shape[1]}x{img.shape[0]}", "yellow")

            # 编码为JPEG并压缩
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]  # 质量70%
            _, buffer = cv2.imencode('.jpg', img, encode_param)
            base64_img = base64.b64encode(buffer).decode('utf-8')

            print_with_color(f"    [DEBUG] Image compressed: {len(base64_img)} chars (was 1952160)", "yellow")

            debug_payload = {
                "model": configs["OPENAI_API_MODEL"],
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                    ]
                }],
                "temperature": configs["TEMPERATURE"],
                "max_tokens": configs["MAX_TOKENS"]
            }
            debug_headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {configs['OPENAI_API_KEY']}"
            }

            debug_resp = requests.post(configs["OPENAI_API_BASE"], headers=debug_headers, json=debug_payload)
            debug_json = debug_resp.json()

            print_with_color(f"    [DEBUG] HTTP Status: {debug_resp.status_code}", "yellow")
            print_with_color(f"    [DEBUG] Response keys: {list(debug_json.keys())}", "yellow")

            if "error" in debug_json:
                print_with_color(f"    [DEBUG] API Error: {debug_json['error']}", "red")
                predictions.append({"action": "wait", "bbox": None})
                continue

            if "choices" in debug_json:
                message = debug_json["choices"][0]["message"]
                content = message.get("content", "")

                print_with_color(f"    [DEBUG] Message keys: {list(message.keys())}", "cyan")
                print_with_color(f"    [DEBUG] Content type: {type(content)}", "cyan")
                print_with_color(f"    [DEBUG] Content length: {len(content) if content else 0}", "cyan")

                if "refusal" in message and message["refusal"]:
                    print_with_color(f"    [DEBUG] Refusal: {message['refusal']}", "red")

                if not content:
                    print_with_color("    [DEBUG] Empty content, saving raw response for analysis", "red")
                    # 保存完整响应用于调试
                    with open("debug_response.json", "w") as f:
                        import json
                        json.dump(debug_json, f, indent=2)
                    print_with_color("    [DEBUG] Saved to debug_response.json", "red")
                    predictions.append({"action": "wait", "bbox": None})
                    continue

                rsp = content
                status = True
            else:
                print_with_color(f"    [DEBUG] No choices in response", "red")
                predictions.append({"action": "wait", "bbox": None})
                continue

        except Exception as e:
            import traceback
            print_with_color(f"    [DEBUG] Exception: {e}", "red")
            print_with_color(f"    [DEBUG] Traceback: {traceback.format_exc()}", "red")
            predictions.append({"action": "wait", "bbox": None})
            continue

        # 调试：打印状态和响应
        print_with_color(f"    API Status: {status}", "cyan")
        print_with_color(f"    Response length: {len(rsp) if isinstance(rsp, str) else 'not string'}", "cyan")

        if status:
            # 解析响应
            res = parse_explore_rsp(rsp)
            act_name = res[0]

            # ===== 核心修改3：映射动作类型 =====
            action_mapped = map_action(act_name)

            # ===== 核心修改4：收集预测结果（不执行） =====
            prediction = {
                "action": action_mapped,
                "bbox": None
            }

            # 获取bbox
            if act_name == "tap" or act_name == "long_press":
                if len(res) > 1:
                    area = res[1]
                    if area <= len(elem_list):
                        tl, br = elem_list[area - 1].bbox
                        prediction["bbox"] = [tl[0], tl[1], br[0], br[1]]

            elif act_name == "swipe":
                if len(res) > 1:
                    area = res[1]
                    if area <= len(elem_list):
                        tl, br = elem_list[area - 1].bbox
                        prediction["bbox"] = [tl[0], tl[1], br[0], br[1]]

            # 如果是input，记录文本
            if act_name == "text" and len(res) > 1:
                prediction["input_value"] = res[1]

            predictions.append(prediction)

            # 更新last_act
            if len(res) > 2:
                last_act = res[-1]

            print_with_color(f"    Pred: {action_mapped}", "magenta")

            # 清理临时文件
            if os.path.exists(labeled_path):
                os.remove(labeled_path)

        else:
            print_with_color(f"    ERROR: {rsp}", "red")
            predictions.append({"action": "wait", "bbox": None})

        # API请求间隔
        time.sleep(configs["REQUEST_INTERVAL"])

    return predictions


def main():
    # ===== 核心修改5：参数解析 =====
    parser = argparse.ArgumentParser(description="AppAgent Offline Executor")
    parser.add_argument("--dataset", "-d", required=True, help="数据集JSON路径")
    parser.add_argument("--output", "-o", default="appagent_results.json", help="输出结果路径")
    args = parser.parse_args()

    # 加载配置
    configs = load_config()

    # 初始化模型
    if configs["MODEL"] == "OpenAI":
        mllm = OpenAIModel(
            base_url=configs["OPENAI_API_BASE"],
            api_key=configs["OPENAI_API_KEY"],
            model=configs["OPENAI_API_MODEL"],
            temperature=configs["TEMPERATURE"],
            max_tokens=configs["MAX_TOKENS"]
        )
    elif configs["MODEL"] == "Qwen":
        mllm = QwenModel(
            api_key=configs["DASHSCOPE_API_KEY"],
            model=configs["QWEN_MODEL"]
        )
    else:
        print_with_color(f"ERROR: Unsupported model {configs['MODEL']}", "red")
        sys.exit(1)

    # ===== 核心修改6：加载数据集 =====
    dataset_path = os.path.abspath(args.dataset)
    dataset_dir = os.path.dirname(dataset_path)  # 数据集所在目录

    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    app_name = dataset.get("app", "unknown")
    testcases = dataset.get("testcases", [])

    # ===== 创建输出目录 =====
    output_path = os.path.abspath(args.output)
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print_with_color(f"Created output directory: {output_dir}", "yellow")

    # ===== 加载已有结果（支持累积追加） =====
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            results = existing_results
            print_with_color(f"Loaded existing results: {len(results['results'])} apps already tested", "yellow")
        except Exception as e:
            print_with_color(f"Failed to load existing results, starting fresh: {e}", "yellow")
            results = {
                "agent_name": "AppAgent",
                "results": []
            }
    else:
        results = {
            "agent_name": "AppAgent",
            "results": []
        }

    # ===== 查找或创建当前app的结果对象 =====
    app_result = None
    for r in results["results"]:
        if r.get("app_tested") == app_name:
            app_result = r
            print_with_color(f"Found existing results for {app_name}, will append tasks", "yellow")
            break

    if app_result is None:
        app_result = {"app_tested": app_name}
        results["results"].append(app_result)
        print_with_color(f"Created new entry for {app_name}", "yellow")

    print_with_color(f"\n{'=' * 60}", "yellow")
    print_with_color(f"App: {app_name}", "yellow")
    print_with_color(f"Testcases: {len(testcases)}", "yellow")
    print_with_color(f"{'=' * 60}\n", "yellow")

    # ===== 核心修改7：处理每个testcase =====
    for testcase in testcases:
        testcase_id = testcase["testcase_id"]
        task_desc = testcase["testcase_desc"]

        print_with_color(f"\nTestcase {testcase_id}: {task_desc[:60]}...", "yellow")

        task_results = []

        # 处理clean trace
        if "clean" in testcase:
            clean_trace = testcase["clean"]["trace"]
            print_with_color(f"  Clean trace: {len(clean_trace)} steps", "yellow")

            trace_output = process_trace(clean_trace, task_desc, mllm, configs, dataset_dir)

            task_results.append({
                "type": "clean",
                "trace_output": trace_output
            })

        # 处理noise traces
        if "noise" in testcase:
            for noise_item in testcase["noise"]:
                noise_type = noise_item["type"]
                noise_trace = noise_item["trace"]
                print_with_color(f"  {noise_type} trace: {len(noise_trace)} steps", "yellow")

                trace_output = process_trace(noise_trace, task_desc, mllm, configs, dataset_dir)

                task_results.append({
                    "type": noise_type,
                    "trace_output": trace_output
                })

        # ===== 核心修改8：保存结果（追加到app对象中） =====
        app_result[f"task_{testcase_id}"] = task_results

        # 每个testcase结束立即保存（indent=4，bbox单行）
        json_str = json.dumps(results, indent=4, ensure_ascii=False)
        json_str = format_bbox_single_line(json_str)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_str)

        print_with_color(f"  ✓ Saved to {output_path}", "green")

    print_with_color(f"\n{'=' * 60}", "yellow")
    print_with_color(f"✓ 完成！结果保存到: {output_path}", "yellow")
    print_with_color(f"{'=' * 60}\n", "yellow")


if __name__ == "__main__":
    main()
