#!/usr/bin/env python3
"""
在JSON文件的指定testcase中插入噪声trace
支持自定义噪声内容
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Any


def generate_noise_checkpoints(
        noise_info: List[Dict[str, Any]],
        noise_type: str,
        task_name: str,
        insert_point: int
) -> List[Dict[str, Any]]:
    """
    生成噪声checkpoint列表

    Args:
        noise_info: 噪声信息列表，每个元素包含 action, bbox, input_value(可选)
        noise_type: 噪声类型
        task_name: 任务名称
        insert_point: 插入点（用于文件名编号）

    Returns:
        噪声checkpoint列表
    """
    noise_checkpoints = []

    for i, info in enumerate(noise_info):
        checkpoint_id = insert_point + i

        checkpoint = {
            "checkpoint_id": checkpoint_id,
            "screenshot_path": f"./{task_name}_{noise_type}_{insert_point + i}.png",
            "xml_path": f"./{task_name}_{noise_type}_{insert_point + i}.xml",
            "action": info["action"]
        }
        if "bbox" in info:
            checkpoint["bbox"] = info["bbox"]
        # 如果有input_value，添加进去
        if "input_value" in info:
            checkpoint["input_value"] = info["input_value"]

        noise_checkpoints.append(checkpoint)

    return noise_checkpoints


def insert_noise_to_trace(
        clean_trace: List[Dict[str, Any]],
        noise_info: List[Dict[str, Any]],
        noise_type: str,
        insert_point: int,
        resume_point: int,
        task_name: str
) -> Dict[str, Any]:
    """
    在标准trace中插入噪声
    """
    noise_length = len(noise_info)

    # 获取insert_point之前的标准trace部分
    prefix_trace = []
    for checkpoint in clean_trace:
        if checkpoint["checkpoint_id"] < insert_point:
            new_checkpoint = checkpoint.copy()
            prefix_trace.append(new_checkpoint)

    # 生成噪声checkpoint
    noise_checkpoints = generate_noise_checkpoints(
        noise_info, noise_type, task_name, insert_point
    )

    # 获取resume_point之后的标准trace部分
    resume_trace = []
    for checkpoint in clean_trace:
        if checkpoint["checkpoint_id"] >= resume_point:
            new_checkpoint = checkpoint.copy()
            new_checkpoint["checkpoint_id"] = len(prefix_trace) + noise_length + (
                        checkpoint["checkpoint_id"] - resume_point + 1)
            resume_trace.append(new_checkpoint)

    # 重新编号前缀部分
    for i, checkpoint in enumerate(prefix_trace):
        checkpoint["checkpoint_id"] = i + 1

    # 合并
    new_trace = prefix_trace + noise_checkpoints + resume_trace

    return {
        "type": noise_type,
        "trace": new_trace
    }


def process_json_file(
        json_path: str,
        testcase_id: int,
        noise_info: List[Dict[str, Any]],
        noise_type: str,
        insert_point: int,
        resume_point: int
):
    """
    处理JSON文件，在指定testcase中插入噪声trace
    """
    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 找到指定的testcase
    target_testcase = None
    for testcase in data["testcases"]:
        if testcase["testcase_id"] == testcase_id:
            target_testcase = testcase
            break

    if target_testcase is None:
        print(f"错误: 未找到 testcase_id={testcase_id}")
        return

    # 获取标准trace
    clean_trace = target_testcase["clean"]["trace"]

    # 获取任务名称
    first_checkpoint = clean_trace[0]
    screenshot_name = Path(first_checkpoint["screenshot_path"]).stem
    task_name = "_".join(screenshot_name.split("_")[:-2])

    print(f"任务名称: {task_name}")
    print(f"Testcase ID: {testcase_id}")
    print(f"标准trace长度: {len(clean_trace)}")
    print(f"噪声类型: {noise_type}, 噪声长度: {len(noise_info)}")
    print(f"插入点: {insert_point}, 回拼点: {resume_point}")

    # 生成噪声trace
    noise_trace = insert_noise_to_trace(
        clean_trace, noise_info, noise_type, insert_point, resume_point, task_name
    )

    # 添加到testcase
    if "noise" not in target_testcase:
        target_testcase["noise"] = []

    target_testcase["noise"].append(noise_trace)

    # 保存到原文件
    # 先序列化为字符串，处理bbox不换行
    json_str = json.dumps(data, ensure_ascii=False, indent=4)

    # 使用正则表达式将bbox数组压缩到一行
    json_str = re.sub(
        r'"bbox":\s*\[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\]',
        r'"bbox": [\1, \2, \3, \4]',
        json_str
    )

    with open(json_path, 'w', encoding='utf-8') as f:
        f.write(json_str)

    print(f"\n成功添加噪声trace!")
    print(f"噪声trace长度: {len(noise_trace['trace'])}")

    # 打印新生成的trace结构
    print("\n生成的噪声trace结构:")
    for cp in noise_trace["trace"]:
        print(f"  checkpoint {cp['checkpoint_id']}: {cp['screenshot_path']}")


if __name__ == "__main__":

    json_path = "./traces_new/taobao-16/checkpoint_taobao.json" # <json文件>
    testcase_id = 1 # <testcase_id>

    noise_type = "EA-T" # <噪声类型>

    noise_info = [ # 自定义噪声信息
        {
            "action": "wait"
        }
    ] # 长度与noise length匹配

    insert_point = 3 # <插入点>（在标准trace的第几个checkpoint前插入噪声）
    resume_point = 3 # <回拼点>（从标准trace的第几个checkpoint开始拼接回原trace）

    process_json_file(json_path, testcase_id, noise_info, noise_type, insert_point, resume_point)

#广告弹窗/遮挡
# {
#     "action": "click",
#     "bbox": [330,1195,390,1255]
# }


#滑动
# {
#     "action": "scroll",
#     "bbox": [100, 64, 616, 152],
#     "input_value": "right"
# },
# {
#     "action": "click",
#     "bbox": [192, 64, 284, 152]
# }

#全屏广告
# {
#     "action": "click",
#     "bbox": [420, 140, 690, 270]
# }