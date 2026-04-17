#!/usr/bin/env python3
"""
合并结果文件：将每个task单独一个对象的结构，转换为按app分组的结构

输入格式：
{
  "agent_name": "AppAgent",
  "results": [
    {"app_tested": "douyin", "task_1": [...]},
    {"app_tested": "douyin", "task_2": [...]},
    {"app_tested": "taobao", "task_1": [...]}
  ]
}

输出格式：
{
  "agent_name": "AppAgent",
  "results": [
    {
      "app_tested": "douyin",
      "task_1": [...],
      "task_2": [...]
    },
    {
      "app_tested": "taobao",
      "task_1": [...]
    }
  ]
}
"""

import json
import argparse
import os
import re


def format_bbox_single_line(json_str):
    """将bbox数组格式化为单行"""
    # 匹配 "bbox": [ ... ] 并将其压缩为单行
    pattern = r'"bbox":\s*\[\s*([\d\.\-\s,]+?)\s*\]'

    def replace_bbox(match):
        numbers = match.group(1)
        # 移除多余空格和换行，保留数字和逗号
        numbers = re.sub(r'\s+', '', numbers)
        return f'"bbox": [{numbers}]'

    return re.sub(pattern, replace_bbox, json_str)


def merge_results(input_path, output_path=None):
    """合并结果文件"""

    # 读取输入文件
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    agent_name = data.get("agent_name", "AppAgent")
    old_results = data.get("results", [])

    # 按app分组
    app_dict = {}  # {app_name: {task_1: [...], task_2: [...]}}

    for item in old_results:
        app_name = item.get("app_tested")
        if not app_name:
            continue

        # 初始化app条目
        if app_name not in app_dict:
            app_dict[app_name] = {"app_tested": app_name}

        # 合并所有task字段
        for key, value in item.items():
            if key.startswith("task_"):
                app_dict[app_name][key] = value

    # 转换为列表（保持app的顺序）
    new_results = list(app_dict.values())

    # 构建新结构
    new_data = {
        "agent_name": agent_name,
        "results": new_results
    }

    # 确定输出路径
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_merged{ext}"

    # 保存（indent=4，bbox单行）
    json_str = json.dumps(new_data, indent=4, ensure_ascii=False)
    json_str = format_bbox_single_line(json_str)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(json_str)

    # 打印统计
    print(f"合并完成！")
    print(f"  - 原始条目数: {len(old_results)}")
    print(f"  - 合并后app数: {len(new_results)}")
    for app in new_results:
        task_count = sum(1 for k in app.keys() if k.startswith("task_"))
        print(f"    - {app['app_tested']}: {task_count} tasks")
    print(f"  - 输出文件: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合并AppAgent结果文件")
    parser.add_argument("--input", "-i", required=True, help="输入结果文件路径")
    parser.add_argument("--output", "-o", help="输出文件路径（默认为输入文件名_merged.json）")
    args = parser.parse_args()

    merge_results(args.input, args.output)
