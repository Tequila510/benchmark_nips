"""
评估指标计算模块 - 多App综合版本
实现四个核心指标：AAR、NII、TCR、RDR
支持统计所有app的综合结果

公式定义：
- AAR = (1 / Σ T_i) * Σ I[â_t = a_t]
- NII_d = (1 - NPA_d) * 100%  (噪声干扰指数，越高干扰越强)
- TCR = (1 / N) * Σ I[Π I[â_t = a_t] = 1]
- RDR_d = (N_clean_completed - N_perturbed_completed) / N_clean_completed * 100%
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
import json
from collections import defaultdict
import argparse
import os


@dataclass
class CheckpointResult:
    """单个checkpoint的预测结果"""
    app_name: str
    testcase_id: int
    noise_type: str  # "clean" 或具体的噪声类型如 "EO", "VE" 等
    checkpoint_id: int
    predicted_action: str
    ground_truth_action: str
    predicted_bbox: Optional[List[int]] = None
    ground_truth_bbox: Optional[List[int]] = None
    predicted_input_value: Optional[str] = None
    ground_truth_input_value: Optional[str] = None
    is_correct: bool = False
    is_noise_checkpoint: bool = False  # 是否为噪声checkpoint


@dataclass
class TraceResult:
    """一条trace的完整结果"""
    agent_name: str
    app_name: str
    testcase_id: int
    testcase_desc: str
    noise_type: str  # "clean" 或具体的噪声类型
    checkpoints: List[CheckpointResult] = field(default_factory=list)
    is_completed: bool = False  # 是否所有checkpoint都正确
    noise_indices: List[int] = field(default_factory=list)  # 噪声checkpoint位置


# ============================================================
# Metric 1: AAR (Action Accuracy Rate) - 动作准确率
# ============================================================
def calculate_aar(results: List[TraceResult]) -> float:
    """
    Metric 1: Action Accuracy Rate (AAR) - 动作准确率
    公式：AAR = (1 / Σ T_i) * Σ I[â_t = a_t]
    """
    total_checkpoints = 0
    correct_predictions = 0
    for trace in results:
        for checkpoint in trace.checkpoints:
            total_checkpoints += 1
            if checkpoint.is_correct:
                correct_predictions += 1
    if total_checkpoints == 0:
        return 0.0
    return correct_predictions / total_checkpoints


def calculate_aar_by_group(
    results: List[TraceResult],
    group_by: str = "noise_type"
) -> Dict[str, float]:
    """
    按指定维度分组计算AAR
    Args:
        results: Trace结果列表
        group_by: 分组维度，可选 "noise_type", "testcase_id", "app_name"
    """
    grouped = defaultdict(list)
    for trace in results:
        if group_by == "noise_type":
            key = trace.noise_type
        elif group_by == "testcase_id":
            key = trace.testcase_id
        elif group_by == "app_name":
            key = trace.app_name
        else:
            key = "all"
        grouped[key].append(trace)

    aar_results = {}
    for key, traces in grouped.items():
        aar_results[key] = calculate_aar(traces)
    return aar_results


# ============================================================
# Metric 2: NII (Noise Interference Index) - 噪声干扰指数
# ============================================================
def calculate_nii(results: List[TraceResult]) -> Dict[str, float]:
    """
    Metric 2: Noise Interference Index (NII) - 噪声干扰指数

    只统计噪声checkpoint，数值越高表示干扰越强
    公式：NII_d = (1 - NPA_d) * 100%
    其中 NPA_d = 噪声点准确率

    输出：按噪声类型分类，所有app累计
    """
    noise_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for trace in results:
        # 跳过 clean trace
        if trace.noise_type == "clean":
            continue

        noise_type = trace.noise_type
        noise_indices = trace.noise_indices

        # 只统计噪声checkpoint
        for idx in noise_indices:
            if idx < len(trace.checkpoints):
                checkpoint = trace.checkpoints[idx]
                noise_stats[noise_type]["total"] += 1
                if checkpoint.is_correct:
                    noise_stats[noise_type]["correct"] += 1

    nii_results = {}
    for noise_type, stats in noise_stats.items():
        if stats["total"] > 0:
            npa = stats["correct"] / stats["total"]  # 噪声点准确率
            nii_results[noise_type] = (1 - npa) * 100  # 转为干扰指数百分比
        else:
            nii_results[noise_type] = 0.0

    return nii_results


def calculate_npa(results: List[TraceResult]) -> Dict[str, float]:
    """
    Noise Point Accuracy (NPA) - 噪声点准确率

    只统计噪声checkpoint的准确率
    公式：NPA_d = 正确噪声点数 / 总噪声点数
    """
    noise_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for trace in results:
        if trace.noise_type == "clean":
            continue

        noise_type = trace.noise_type
        noise_indices = trace.noise_indices

        for idx in noise_indices:
            if idx < len(trace.checkpoints):
                checkpoint = trace.checkpoints[idx]
                noise_stats[noise_type]["total"] += 1
                if checkpoint.is_correct:
                    noise_stats[noise_type]["correct"] += 1

    npa_results = {}
    for noise_type, stats in noise_stats.items():
        if stats["total"] > 0:
            npa_results[noise_type] = stats["correct"] / stats["total"]
        else:
            npa_results[noise_type] = 0.0

    return npa_results


# ============================================================
# Metric 3: TCR (Task Completion Rate) - 任务完成率
# ============================================================
def calculate_tcr(results: List[TraceResult]) -> float:
    """
    Metric 3: Task Completion Rate (TCR) - 任务完成率
    公式：TCR = (1 / N) * Σ I[Π I[â_t = a_t] = 1]
    """
    total_traces = len(results)
    completed_traces = 0
    for trace in results:
        if trace.is_completed:
            completed_traces += 1
    if total_traces == 0:
        return 0.0
    return completed_traces / total_traces


def calculate_tcr_by_group(
    results: List[TraceResult],
    group_by: str = "noise_type"
) -> Dict[str, float]:
    """按指定维度分组计算TCR"""
    grouped = defaultdict(list)
    for trace in results:
        if group_by == "noise_type":
            key = trace.noise_type
        elif group_by == "testcase_id":
            key = trace.testcase_id
        elif group_by == "app_name":
            key = trace.app_name
        else:
            key = "all"
        grouped[key].append(trace)

    tcr_results = {}
    for key, traces in grouped.items():
        tcr_results[key] = calculate_tcr(traces)
    return tcr_results


# ============================================================
# Metric 4: RDR (Robustness Degradation Rate) - 鲁棒性下降度
# ============================================================
def calculate_rdr(
    clean_results: List[TraceResult],
    perturbed_results: List[TraceResult]
) -> Dict[str, float]:
    """
    Metric 4: Robustness Degradation Rate (RDR) - 鲁棒性下降度

    只统计在 clean 环境下已完成的 trace，看这些 trace 在噪声环境下是否还能完成
    公式：RDR_d = (N_clean_completed - N_perturbed_completed) / N_clean_completed * 100%

    其中：
    - N_clean_completed = clean 环境下完成的 trace 数量
    - N_perturbed_completed = 同一批 trace 在噪声环境下完成的数量
    """
    # Step 1: 找出 clean 环境下完成的 trace（用 app_name + testcase_id 标识）
    clean_completed_keys = set()
    for trace in clean_results:
        if trace.is_completed:
            key = (trace.app_name, trace.testcase_id)
            clean_completed_keys.add(key)

    num_clean_completed = len(clean_completed_keys)
    if num_clean_completed == 0:
        return {"error": "No clean traces completed, cannot calculate RDR"}

    # Step 2: 按噪声类型分组，只统计 clean 已完成的 trace
    perturbed_by_type = defaultdict(list)
    for trace in perturbed_results:
        key = (trace.app_name, trace.testcase_id)
        if key in clean_completed_keys:
            perturbed_by_type[trace.noise_type].append(trace)

    # Step 3: 计算每种噪声类型的 RDR
    rdr_results = {
        "clean_completed": num_clean_completed  # 记录基准数量
    }
    for noise_type, traces in perturbed_by_type.items():
        # 统计该噪声类型下完成的 trace 数量
        num_perturbed_completed = sum(1 for t in traces if t.is_completed)
        rdr = (num_clean_completed - num_perturbed_completed) / num_clean_completed * 100
        rdr_results[noise_type] = rdr
        rdr_results[f"{noise_type}_completed"] = num_perturbed_completed

    # 计算平均 RDR
    valid_rdr_values = [v for k, v in rdr_results.items()
                        if not k.endswith("_completed") and k != "clean_completed"
                        and isinstance(v, (int, float))]
    if valid_rdr_values:
        rdr_results["average"] = sum(valid_rdr_values) / len(valid_rdr_values)

    return rdr_results


# ============================================================
# 综合评估函数
# ============================================================
def evaluate_all_apps(
    agent_name: str,
    ground_truth_files: List[str],
    prediction_path: str,
    metrics: Optional[Set[str]] = None,
    group_by: Optional[str] = None
) -> Dict:
    """
    读取所有app的ground truth和预测结果，计算综合指标
    Args:
        agent_name: Agent名称
        ground_truth_files: 所有app的ground truth JSON文件路径列表
        prediction_path: 预测结果JSON文件路径
        metrics: 要计算的指标集合，如 {"AAR", "NII", "TCR", "RDR"}
        group_by: 分组维度，可选 "noise_type", "testcase_id", "app_name"
    """
    if metrics is None:
        metrics = {"AAR", "NII", "TCR", "RDR"}

    # 读取预测结果
    with open(prediction_path, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)

    # 构建预测结果索引: {(app_name, task_key, noise_type): trace_output}
    pred_index = _build_prediction_index(pred_data)

    # 构建结果列表
    clean_results = []
    perturbed_results = []
    app_names = []

    # 处理每个app的ground truth
    for gt_path in ground_truth_files:
        if not os.path.exists(gt_path):
            print(f"Warning: Ground truth file not found: {gt_path}")
            continue
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        app_name = gt_data.get("app", "unknown")
        app_names.append(app_name)

        # 处理每个testcase
        for testcase in gt_data.get("testcases", []):
            testcase_id = testcase["testcase_id"]
            testcase_desc = testcase["testcase_desc"]

            # 处理clean trace
            if "clean" in testcase:
                clean_trace = testcase["clean"]["trace"]
                pred_key = (app_name, f"task_{testcase_id}", "clean")
                pred_trace = pred_index.get(pred_key, [])
                trace_result = _process_trace(
                    agent_name, app_name, testcase_id, testcase_desc,
                    "clean", clean_trace, pred_trace, noise_indices=[]
                )
                clean_results.append(trace_result)

            # 处理noise traces
            if "noise" in testcase:
                for noise_item in testcase["noise"]:
                    noise_type = noise_item["type"]
                    noise_trace = noise_item["trace"]
                    # 读取噪声位置标注（如果有的话）
                    noise_indices = noise_item.get("noise_indices", [])
                    pred_key = (app_name, f"task_{testcase_id}", noise_type)
                    pred_trace = pred_index.get(pred_key, [])
                    trace_result = _process_trace(
                        agent_name, app_name, testcase_id, testcase_desc,
                        noise_type, noise_trace, pred_trace, noise_indices=noise_indices
                    )
                    perturbed_results.append(trace_result)

    # 合并所有结果
    all_results = clean_results + perturbed_results

    # 根据参数计算指标
    result = {
        "agent_name": agent_name,
        "apps_evaluated": app_names,
        "total_apps": len(app_names),
        "total_traces": len(all_results),
        "total_checkpoints": sum(len(t.checkpoints) for t in all_results),
    }

    # AAR
    if "AAR" in metrics:
        result["AAR"] = calculate_aar(all_results)
        if group_by:
            result[f"AAR_by_{group_by}"] = calculate_aar_by_group(all_results, group_by)

    # NII
    if "NII" in metrics:
        result["NII"] = calculate_nii(perturbed_results)
        result["NPA"] = calculate_npa(perturbed_results)

    # TCR
    if "TCR" in metrics:
        result["TCR_clean"] = calculate_tcr(clean_results)
        result["TCR_perturbed"] = calculate_tcr(perturbed_results)
        result["TCR_all"] = calculate_tcr(all_results)
        if group_by:
            result[f"TCR_by_{group_by}"] = calculate_tcr_by_group(all_results, group_by)

    # RDR
    if "RDR" in metrics:
        result["RDR"] = calculate_rdr(clean_results, perturbed_results)

    return result


def _build_prediction_index(pred_data: Dict) -> Dict:
    """
    构建预测结果索引
    输出: {(app_name, task_key, noise_type): trace_output}
    """
    index = {}
    for app_result in pred_data.get("results", []):
        app_name = app_result.get("app_tested", "unknown")
        # 遍历所有task
        for key, value in app_result.items():
            if key.startswith("task_"):
                task_key = key
                # value是数组，每个元素有type和trace_output
                if isinstance(value, list):
                    for item in value:
                        noise_type = item.get("type", "unknown")
                        trace_output = item.get("trace_output", [])
                        index[(app_name, task_key, noise_type)] = trace_output
    return index


def _extract_noise_indices_from_trace(gt_trace: List[Dict], noise_type: str) -> List[int]:
    """
    从 screenshot_path 文件名中自动提取噪声checkpoint索引

    规则：
    - 文件名包含噪声类型标记（如 _VE_, _EO_, _EA_, _SC_）则是噪声checkpoint
    - 文件名包含 _b_ 表示 baseline/clean，不是噪声

    Args:
        gt_trace: ground truth trace
        noise_type: 噪声类型（如 "VE", "EO", "EA", "SC"）

    Returns:
        噪声checkpoint的索引列表
    """
    if noise_type == "clean":
        return []

    noise_indices = []
    noise_marker = f"_{noise_type}_"

    for i, checkpoint in enumerate(gt_trace):
        screenshot_path = checkpoint.get("screenshot_path", "")
        # 检查文件名是否包含噪声类型标记
        if noise_marker in screenshot_path:
            noise_indices.append(i)

    return noise_indices


def _process_trace(
    agent_name: str,
    app_name: str,
    testcase_id: int,
    testcase_desc: str,
    noise_type: str,
    gt_trace: List[Dict],
    pred_trace: List[Dict],
    noise_indices: List[int] = None
) -> TraceResult:
    """处理单条trace，构建TraceResult对象"""
    # 如果没有提供noise_indices，从文件名自动提取
    if noise_indices is None or len(noise_indices) == 0:
        noise_indices = _extract_noise_indices_from_trace(gt_trace, noise_type)

    trace_result = TraceResult(
        agent_name=agent_name,
        app_name=app_name,
        testcase_id=testcase_id,
        testcase_desc=testcase_desc,
        noise_type=noise_type,
        noise_indices=noise_indices
    )

    # 处理每个checkpoint
    all_correct = True
    for i, gt_checkpoint in enumerate(gt_trace):
        checkpoint_id = gt_checkpoint["checkpoint_id"]
        gt_action = gt_checkpoint["action"]
        gt_bbox = gt_checkpoint.get("bbox")
        gt_input_value = gt_checkpoint.get("input_value")

        # 获取预测结果
        pred_action = None
        pred_bbox = None
        pred_input_value = None
        if i < len(pred_trace):
            pred_item = pred_trace[i]
            pred_action = pred_item.get("action")
            pred_input_value = pred_item.get("input_value")

            # bbox可能是字符串或list
            pred_bbox_raw = pred_item.get("bbox")
            if isinstance(pred_bbox_raw, str):
                # 尝试解析字符串格式的bbox
                try:
                    import ast
                    pred_bbox = ast.literal_eval(pred_bbox_raw)
                except:
                    pred_bbox = None
            else:
                pred_bbox = pred_bbox_raw

        # 判断是否正确（细粒度匹配）
        is_correct = _check_correctness(
            pred_action, gt_action, pred_bbox, gt_bbox,
            pred_input_value, gt_input_value
        )
        if not is_correct:
            all_correct = False

        # 判断是否为噪声checkpoint
        is_noise_checkpoint = i in noise_indices

        checkpoint_result = CheckpointResult(
            app_name=app_name,
            testcase_id=testcase_id,
            noise_type=noise_type,
            checkpoint_id=checkpoint_id,
            predicted_action=pred_action or "UNKNOWN",
            ground_truth_action=gt_action,
            predicted_bbox=pred_bbox,
            ground_truth_bbox=gt_bbox,
            predicted_input_value=pred_input_value,
            ground_truth_input_value=gt_input_value,
            is_correct=is_correct,
            is_noise_checkpoint=is_noise_checkpoint
        )
        trace_result.checkpoints.append(checkpoint_result)

    trace_result.is_completed = all_correct
    return trace_result


def _check_correctness(
    pred_action: Optional[str],
    gt_action: str,
    pred_bbox: Optional[List[int]],
    gt_bbox: Optional[List[int]],
    pred_input_value: Optional[str],
    gt_input_value: Optional[str]
) -> bool:
    """
    细粒度判断预测是否正确
    规则：
    1. action不匹配 → 直接判错
    2. action匹配：
        - click: 计算pred的bbox中心点是否落在GT的bbox中
        - scroll: 先比对input_value，input_value对再比对bbox中心点
        - input: 先比对input_value，input_value对则判对（若有bbox再比对）
        - wait/finish: 直接判对
    """
    # Step 1: action不匹配直接判错
    if pred_action != gt_action:
        return False

    # Step 2: action匹配，根据类型进一步判断
    if gt_action == "click":
        # click: pred的中心点落在GT bbox内即可
        return _compare_bbox_for_click(pred_bbox, gt_bbox)
    elif gt_action == "scroll":
        # scroll先比对input_value（方向）
        if pred_input_value != gt_input_value:
            return False
        # input_value匹配后，再比对bbox（中心点判定）
        return _compare_bbox_for_click(pred_bbox, gt_bbox)
    elif gt_action == "input":
        # input先比对input_value
        if pred_input_value != gt_input_value:
            return False
        # input_value匹配后，如果pred有bbox则需要比对（中心点判定）
        if pred_bbox is not None:
            return _compare_bbox_for_click(pred_bbox, gt_bbox)
        # 预测没有bbox，只要action和input_value对了就算对
        return True
    elif gt_action in ["wait", "finish"]:
        # wait和finish直接判对
        return True
    else:
        # 其他未知action类型，只匹配action
        return True


def _compare_bbox(
    pred_bbox: Optional[List[int]],
    gt_bbox: Optional[List[int]]
) -> bool:
    """
    比较两个bbox是否完全相等
    规则：
    - 如果gt_bbox为None，则pred_bbox任意值都算对
    - 如果gt_bbox不为None，pred_bbox为None则判错
    - 两个bbox完全相等才算对
    """
    if gt_bbox is None:
        return True
    if pred_bbox is None:
        return False
    return pred_bbox == gt_bbox


def _compare_bbox_for_click(
    pred_bbox: Optional[List[int]],
    gt_bbox: Optional[List[int]]
) -> bool:
    """
    专门用于click动作的bbox比对
    规则：
    - 如果gt_bbox为None，则pred_bbox任意值都算对
    - 如果gt_bbox不为None，pred_bbox为None则判错
    - 计算pred_bbox的中心点，判断是否落在gt_bbox区域内
    """
    if gt_bbox is None:
        return True
    if pred_bbox is None or len(pred_bbox) == 0:
        return False

    # 计算pred_bbox的中心点
    pred_center_x = (pred_bbox[0] + pred_bbox[2]) / 2
    pred_center_y = (pred_bbox[1] + pred_bbox[3]) / 2

    # GT bbox的边界
    gt_x1, gt_y1, gt_x2, gt_y2 = gt_bbox

    # 判断pred中心点是否落在GT bbox内
    return (gt_x1 <= pred_center_x <= gt_x2) and (gt_y1 <= pred_center_y <= gt_y2)


def print_metrics(metrics: Dict):
    """打印指标结果"""
    print("\n" + "="*70)
    print(f"评估结果 - {metrics['agent_name']}")
    print(f"评估的App数量: {metrics['total_apps']}")
    print(f"App列表: {', '.join(metrics['apps_evaluated'])}")
    print("="*70)
    print(f"总Trace数: {metrics['total_traces']}")
    print(f"总Checkpoint数: {metrics['total_checkpoints']}")
    print("-"*70)

    if "AAR" in metrics:
        print(f"AAR (动作准确率): {metrics['AAR']:.6f} ({metrics['AAR']*100:.3f}%)")
        for key in metrics:
            if key.startswith("AAR_by_"):
                print(f" {key}:")
                for k, v in metrics[key].items():
                    print(f" {k}: {v:.6f} ({v*100:.3f}%)")

    if "NII" in metrics:
        print("-"*70)
        print("NII (噪声干扰指数) - 越高干扰越强:")
        if isinstance(metrics['NII'], dict):
            # 按干扰强度排序（从高到低）
            sorted_nii = sorted(metrics['NII'].items(), key=lambda x: x[1], reverse=True)
            for noise_type, nii in sorted_nii:
                print(f" {noise_type}: {nii:.3f}%")
        print("-"*70)
        print("NPA (噪声点准确率):")
        if isinstance(metrics.get('NPA'), dict):
            for noise_type, npa in metrics['NPA'].items():
                print(f" {noise_type}: {npa:.6f} ({npa*100:.3f}%)")

    if "TCR_clean" in metrics:
        print("-"*70)
        print(f"TCR_clean (标准环境任务完成率): {metrics['TCR_clean']:.6f} ({metrics['TCR_clean']*100:.3f}%)")
        print(f"TCR_perturbed (扰动环境任务完成率): {metrics['TCR_perturbed']:.6f} ({metrics['TCR_perturbed']*100:.3f}%)")
        print(f"TCR_all (全部任务完成率): {metrics['TCR_all']:.6f} ({metrics['TCR_all']*100:.3f}%)")

    if "RDR" in metrics:
        print("-"*70)
        print("RDR (鲁棒性下降度) - 基于 clean 已完成的 trace:")
        if isinstance(metrics['RDR'], dict):
            rdr_data = metrics['RDR']
            # 打印基准信息
            if "clean_completed" in rdr_data:
                print(f" Clean 环境完成的 trace 数: {rdr_data['clean_completed']}")
            # 打印每种噪声类型的 RDR（按值排序，从高到低）
            noise_rdrs = [(k, v) for k, v in rdr_data.items()
                         if not k.endswith("_completed") and k not in ["clean_completed", "average", "error"]
                         and isinstance(v, (int, float))]
            if noise_rdrs:
                sorted_rdrs = sorted(noise_rdrs, key=lambda x: x[1], reverse=True)
                for noise_type, rdr in sorted_rdrs:
                    completed_key = f"{noise_type}_completed"
                    completed_num = rdr_data.get(completed_key, "N/A")
                    print(f" {noise_type}: {rdr:.3f}% (完成数: {completed_num})")
            # 打印平均值
            if "average" in rdr_data:
                print(f" 平均 RDR: {rdr_data['average']:.3f}%")
            # 打印错误信息
            if "error" in rdr_data:
                print(f" 错误: {rdr_data['error']}")

    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="计算所有App的综合评估指标")
    parser.add_argument("prediction", help="预测结果JSON文件路径")
    parser.add_argument("--agent", "-a", default=None, help="Agent名称（默认从预测文件读取）")
    parser.add_argument("--metrics", "-m", nargs="+",
                        choices=["AAR", "NII", "TCR", "RDR"],
                        help="要计算的指标（默认全部）")
    parser.add_argument("--group-by", "-g",
                        choices=["noise_type", "testcase_id", "app_name"],
                        help="分组维度")
    parser.add_argument("--output", "-o", default="metrics_results.txt",
                        help="输出文件路径")
    args = parser.parse_args()

    # ============================================================
    # 在这里写死所有app的ground truth文件路径
    # ============================================================
    GROUND_TRUTH_FILES = [
        # 示例：
        "../data/douyin-20/checkpoint_douyin.json",
        "../data/taobao-16/checkpoint_taobao.json",
        "../data/weipinhui-15/checkpoint_weipinhui.json",
        "../data/wangyiyun-16/checkpoint_wangyiyun.json",
    ]

    # 从预测文件读取agent_name
    if args.agent is None:
        with open(args.prediction, 'r', encoding='utf-8') as f:
            pred_data = json.load(f)
        agent_name = pred_data.get("agent_name", "Unknown_Agent")
    else:
        agent_name = args.agent

    metrics_set = set(args.metrics) if args.metrics else None
    metrics = evaluate_all_apps(
        agent_name,
        GROUND_TRUTH_FILES,
        args.prediction,
        metrics=metrics_set,
        group_by=args.group_by
    )
    print_metrics(metrics)

    # 保存结果
    output_path = args.output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"评估结果 - {metrics['agent_name']}\n")
        f.write(f"评估的App数量: {metrics['total_apps']}\n")
        f.write(f"App列表: {', '.join(metrics['apps_evaluated'])}\n")
        f.write("="*70 + "\n")
        f.write(f"总Trace数: {metrics['total_traces']}\n")
        f.write(f"总Checkpoint数: {metrics['total_checkpoints']}\n")
        f.write("-"*70 + "\n")

        if "AAR" in metrics:
            f.write(f"AAR (动作准确率): {metrics['AAR']:.6f} ({metrics['AAR']*100:.3f}%)\n")
            for key in metrics:
                if key.startswith("AAR_by_"):
                    f.write(f" {key}:\n")
                    for k, v in metrics[key].items():
                        f.write(f" {k}: {v:.6f} ({v*100:.3f}%)\n")

        if "NII" in metrics:
            f.write("-"*70 + "\n")
            f.write("NII (噪声干扰指数) - 越高干扰越强:\n")
            if isinstance(metrics['NII'], dict):
                sorted_nii = sorted(metrics['NII'].items(), key=lambda x: x[1], reverse=True)
                for noise_type, nii in sorted_nii:
                    f.write(f" {noise_type}: {nii:.3f}%\n")
            f.write("-"*70 + "\n")
            f.write("NPA (噪声点准确率):\n")
            if isinstance(metrics.get('NPA'), dict):
                for noise_type, npa in metrics['NPA'].items():
                    f.write(f" {noise_type}: {npa:.6f} ({npa*100:.3f}%)\n")

        if "TCR_clean" in metrics:
            f.write("-"*70 + "\n")
            f.write(f"TCR_clean (标准环境任务完成率): {metrics['TCR_clean']:.6f} ({metrics['TCR_clean']*100:.3f}%)\n")
            f.write(f"TCR_perturbed (扰动环境任务完成率): {metrics['TCR_perturbed']:.6f} ({metrics['TCR_perturbed']*100:.3f}%)\n")
            f.write(f"TCR_all (全部任务完成率): {metrics['TCR_all']:.6f} ({metrics['TCR_all']*100:.3f}%)\n")

        if "RDR" in metrics:
            f.write("-"*70 + "\n")
            f.write("RDR (鲁棒性下降度) - 基于 clean 已完成的 trace:\n")
            if isinstance(metrics['RDR'], dict):
                rdr_data = metrics['RDR']
                # 写入基准信息
                if "clean_completed" in rdr_data:
                    f.write(f" Clean 环境完成的 trace 数: {rdr_data['clean_completed']}\n")
                # 写入每种噪声类型的 RDR
                noise_rdrs = [(k, v) for k, v in rdr_data.items()
                             if not k.endswith("_completed") and k not in ["clean_completed", "average", "error"]
                             and isinstance(v, (int, float))]
                if noise_rdrs:
                    sorted_rdrs = sorted(noise_rdrs, key=lambda x: x[1], reverse=True)
                    for noise_type, rdr in sorted_rdrs:
                        completed_key = f"{noise_type}_completed"
                        completed_num = rdr_data.get(completed_key, "N/A")
                        f.write(f" {noise_type}: {rdr:.3f}% (完成数: {completed_num})\n")
                # 写入平均值
                if "average" in rdr_data:
                    f.write(f" 平均 RDR: {rdr_data['average']:.3f}%\n")
                # 写入错误信息
                if "error" in rdr_data:
                    f.write(f" 错误: {rdr_data['error']}\n")

        f.write("="*70 + "\n")

    print(f"指标结果已保存到: {output_path}")
