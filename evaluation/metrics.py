"""
评估指标计算模块
实现四个核心指标：AAR、PSSR、TCR、RDR

公式定义：
- AAR = (1 / Σ T_i) * Σ I[â_t = a_t]
- PSSR_d = (1 / |S_d|) * Σ I[â_t = a_t]
- TCR = (1 / N) * Σ I[Π I[â_t = a_t] = 1]
- RDR_TCR = (TCR_standard - TCR_perturbed) / TCR_standard * 100%

结果格式：
{
  "agent_name": "xxx",
  "results": [
    {
      "app_tested": "App Name",
      "task_1": [{"type": "clean", "trace_output": [...]}, {"type": "VE", "trace_output": [...]}],
      "task_2": [...]
    }
  ]
}
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
import json
from collections import defaultdict
import argparse


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
    is_correct: bool = False


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


# ============================================================
# Metric 1: AAR (Action Accuracy Rate) - 动作准确率
# ============================================================
def calculate_aar(results: List[TraceResult]) -> float:
    """
    Metric 1: Action Accuracy Rate (AAR) - 动作准确率

    公式：AAR = (1 / Σ T_i) * Σ I[â_t = a_t]

    含义：
    - Σ T_i: 所有trace的checkpoint总数
    - Σ I[â_t = a_t]: 预测正确的checkpoint数
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
# Metric 2: PSSR (Perturbation-Specific Success Rate) - 扰动特定成功率
# ============================================================
def calculate_pssr(results: List[TraceResult]) -> Dict[str, float]:
    """
    Metric 2: Perturbation-Specific Success Rate (PSSR) - 扰动特定成功率

    公式：PSSR_d = (1 / |S_d|) * Σ I[â_t = a_t]
    """
    perturbation_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for trace in results:
        noise_type = trace.noise_type
        for checkpoint in trace.checkpoints:
            perturbation_stats[noise_type]["total"] += 1
            if checkpoint.is_correct:
                perturbation_stats[noise_type]["correct"] += 1

    pssr_results = {}
    for noise_type, stats in perturbation_stats.items():
        if stats["total"] > 0:
            pssr_results[noise_type] = stats["correct"] / stats["total"]
        else:
            pssr_results[noise_type] = 0.0

    return pssr_results


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

    公式：RDR_TCR = (TCR_standard - TCR_perturbed) / TCR_standard * 100%
    """
    tcr_clean = calculate_tcr(clean_results)

    if tcr_clean == 0:
        return {"error": "Clean TCR is 0, cannot calculate RDR"}

    # 按扰动类型分组计算RDR
    perturbed_by_type = defaultdict(list)
    for trace in perturbed_results:
        perturbed_by_type[trace.noise_type].append(trace)

    rdr_results = {}
    for noise_type, traces in perturbed_by_type.items():
        tcr_perturbed = calculate_tcr(traces)
        rdr = (tcr_clean - tcr_perturbed) / tcr_clean * 100
        rdr_results[noise_type] = rdr

    # 计算平均RDR
    if rdr_results:
        valid_values = [v for v in rdr_results.values() if isinstance(v, (int, float))]
        if valid_values:
            rdr_results["average"] = sum(valid_values) / len(valid_values)

    return rdr_results


def calculate_rdr_by_testcase(
    clean_results: List[TraceResult],
    perturbed_results: List[TraceResult]
) -> Dict[int, Dict[str, float]]:
    """按testcase分组计算RDR"""
    clean_by_tc = defaultdict(list)
    for trace in clean_results:
        clean_by_tc[trace.testcase_id].append(trace)

    perturbed_by_tc = defaultdict(lambda: defaultdict(list))
    for trace in perturbed_results:
        perturbed_by_tc[trace.testcase_id][trace.noise_type].append(trace)

    rdr_by_testcase = {}
    for testcase_id in clean_by_tc:
        tcr_clean = calculate_tcr(clean_by_tc[testcase_id])

        if tcr_clean == 0:
            rdr_by_testcase[testcase_id] = {"error": "Clean TCR is 0"}
            continue

        rdr_by_testcase[testcase_id] = {}
        for noise_type, traces in perturbed_by_tc[testcase_id].items():
            tcr_perturbed = calculate_tcr(traces)
            rdr = (tcr_clean - tcr_perturbed) / tcr_clean * 100
            rdr_by_testcase[testcase_id][noise_type] = rdr

    return rdr_by_testcase


# ============================================================
# 综合评估函数
# ============================================================
def evaluate_results(
    agent_name: str,
    ground_truth_path: str,
    prediction_path: str,
    metrics: Optional[Set[str]] = None,
    group_by: Optional[str] = None
) -> Dict:
    """
    读取预测结果和ground truth，计算指定指标

    Args:
        agent_name: Agent名称
        ground_truth_path: ground truth JSON文件路径
        prediction_path: 预测结果JSON文件路径（新格式）
        metrics: 要计算的指标集合，如 {"AAR", "PSSR", "TCR", "RDR"}
        group_by: 分组维度，可选 "noise_type", "testcase_id", "app_name"
    """
    if metrics is None:
        metrics = {"AAR", "PSSR", "TCR", "RDR"}

    # 读取ground truth (格式: {"app": "xxx", "testcases": [...]})
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)

    # 读取预测结果 (新格式)
    with open(prediction_path, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)

    # 构建预测结果索引: {(app_name, task_key, noise_type): trace_output}
    pred_index = _build_prediction_index(pred_data)

    # 构建结果列表
    clean_results = []
    perturbed_results = []

    # 获取app名称
    app_name = gt_data.get("app", "unknown")

    # 处理每个testcase
    for testcase in gt_data.get("testcases", []):
        testcase_id = testcase["testcase_id"]
        testcase_desc = testcase["testcase_desc"]

        # 处理clean trace
        if "clean" in testcase:
            clean_trace = testcase["clean"]["trace"]
            # 查找对应预测: task_{testcase_id}, type="clean"
            pred_key = (app_name, f"task_{testcase_id}", "clean")
            pred_trace = pred_index.get(pred_key, [])

            trace_result = _process_trace(
                agent_name, app_name, testcase_id, testcase_desc,
                "clean", clean_trace, pred_trace
            )
            clean_results.append(trace_result)

        # 处理noise traces
        if "noise" in testcase:
            for noise_item in testcase["noise"]:
                noise_type = noise_item["type"]
                noise_trace = noise_item["trace"]
                # 查找对应预测
                pred_key = (app_name, f"task_{testcase_id}", noise_type)
                pred_trace = pred_index.get(pred_key, [])

                trace_result = _process_trace(
                    agent_name, app_name, testcase_id, testcase_desc,
                    noise_type, noise_trace, pred_trace
                )
                perturbed_results.append(trace_result)

    # 合并所有结果
    all_results = clean_results + perturbed_results

    # 根据参数计算指标
    result = {
        "agent_name": agent_name,
        "app": app_name,
        "total_traces": len(all_results),
        "total_checkpoints": sum(len(t.checkpoints) for t in all_results),
    }

    # AAR
    if "AAR" in metrics:
        result["AAR"] = calculate_aar(all_results)
        if group_by:
            result[f"AAR_by_{group_by}"] = calculate_aar_by_group(all_results, group_by)

    # PSSR
    if "PSSR" in metrics:
        result["PSSR"] = calculate_pssr(all_results)

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
        if group_by == "testcase_id":
            result["RDR_by_testcase"] = calculate_rdr_by_testcase(clean_results, perturbed_results)

    return result


def _build_prediction_index(pred_data: Dict) -> Dict:
    """
    构建预测结果索引

    输入格式:
    {
      "agent_name": "xxx",
      "results": [
        {
          "app_tested": "App Name",
          "task_1": [{"type": "clean", "trace_output": [...]}, {"type": "VE", "trace_output": [...]}],
          "task_2": [...]
        }
      ]
    }

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


def _process_trace(
    agent_name: str,
    app_name: str,
    testcase_id: int,
    testcase_desc: str,
    noise_type: str,
    gt_trace: List[Dict],
    pred_trace: List[Dict]
) -> TraceResult:
    """处理单条trace，构建TraceResult对象"""
    trace_result = TraceResult(
        agent_name=agent_name,
        app_name=app_name,
        testcase_id=testcase_id,
        testcase_desc=testcase_desc,
        noise_type=noise_type
    )

    # 处理每个checkpoint
    all_correct = True
    for i, gt_checkpoint in enumerate(gt_trace):
        checkpoint_id = gt_checkpoint["checkpoint_id"]
        gt_action = gt_checkpoint["action"]
        gt_bbox = gt_checkpoint.get("bbox")

        # 获取预测结果
        pred_action = None
        pred_bbox = None
        if i < len(pred_trace):
            pred_item = pred_trace[i]
            pred_action = pred_item.get("action")
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

        # 判断是否正确
        is_correct = (pred_action == gt_action)
        if not is_correct:
            all_correct = False

        checkpoint_result = CheckpointResult(
            app_name=app_name,
            testcase_id=testcase_id,
            noise_type=noise_type,
            checkpoint_id=checkpoint_id,
            predicted_action=pred_action or "UNKNOWN",
            ground_truth_action=gt_action,
            predicted_bbox=pred_bbox,
            ground_truth_bbox=gt_bbox,
            is_correct=is_correct
        )
        trace_result.checkpoints.append(checkpoint_result)

    trace_result.is_completed = all_correct
    return trace_result


def print_metrics(metrics: Dict):
    """打印指标结果"""
    print("\n" + "="*60)
    print(f"评估结果 - {metrics['agent_name']}")
    print(f"App: {metrics.get('app', 'N/A')}")
    print("="*60)
    print(f"总Trace数: {metrics['total_traces']}")
    print(f"总Checkpoint数: {metrics['total_checkpoints']}")
    print("-"*60)

    if "AAR" in metrics:
        print(f"AAR (动作准确率): {metrics['AAR']:.4f} ({metrics['AAR']*100:.2f}%)")
        for key in metrics:
            if key.startswith("AAR_by_"):
                print(f"  {key}:")
                for k, v in metrics[key].items():
                    print(f"    {k}: {v:.4f} ({v*100:.2f}%)")

    if "PSSR" in metrics:
        print("-"*60)
        print("PSSR (扰动特定成功率):")
        for noise_type, pssr in metrics['PSSR'].items():
            print(f"  {noise_type}: {pssr:.4f} ({pssr*100:.2f}%)")

    if "TCR_clean" in metrics:
        print("-"*60)
        print(f"TCR_clean (标准环境任务完成率): {metrics['TCR_clean']:.4f} ({metrics['TCR_clean']*100:.2f}%)")
        print(f"TCR_perturbed (扰动环境任务完成率): {metrics['TCR_perturbed']:.4f} ({metrics['TCR_perturbed']*100:.2f}%)")
        print(f"TCR_all (全部任务完成率): {metrics['TCR_all']:.4f} ({metrics['TCR_all']*100:.2f}%)")

    if "RDR" in metrics:
        print("-"*60)
        print("RDR (鲁棒性下降度):")
        if isinstance(metrics['RDR'], dict):
            for key, value in metrics['RDR'].items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.2f}%")
                else:
                    print(f"  {key}: {value}")

    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="计算评估指标")
    parser.add_argument("ground_truth", help="Ground truth JSON文件路径")
    parser.add_argument("prediction", help="预测结果JSON文件路径")
    parser.add_argument("--agent", "-a", default=None, help="Agent名称（默认从预测文件读取）")
    parser.add_argument("--metrics", "-m", nargs="+",
                        choices=["AAR", "PSSR", "TCR", "RDR"],
                        help="要计算的指标（默认全部）")
    parser.add_argument("--group-by", "-g",
                        choices=["noise_type", "testcase_id", "app_name"],
                        help="分组维度")

    args = parser.parse_args()

    # 从预测文件读取agent_name
    if args.agent is None:
        with open(args.prediction, 'r', encoding='utf-8') as f:
            pred_data = json.load(f)
            agent_name = pred_data.get("agent_name", "Unknown_Agent")
    else:
        agent_name = args.agent

    metrics_set = set(args.metrics) if args.metrics else None

    metrics = evaluate_results(
        agent_name,
        args.ground_truth,
        args.prediction,
        metrics=metrics_set,
        group_by=args.group_by
    )
    print_metrics(metrics)

    # 保存结果到当前目录下的 metrics_results.txt
    output_path = "metrics_results.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"评估结果 - {metrics['agent_name']}\n")
        f.write(f"App: {metrics.get('app', 'N/A')}\n")
        f.write("="*60 + "\n")
        f.write(f"总Trace数: {metrics['total_traces']}\n")
        f.write(f"总Checkpoint数: {metrics['total_checkpoints']}\n")
        f.write("-"*60 + "\n")

        if "AAR" in metrics:
            f.write(f"AAR (动作准确率): {metrics['AAR']:.4f} ({metrics['AAR']*100:.2f}%)\n")
            for key in metrics:
                if key.startswith("AAR_by_"):
                    f.write(f"  {key}:\n")
                    for k, v in metrics[key].items():
                        f.write(f"    {k}: {v:.4f} ({v*100:.2f}%)\n")

        if "PSSR" in metrics:
            f.write("-"*60 + "\n")
            f.write("PSSR (扰动特定成功率):\n")
            for noise_type, pssr in metrics['PSSR'].items():
                f.write(f"  {noise_type}: {pssr:.4f} ({pssr*100:.2f}%)\n")

        if "TCR_clean" in metrics:
            f.write("-"*60 + "\n")
            f.write(f"TCR_clean (标准环境任务完成率): {metrics['TCR_clean']:.4f} ({metrics['TCR_clean']*100:.2f}%)\n")
            f.write(f"TCR_perturbed (扰动环境任务完成率): {metrics['TCR_perturbed']:.4f} ({metrics['TCR_perturbed']*100:.2f}%)\n")
            f.write(f"TCR_all (全部任务完成率): {metrics['TCR_all']:.4f} ({metrics['TCR_all']*100:.2f}%)\n")

        if "RDR" in metrics:
            f.write("-"*60 + "\n")
            f.write("RDR (鲁棒性下降度):\n")
            if isinstance(metrics['RDR'], dict):
                for key, value in metrics['RDR'].items():
                    if isinstance(value, (int, float)):
                        f.write(f"  {key}: {value:.2f}%\n")
                    else:
                        f.write(f"  {key}: {value}\n")

        f.write("="*60 + "\n")

    print(f"\n指标结果已保存到: {output_path}")
