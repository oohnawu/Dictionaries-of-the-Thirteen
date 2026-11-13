#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""统一评估论文补充实验，并复用已经计算完成的指标。

默认行为：
1. 直接读取旧 ``baseline_run_summary.json`` 中已经完成的分类准确率；
2. 直接读取原论文 Excel 汇总表中的 Qwen3-8B D0 指标；
3. 为释义 BM25、语境 BM25、释义 D2/D3/D4 和语境 RAG 计算
   ROUGE-1/2/L F1 与 Sentence-BERT 余弦相似度；
4. 通过输入文件 SHA-256 和指标配置判断缓存是否有效，只补算缺失指标。

本程序不写入原始数据、原评测结果、数据审计或模型污染测试目录。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable

from openpyxl import load_workbook

from baseline_common import (
    atomic_write_json,
    clean,
    compact_key,
    read_json_list,
    to_simplified,
)


EVALUATOR_VERSION = "supplementary-baselines-v1"
DEFAULT_SBERT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
METRIC_FIELDS = (
    "rouge_1_f1",
    "rouge_2_f1",
    "rouge_l_f1",
    "rouge_avg_f1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="./evaluation_results",
        help="派生评估结果目录；默认位于正式实验目录内。",
    )
    parser.add_argument(
        "--sbert-model",
        default=DEFAULT_SBERT_MODEL,
        help="原论文Sentence-BERT模型的HF ID或本地模型目录。",
    )
    parser.add_argument(
        "--sbert-device",
        default=None,
        help=(
            "可选：cpu、cuda或cuda:0；默认由sentence-transformers决定。"
            "若指定CUDA但当前PyTorch不支持，默认自动退回CPU。"
        ),
    )
    parser.add_argument(
        "--require-sbert-device",
        action="store_true",
        help=(
            "严格要求--sbert-device可用；例如CUDA不可用时直接报错，"
            "而不是自动退回CPU。"
        ),
    )
    parser.add_argument("--sbert-batch-size", type=int, default=32)
    parser.add_argument(
        "--skip-sbert",
        action="store_true",
        help="只计算/复用ROUGE；用于无sentence-transformers环境的检查。",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help=(
            "只处理指定ID；可选：definition_bm25 context_bm25 "
            "definition_d2 definition_d3 definition_d4 context_rag。"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="忽略语义指标缓存并重新计算；正式复跑通常不需要。",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON顶层应为对象：{path}")
    return data


def normalized_text(value: Any) -> str:
    """NFKC、去除首尾空白并统一为简体。"""
    return to_simplified(clean(value))


def extract_definition(value: Any) -> str:
    """复现原 ``Similarity_sentenceBERT.py`` 的含义字段提取逻辑。"""
    text = normalized_text(value)
    patterns = (
        r"含义[：:]?\s*(.*?)(?:\n|$)",
        r"含义为(.*?)[。\n]",
        r"意思是(.*?)[。\n]",
        r"[：:]\s*(.*?)[。\n]",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match and match.group(1):
            return normalized_text(match.group(1).strip("。. "))
    return text


def jieba_tokens(value: Any) -> list[str]:
    """与正式释义评估脚本一致：简体化后jieba分词并去标点/空白。"""
    try:
        import jieba
    except ImportError as error:
        raise RuntimeError(
            "ROUGE需要jieba。请在正式环境运行：pip install -r requirements-rag.txt"
        ) from error
    tokens: list[str] = []
    for token in jieba.lcut(normalized_text(value), cut_all=False):
        token_key = compact_key(token)
        if token_key:
            tokens.append(token_key)
    return tokens


def ngrams(tokens: list[str], size: int) -> list[tuple[str, ...]]:
    return [
        tuple(tokens[index : index + size])
        for index in range(max(len(tokens) - size + 1, 0))
    ]


def overlap_f1(
    prediction_tokens: list[str],
    reference_tokens: list[str],
    *,
    size: int,
) -> float:
    prediction_counts = Counter(ngrams(prediction_tokens, size))
    reference_counts = Counter(ngrams(reference_tokens, size))
    prediction_total = sum(prediction_counts.values())
    reference_total = sum(reference_counts.values())
    if prediction_total == 0 or reference_total == 0:
        return 0.0
    overlap = sum(
        min(count, reference_counts[gram])
        for gram, count in prediction_counts.items()
    )
    precision = overlap / prediction_total
    recall = overlap / reference_total
    return (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def lcs_length(left: list[str], right: list[str]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(
    prediction_tokens: list[str],
    reference_tokens: list[str],
) -> float:
    if not prediction_tokens or not reference_tokens:
        return 0.0
    common = lcs_length(prediction_tokens, reference_tokens)
    precision = common / len(prediction_tokens)
    recall = common / len(reference_tokens)
    return (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def add_rouge(row: dict[str, Any]) -> None:
    prediction_tokens = jieba_tokens(row["prediction"])
    reference_tokens = jieba_tokens(row["reference"])
    rouge_1 = overlap_f1(prediction_tokens, reference_tokens, size=1)
    rouge_2 = overlap_f1(prediction_tokens, reference_tokens, size=2)
    rouge_l = rouge_l_f1(prediction_tokens, reference_tokens)
    row.update(
        {
            "rouge_1_f1": rouge_1,
            "rouge_2_f1": rouge_2,
            "rouge_l_f1": rouge_l,
            "rouge_avg_f1": (rouge_1 + rouge_2 + rouge_l) / 3.0,
            "prediction_tokens": len(prediction_tokens),
            "reference_tokens": len(reference_tokens),
        }
    )


def load_sbert(
    model_path_or_id: str,
    *,
    device: str | None,
    require_device: bool,
) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "语义相似度需要sentence-transformers。请运行："
            "pip install -r requirements-rag.txt"
        ) from error
    effective_device = device
    if device and device.lower().startswith("cuda"):
        try:
            import torch
        except ImportError as error:
            raise RuntimeError(
                "指定了CUDA，但当前环境没有安装PyTorch。"
            ) from error
        cuda_available = bool(torch.cuda.is_available())
        torch_cuda_version = getattr(torch.version, "cuda", None)
        if not cuda_available:
            message = (
                "指定了--sbert-device "
                f"{device!r}，但当前PyTorch不支持可用CUDA："
                f"torch={torch.__version__}, "
                f"torch.version.cuda={torch_cuda_version!r}, "
                f"torch.cuda.is_available()={cuda_available}。"
            )
            if require_device:
                raise RuntimeError(
                    message
                    + " 请安装与显卡驱动匹配的CUDA版PyTorch，"
                    "或删除--require-sbert-device以允许CPU回退。"
                )
            effective_device = "cpu"
            print(f"[SBERT设备回退] {message}")
            print(
                "[SBERT设备回退] 本次改用CPU；指标公式和结果口径不变，"
                "只会影响计算速度。"
            )
    kwargs: dict[str, Any] = {}
    if effective_device:
        kwargs["device"] = effective_device
    return SentenceTransformer(model_path_or_id, **kwargs)


def add_sbert(
    rows: list[dict[str, Any]],
    *,
    model: Any,
    batch_size: int,
) -> None:
    """批量实现原代码逐对编码后计算余弦相似度的等价形式。"""
    predictions = [extract_definition(row["prediction"]) for row in rows]
    references = [extract_definition(row["reference"]) for row in rows]
    prediction_embeddings = model.encode(
        predictions,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    reference_embeddings = model.encode(
        references,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    for row, prediction_embedding, reference_embedding in zip(
        rows,
        prediction_embeddings,
        reference_embeddings,
    ):
        similarity = float(prediction_embedding @ reference_embedding)
        row["sbert_cosine"] = max(0.0, min(1.0, similarity))


def average_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows_list = list(rows)
    output: dict[str, Any] = {"count": len(rows_list)}
    for field in METRIC_FIELDS:
        values = [float(row[field]) for row in rows_list if field in row]
        output[field] = round(mean(values), 8) if values else None
        output[f"{field}_percent"] = (
            round(output[field] * 100.0, 6)
            if output[field] is not None
            else None
        )
    sbert_values = [
        float(row["sbert_cosine"])
        for row in rows_list
        if "sbert_cosine" in row
    ]
    output["sbert_cosine"] = (
        round(mean(sbert_values), 8) if sbert_values else None
    )
    output["sbert_cosine_percent"] = (
        round(output["sbert_cosine"] * 100.0, 6)
        if output["sbert_cosine"] is not None
        else None
    )
    return output


def grouped_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups["all"].append(row)
        groups[
            "retrieval_hit" if row["retrieval_hit"] else "no_evidence"
        ].append(row)
        groups[f"scope:{row['retrieval_scope']}"].append(row)
    return {
        name: average_metrics(group_rows)
        for name, group_rows in sorted(groups.items())
    }


def target_definitions(script_dir: Path) -> list[dict[str, Any]]:
    workspace = script_dir.parent
    old_results = workspace / "baseline_results"
    formal_results = script_dir / "baseline_results"
    return [
        {
            "id": "definition_bm25",
            "label": "释义BM25（旧严格基线）",
            "task": "释义能力",
            "prediction_path": old_results
            / "释义能力"
            / "0-shot"
            / "bm25-top1-strict#释义能力#0-shot.json",
        },
        {
            "id": "context_bm25",
            "label": "语境理解BM25",
            "task": "语境理解",
            "prediction_path": old_results
            / "语境理解"
            / "0-shot"
            / "bm25-top1-strict#语境理解#0-shot.json",
        },
        {
            "id": "definition_d2",
            "label": "D2：跨辞书RAG（仅例句）",
            "task": "释义能力",
            "prediction_path": formal_results
            / "释义能力"
            / "0-shot"
            / "qwen3-8b-bailian-rag-formal-d2#释义能力#0-shot.json",
        },
        {
            "id": "definition_d3",
            "label": "D3：跨辞书释义RAG（释义+词性）",
            "task": "释义能力",
            "prediction_path": formal_results
            / "释义能力"
            / "0-shot"
            / "qwen3-8b-bailian-rag-formal-d3#释义能力#0-shot.json",
        },
        {
            "id": "definition_d4",
            "label": "D4：跨辞书释义RAG（释义+词性+例证）",
            "task": "释义能力",
            "prediction_path": formal_results
            / "释义能力"
            / "0-shot"
            / "qwen3-8b-bailian-rag-formal-d4#释义能力#0-shot.json",
        },
        {
            "id": "context_rag",
            "label": "语境理解RAG",
            "task": "语境理解",
            "prediction_path": formal_results
            / "语境理解"
            / "0-shot"
            / "qwen3-8b-bailian-rag-formal#语境理解#0-shot.json",
        },
    ]


def gold_path_for(script_dir: Path, task: str) -> Path:
    return (
        script_dir
        / "eval_data"
        / "json"
        / task
        / "0-shot"
        / f"{task}_eval.json"
    )


def aligned_rows(
    predictions: list[dict[str, Any]],
    gold_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(predictions) != len(gold_items):
        raise ValueError(
            f"预测数({len(predictions)})和金标准数({len(gold_items)})不一致。"
        )
    rows: list[dict[str, Any]] = []
    for index, (prediction_item, gold_item) in enumerate(
        zip(predictions, gold_items),
        start=1,
    ):
        prediction_headword = clean(prediction_item.get("input"))
        gold_headword = clean(gold_item.get("input"))
        if prediction_headword != gold_headword:
            raise ValueError(
                f"第{index}条字头不一致："
                f"prediction={prediction_headword!r}, gold={gold_headword!r}"
            )
        if prediction_item.get("error"):
            raise ValueError(
                f"第{index}条仍有生成错误：{prediction_item['error']}"
            )
        if "prediction" not in prediction_item:
            raise ValueError(f"第{index}条缺少prediction字段。")
        prediction = clean(prediction_item.get("prediction"))
        reference = clean(gold_item.get("output"))
        if not prediction:
            raise ValueError(f"第{index}条prediction为空。")
        if not reference:
            raise ValueError(f"第{index}条金标准output为空。")
        retrieval = prediction_item.get("retrieval")
        retrieval_list = retrieval if isinstance(retrieval, list) else []
        rows.append(
            {
                "index": index,
                "headword": gold_headword,
                "prediction": prediction,
                "reference": reference,
                "retrieval_hit": bool(
                    prediction_item.get("retrieval_hit", retrieval_list)
                ),
                "retrieval_scope": clean(
                    prediction_item.get("retrieval_scope")
                )
                or "unknown",
                "retrieval_count": len(retrieval_list),
            }
        )
    return rows


def report_cache_valid(
    report: dict[str, Any],
    *,
    prediction_sha256: str,
    gold_sha256: str,
) -> bool:
    return (
        report.get("evaluator_version") == EVALUATOR_VERSION
        and report.get("prediction_sha256") == prediction_sha256
        and report.get("gold_sha256") == gold_sha256
        and report.get("rouge_tokenizer") == "jieba-compact"
        and report.get("traditional_to_simplified_before_scoring") is True
    )


def evaluate_target(
    target: dict[str, Any],
    *,
    script_dir: Path,
    output_dir: Path,
    want_sbert: bool,
    sbert_model_name: str,
    sbert_model: Any | None,
    batch_size: int,
    force: bool,
) -> tuple[dict[str, Any], Any | None, str]:
    prediction_path = Path(target["prediction_path"]).resolve()
    gold_path = gold_path_for(script_dir, target["task"]).resolve()
    if not prediction_path.exists():
        raise FileNotFoundError(f"预测文件不存在：{prediction_path}")
    if not gold_path.exists():
        raise FileNotFoundError(f"金标准不存在：{gold_path}")

    prediction_sha = sha256_file(prediction_path)
    gold_sha = sha256_file(gold_path)
    report_path = output_dir / f"{target['id']}_metrics.json"
    per_item_path = output_dir / f"{target['id']}_per_item.json"
    cached_report: dict[str, Any] | None = None
    rows: list[dict[str, Any]] | None = None
    cache_valid = False
    if not force and report_path.exists() and per_item_path.exists():
        cached_report = load_json_object(report_path)
        cache_valid = report_cache_valid(
            cached_report,
            prediction_sha256=prediction_sha,
            gold_sha256=gold_sha,
        )
        if cache_valid:
            cached_rows = read_json_list(per_item_path)
            if len(cached_rows) == len(read_json_list(gold_path)):
                rows = cached_rows
            else:
                cache_valid = False
                rows = None

    has_rouge = bool(
        cache_valid
        and rows
        and all(all(field in row for field in METRIC_FIELDS) for row in rows)
    )
    has_requested_sbert = bool(
        cache_valid
        and rows
        and cached_report
        and cached_report.get("sbert_model") == sbert_model_name
        and all("sbert_cosine" in row for row in rows)
    )
    if has_rouge and (not want_sbert or has_requested_sbert):
        assert cached_report is not None
        return cached_report, sbert_model, "reused"

    if rows is None or not cache_valid:
        rows = aligned_rows(
            read_json_list(prediction_path),
            read_json_list(gold_path),
        )
    if not has_rouge:
        for row in rows:
            add_rouge(row)

    sbert_was_added = False
    if want_sbert and not has_requested_sbert:
        if sbert_model is None:
            raise RuntimeError("内部错误：需要SBERT但模型尚未加载。")
        add_sbert(rows, model=sbert_model, batch_size=batch_size)
        sbert_was_added = True

    groups = grouped_summary(rows)
    report = {
        "evaluator_version": EVALUATOR_VERSION,
        "target_id": target["id"],
        "label": target["label"],
        "task": target["task"],
        "prediction_path": str(prediction_path),
        "prediction_sha256": prediction_sha,
        "gold_path": str(gold_path),
        "gold_sha256": gold_sha,
        "evaluated_items": len(rows),
        "rouge_tokenizer": "jieba-compact",
        "traditional_to_simplified_before_scoring": True,
        "sbert_model": sbert_model_name if want_sbert else None,
        "groups": groups,
        "metric_definition": {
            "rouge_1_2": "multiset n-gram overlap F1",
            "rouge_l": "longest-common-subsequence F1",
            "rouge_avg": "mean(ROUGE-1 F1, ROUGE-2 F1, ROUGE-L F1)",
            "aggregation": "per-item score, then macro average over 200 items",
            "sbert": (
                "cosine similarity after original extract_definition logic"
            ),
        },
        "per_item_path": str(per_item_path.resolve()),
    }
    atomic_write_json(per_item_path, rows)
    atomic_write_json(report_path, report)
    if sbert_was_added and has_rouge:
        status = "computed-sbert-only"
    elif sbert_was_added:
        status = "computed-rouge-and-sbert"
    else:
        status = "computed-rouge-only"
    return report, sbert_model, status


def reused_accuracy_metrics(workspace: Path) -> dict[str, Any]:
    summary_path = workspace / "baseline_results" / "baseline_run_summary.json"
    source = load_json_object(summary_path)
    task_map = (
        ("radical_openbook", "部首识别", "部首open-book lookup"),
        ("pinyin_openbook", "拼音注音", "拼音open-book lookup"),
        ("fanqie_openbook", "反切注音", "反切open-book lookup"),
        ("pos_majority", "词性识别", "词性多数基线"),
        (
            "pos_context_bm25",
            "词性识别（上下文BM25）",
            "上下文词性BM25",
        ),
    )
    output: dict[str, Any] = {
        "source_path": str(summary_path.resolve()),
        "source_sha256": sha256_file(summary_path),
        "status": "reused-not-recomputed",
        "metrics": {},
    }
    for metric_id, source_key, label in task_map:
        if source_key not in source:
            raise KeyError(f"已有准确率汇总缺少任务：{source_key}")
        item = source[source_key]
        output["metrics"][metric_id] = {
            "label": label,
            "items": int(item["items"]),
            "correct": int(item["exact_match_count"]),
            "accuracy": float(item["exact_match_accuracy"]),
            "accuracy_percent": round(
                100.0 * float(item["exact_match_accuracy"]),
                6,
            ),
            "surface_correct_for_audit": int(
                item["surface_exact_match_count"]
            ),
            "surface_accuracy_for_audit": float(
                item["surface_exact_match_accuracy"]
            ),
            "normalization": item["accuracy_normalization"],
            "method": item["method"],
        }
    return output


def read_xlsx_table(path: Path) -> list[dict[str, Any]]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = workbook.worksheets[0]
        rows = sheet.iter_rows(values_only=True)
        headers = [clean(value) for value in next(rows)]
        output: list[dict[str, Any]] = []
        for values in rows:
            record = {
                header: value
                for header, value in zip(headers, values)
                if header
            }
            if any(value is not None for value in record.values()):
                output.append(record)
        return output
    finally:
        workbook.close()


def reused_d0_metrics(workspace: Path) -> dict[str, Any]:
    metric_dir = workspace / "原指标计算结果"
    rouge_path = metric_dir / "all_rouge.xlsx"
    similarity_path = metric_dir / "all_similarity.xlsx"
    rouge_rows = read_xlsx_table(rouge_path)
    similarity_rows = read_xlsx_table(similarity_path)
    output: dict[str, Any] = {
        "status": "reused-not-recomputed",
        "rouge_source_path": str(rouge_path.resolve()),
        "rouge_source_sha256": sha256_file(rouge_path),
        "similarity_source_path": str(similarity_path.resolve()),
        "similarity_source_sha256": sha256_file(similarity_path),
        "metrics": {},
    }
    task_map = {
        "context_d0": "语境理解",
        "definition_d0": "释义生成",
    }
    for metric_id, task_name in task_map.items():
        rouge_match = [
            row
            for row in rouge_rows
            if clean(row.get("模型名")) == "Qwen3-8B"
            and clean(row.get("任务名")) == task_name
            and clean(row.get("评测模式")) == "0-shot"
        ]
        similarity_match = [
            row
            for row in similarity_rows
            if clean(row.get("模型名")) == "Qwen3-8B"
            and clean(row.get("任务名")) == task_name
            and clean(row.get("评测模式")) == "0-shot"
        ]
        if len(rouge_match) != 1 or len(similarity_match) != 1:
            raise ValueError(
                f"原论文D0指标定位失败：task={task_name}, "
                f"rouge={len(rouge_match)}, similarity={len(similarity_match)}"
            )
        rouge = rouge_match[0]
        similarity = similarity_match[0]
        output["metrics"][metric_id] = {
            "label": f"D0：Qwen3-8B {task_name} 0-shot（原论文）",
            "task": task_name,
            "rouge_1_f1": float(rouge["ROUGE-1"]),
            "rouge_2_f1": float(rouge["ROUGE-2"]),
            "rouge_l_f1": float(rouge["ROUGE-L"]),
            "rouge_avg_f1": float(rouge["ROUGE-Avg(%)"]) / 100.0,
            "rouge_avg_f1_percent": float(rouge["ROUGE-Avg(%)"]),
            "sbert_cosine": float(similarity["平均语义相似度"]),
            "sbert_cosine_percent": float(
                similarity["平均语义相似度(%)"]
            ),
            "items": int(similarity["样本数量"]),
        }
    return output


def paper_rows(
    accuracy: dict[str, Any],
    d0: dict[str, Any],
    semantic_reports: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric_id, metric in accuracy["metrics"].items():
        rows.append(
            {
                "id": metric_id,
                "task": metric["label"],
                "group": "supplementary-classification-baseline",
                "items": metric["items"],
                "accuracy_percent": metric["accuracy_percent"],
                "rouge_1_percent": "",
                "rouge_2_percent": "",
                "rouge_l_percent": "",
                "rouge_avg_percent": "",
                "sbert_percent": "",
                "source_status": "reused",
            }
        )
    for metric_id, metric in d0["metrics"].items():
        rows.append(
            {
                "id": metric_id,
                "task": metric["label"],
                "group": "original-paper-D0",
                "items": metric["items"],
                "accuracy_percent": "",
                "rouge_1_percent": round(
                    metric["rouge_1_f1"] * 100.0, 6
                ),
                "rouge_2_percent": round(
                    metric["rouge_2_f1"] * 100.0, 6
                ),
                "rouge_l_percent": round(
                    metric["rouge_l_f1"] * 100.0, 6
                ),
                "rouge_avg_percent": metric["rouge_avg_f1_percent"],
                "sbert_percent": metric["sbert_cosine_percent"],
                "source_status": "reused",
            }
        )
    for metric_id, report in semantic_reports.items():
        metric = report["groups"]["all"]
        rows.append(
            {
                "id": metric_id,
                "task": report["label"],
                "group": "supplementary-semantic-baseline",
                "items": metric["count"],
                "accuracy_percent": "",
                "rouge_1_percent": metric["rouge_1_f1_percent"],
                "rouge_2_percent": metric["rouge_2_f1_percent"],
                "rouge_l_percent": metric["rouge_l_f1_percent"],
                "rouge_avg_percent": metric["rouge_avg_f1_percent"],
                "sbert_percent": metric["sbert_cosine_percent"],
                "source_status": report["calculation_status"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "task",
        "group",
        "items",
        "accuracy_percent",
        "rouge_1_percent",
        "rouge_2_percent",
        "rouge_l_percent",
        "rouge_avg_percent",
        "sbert_percent",
        "source_status",
    ]
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if args.sbert_batch_size < 1:
        raise ValueError("--sbert-batch-size必须大于0。")
    script_dir = Path(__file__).resolve().parent
    workspace = script_dir.parent
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (script_dir / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_targets = target_definitions(script_dir)
    available_ids = {target["id"] for target in all_targets}
    selected_ids = set(args.only or available_ids)
    unknown = selected_ids - available_ids
    if unknown:
        raise ValueError(f"--only包含未知ID：{sorted(unknown)}")
    targets = [
        target for target in all_targets if target["id"] in selected_ids
    ]

    accuracy = reused_accuracy_metrics(workspace)
    d0 = reused_d0_metrics(workspace)
    want_sbert = not args.skip_sbert
    model: Any | None = None
    reports: dict[str, dict[str, Any]] = {}

    # 先检查缓存。仅当至少一个目标缺少SBERT时才加载模型。
    if want_sbert:
        needs_model = False
        for target in targets:
            prediction_path = Path(target["prediction_path"]).resolve()
            gold_path = gold_path_for(script_dir, target["task"]).resolve()
            report_path = output_dir / f"{target['id']}_metrics.json"
            per_item_path = output_dir / f"{target['id']}_per_item.json"
            if args.force or not report_path.exists() or not per_item_path.exists():
                needs_model = True
                break
            cached = load_json_object(report_path)
            if not report_cache_valid(
                cached,
                prediction_sha256=sha256_file(prediction_path),
                gold_sha256=sha256_file(gold_path),
            ):
                needs_model = True
                break
            if cached.get("sbert_model") != args.sbert_model:
                needs_model = True
                break
            cached_rows = read_json_list(per_item_path)
            if not cached_rows or not all(
                "sbert_cosine" in row for row in cached_rows
            ):
                needs_model = True
                break
        if needs_model:
            print(f"[SBERT] 加载模型：{args.sbert_model}")
            model = load_sbert(
                args.sbert_model,
                device=clean(args.sbert_device) or None,
                require_device=args.require_sbert_device,
            )

    for target in targets:
        report, model, status = evaluate_target(
            target,
            script_dir=script_dir,
            output_dir=output_dir,
            want_sbert=want_sbert,
            sbert_model_name=args.sbert_model,
            sbert_model=model,
            batch_size=args.sbert_batch_size,
            force=args.force,
        )
        report = dict(report)
        report["calculation_status"] = status
        reports[target["id"]] = report
        overall = report["groups"]["all"]
        print(
            f"[{status}] {target['label']}："
            f"n={overall['count']}, "
            f"ROUGE-Avg={overall['rouge_avg_f1_percent']}%, "
            f"SBERT={overall['sbert_cosine_percent']}%"
        )

    rows = paper_rows(accuracy, d0, reports)
    summary = {
        "evaluator_version": EVALUATOR_VERSION,
        "policy": {
            "classification_accuracy": (
                "reuse existing NFKC + OpenCC t2s + whitespace-removal "
                "strict accuracy"
            ),
            "semantic_metrics": (
                "ROUGE-1/2/L F1 macro average and original-paper "
                "Sentence-BERT cosine"
            ),
            "already_computed_metrics": (
                "reuse when source SHA-256 and metric configuration match"
            ),
            "d0": "read from original paper workbooks; never recomputed",
            "d5_d6": (
                "left in original-paper results; no mapping guessed by this script"
            ),
        },
        "reused_accuracy": accuracy,
        "reused_d0": d0,
        "semantic_reports": reports,
        "paper_table_rows": rows,
    }
    summary_path = output_dir / "supplementary_baseline_metrics.json"
    csv_path = output_dir / "supplementary_baseline_metrics.csv"
    atomic_write_json(summary_path, summary)
    write_csv(csv_path, rows)
    print(f"[完成] 统一汇总：{summary_path}")
    print(f"[完成] 论文表格：{csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
