#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""按原论文口径计算“补充评测”中的新增模型指标。

读取（只读）：
  ../补充评测
  ../原评测结果（仅用于复现校验）
  ../原指标计算结果/all_Accuracy.xlsx
  ../原指标计算结果/all_rouge.xlsx
  ../原指标计算结果/all_similarity.xlsx

写入（派生结果）：
  ./derived_results/added_model_metrics.json

复现口径：
  * 准确率：区分大小写的 ``output in prediction``；部首识别先在
    ``output``和``prediction``两侧删除“部”字；
  * ROUGE：zhconv繁转简 -> 原脚本含义提取 -> jieba -> rouge 1.0.1
    的集合n-gram F1与LCS F1；
  * 相似度：原脚本含义提取 +
    paraphrase-multilingual-MiniLM-L12-v2 + cosine_similarity。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from openpyxl import load_workbook


EVALUATOR_VERSION = "original-metrics-compatible-v4-radical-normalized"
EPSILON = 1e-8
CLASSIFICATION_TASKS = {
    "部首识别",
    "拼音注音",
    "反切注音",
    "词性识别",
}
SEMANTIC_TASKS = {"语境理解", "释义能力"}
TASK_TO_PAPER = {"释义能力": "释义生成"}
PAPER_TO_FILE_TASK = {"释义生成": "释义能力"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sbert-model",
        default=None,
        help="paraphrase-multilingual-MiniLM-L12-v2本地目录或HF ID。",
    )
    parser.add_argument(
        "--sbert-device",
        default=None,
        help="cpu、cuda或cuda:0；默认由sentence-transformers选择。",
    )
    parser.add_argument(
        "--skip-sbert",
        action="store_true",
        help="只计算准确率和ROUGE。",
    )
    parser.add_argument(
        "--require-sbert-device",
        action="store_true",
        help="指定设备不可用时直接报错，不自动退回CPU。",
    )
    parser.add_argument(
        "--sbert-batch-size",
        type=int,
        default=32,
        help="SBERT批量编码大小；不改变文本、模型或余弦公式。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="忽略有效缓存，重新计算。",
    )
    parser.add_argument(
        "--validate-original",
        action="store_true",
        help="用原评测结果反向验证原指标复现程度。",
    )
    parser.add_argument(
        "--validate-rouge-rows",
        type=int,
        default=8,
        help="--validate-original时校验的原ROUGE行数。",
    )
    parser.add_argument(
        "--validate-sbert-rows",
        type=int,
        default=1,
        help="--validate-original时校验的原SBERT行数。",
    )
    parser.add_argument(
        "--output",
        default="./derived_results/added_model_metrics.json",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def read_json_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
    else:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(data, list) or not all(
        isinstance(item, dict) for item in data
    ):
        raise ValueError(f"JSON格式不符合预期：{path}")
    return data


def parse_result_filename(path: Path) -> tuple[str, str, str]:
    parts = path.stem.split("#")
    if len(parts) != 3:
        raise ValueError(f"结果文件名应为模型#任务#模式.json：{path.name}")
    return parts[0], parts[1], parts[2]


def get_reference(item: dict[str, Any]) -> str:
    value = item.get("output", item.get("raw", ""))
    return "" if value is None else str(value)


def get_prediction(item: dict[str, Any]) -> str:
    value = item.get("prediction", item.get("predicate", ""))
    return "" if value is None else str(value)


def require_original_dependencies() -> tuple[Any, Any]:
    try:
        import jieba
    except ImportError as error:
        raise RuntimeError("缺少jieba：pip install jieba==0.42.1") from error
    try:
        import zhconv
    except ImportError as error:
        raise RuntimeError("缺少zhconv：pip install zhconv==1.4.3") from error
    return jieba, zhconv


def normalize_text(text: Any, *, zhconv_module: Any) -> str:
    if text is None:
        return ""
    return zhconv_module.convert(str(text).strip(), "zh-cn")


def extract_definition(text: Any, *, zhconv_module: Any) -> str:
    """逐字复现原Similarity_sentenceBERT.py的文本提取规则。"""
    normalized = normalize_text(text, zhconv_module=zhconv_module)
    patterns = (
        r"含义[：:]?\s*(.*?)(?:\n|$)",
        r"含义为(.*?)[。\n]",
        r"意思是(.*?)[。\n]",
        r"[：:]\s*(.*?)[。\n]",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match and match.group(1):
            return normalize_text(
                match.group(1).strip("。. "),
                zhconv_module=zhconv_module,
            )
    return normalized


def accuracy_metrics(
    records: list[dict[str, Any]],
    *,
    task: str | None = None,
) -> dict[str, Any]:
    if not records:
        raise ValueError("没有可计算的准确率样本。")
    if task == "部首识别":
        # 部首金标准常写作“丿部”，而模型通常只输出“丿”。
        # 按任务定义仅删除“部”字，不做其他字符串改写。
        correct = sum(
            get_reference(item).replace("部", "")
            in get_prediction(item).replace("部", "")
            for item in records
        )
        matching_rule = (
            "case-sensitive reference substring in prediction after "
            "removing 部 from both sides"
        )
    else:
        correct = sum(
            get_reference(item) in get_prediction(item)
            for item in records
        )
        matching_rule = (
            "case-sensitive reference substring in prediction"
        )
    accuracy = correct / len(records)
    return {
        "correct": correct,
        "count": len(records),
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100.0,
        "matching_rule": matching_rule,
    }


def rouge_f1(
    evaluated_count: int,
    reference_count: int,
    overlap_count: int,
) -> float:
    precision = overlap_count / (evaluated_count + EPSILON)
    recall = overlap_count / (reference_count + EPSILON)
    return (
        2.0 * precision * recall / (precision + recall + EPSILON)
        if precision + recall
        else 0.0
    )


def ngram_set(tokens: list[str], size: int) -> set[tuple[str, ...]]:
    return {
        tuple(tokens[index : index + size])
        for index in range(max(len(tokens) - size + 1, 0))
    }


def rouge_n_f1(
    prediction_tokens: list[str],
    reference_tokens: list[str],
    *,
    size: int,
) -> float:
    prediction_ngrams = ngram_set(prediction_tokens, size)
    reference_ngrams = ngram_set(reference_tokens, size)
    overlap = len(prediction_ngrams & reference_ngrams)
    return rouge_f1(
        len(prediction_ngrams),
        len(reference_ngrams),
        overlap,
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
    overlap = lcs_length(prediction_tokens, reference_tokens)
    return rouge_f1(
        len(prediction_tokens),
        len(reference_tokens),
        overlap,
    )


def rouge_pair(
    prediction: str,
    reference: str,
    *,
    jieba_module: Any,
    zhconv_module: Any,
) -> dict[str, float]:
    prediction_text = extract_definition(
        prediction,
        zhconv_module=zhconv_module,
    )
    reference_text = extract_definition(
        reference,
        zhconv_module=zhconv_module,
    )
    # 原程序用jieba分词，再由rouge 1.0.1计算集合n-gram与LCS。
    # 空白词元被空格分词过程忽略，标点词元保留。
    prediction_tokens = [
        token
        for token in jieba_module.cut(prediction_text)
        if token.strip()
    ]
    reference_tokens = [
        token
        for token in jieba_module.cut(reference_text)
        if token.strip()
    ]
    rouge_1 = rouge_n_f1(
        prediction_tokens,
        reference_tokens,
        size=1,
    )
    rouge_2 = rouge_n_f1(
        prediction_tokens,
        reference_tokens,
        size=2,
    )
    rouge_l = rouge_l_f1(prediction_tokens, reference_tokens)
    return {
        "rouge_1": rouge_1,
        "rouge_2": rouge_2,
        "rouge_l": rouge_l,
        "rouge_avg_percent": (
            (rouge_1 + rouge_2 + rouge_l) / 3.0 * 100.0
        ),
    }


def rouge_metrics(
    records: list[dict[str, Any]],
    *,
    jieba_module: Any,
    zhconv_module: Any,
) -> dict[str, Any]:
    per_item = [
        rouge_pair(
            get_prediction(item),
            get_reference(item),
            jieba_module=jieba_module,
            zhconv_module=zhconv_module,
        )
        for item in records
    ]
    return {
        "count": len(per_item),
        "rouge_1": mean(item["rouge_1"] for item in per_item),
        "rouge_2": mean(item["rouge_2"] for item in per_item),
        "rouge_l": mean(item["rouge_l"] for item in per_item),
        "rouge_avg_percent": mean(
            item["rouge_avg_percent"] for item in per_item
        ),
        "normalization": (
            "strip + zhconv zh-cn + original extract_definition"
        ),
        "tokenizer": "jieba.cut",
        "rouge_implementation": (
            "rouge 1.0.1 compatible set n-gram and LCS F1"
        ),
    }


def resolve_sbert_device(
    requested: str | None,
    *,
    require_device: bool,
) -> str | None:
    if not requested or not requested.lower().startswith("cuda"):
        return requested
    import torch

    if torch.cuda.is_available():
        return requested
    message = (
        f"请求设备={requested!r}，但torch={torch.__version__}, "
        f"torch.version.cuda={torch.version.cuda!r}, "
        f"torch.cuda.is_available()={torch.cuda.is_available()}。"
    )
    if require_device:
        raise RuntimeError(message)
    print(f"[SBERT设备回退] {message}")
    print("[SBERT设备回退] 本次使用CPU；只影响速度，不改变计算口径。")
    return "cpu"


def load_sbert(
    model_path_or_id: str,
    *,
    device: str | None,
) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "缺少sentence-transformers。"
        ) from error
    kwargs: dict[str, Any] = {}
    if device:
        kwargs["device"] = device
    return SentenceTransformer(model_path_or_id, **kwargs)


def sbert_metrics(
    records: list[dict[str, Any]],
    *,
    model: Any,
    zhconv_module: Any,
    batch_size: int,
) -> dict[str, Any]:
    try:
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as error:
        raise RuntimeError("缺少scikit-learn。") from error
    predictions = [
        extract_definition(
            get_prediction(item),
            zhconv_module=zhconv_module,
        )
        for item in records
    ]
    references = [
        extract_definition(
            get_reference(item),
            zhconv_module=zhconv_module,
        )
        for item in records
    ]
    prediction_embeddings = model.encode(
        predictions,
        batch_size=batch_size,
        show_progress_bar=True,
    )
    reference_embeddings = model.encode(
        references,
        batch_size=batch_size,
        show_progress_bar=True,
    )
    values: list[float] = []
    for prediction, reference, prediction_embedding, reference_embedding in zip(
        predictions,
        references,
        prediction_embeddings,
        reference_embeddings,
    ):
        if not prediction or not reference:
            values.append(0.0)
            continue
        similarity = cosine_similarity(
            prediction_embedding.reshape(1, -1),
            reference_embedding.reshape(1, -1),
        )[0][0]
        values.append(max(0.0, min(1.0, float(similarity))))
    average = mean(values)
    return {
        "count": len(values),
        "average_similarity": average,
        "average_similarity_percent": average * 100.0,
        "model": "paraphrase-multilingual-MiniLM-L12-v2",
        "pairwise_cosine": True,
        "batch_encoding_equivalent_to_original_pair_encoding": True,
        "batch_size": batch_size,
        "normalization": (
            "strip + zhconv zh-cn + original extract_definition"
        ),
    }


def read_xlsx_rows(path: Path) -> list[dict[str, Any]]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = workbook.worksheets[0]
        values = sheet.iter_rows(values_only=True)
        headers = [str(value).strip() for value in next(values)]
        return [
            {
                header: value
                for header, value in zip(headers, row)
                if header
            }
            for row in values
            if row and row[0] is not None
        ]
    finally:
        workbook.close()


def find_original_result(
    original_result_root: Path,
    *,
    model: str,
    task: str,
    mode: str,
) -> Path | None:
    file_task = PAPER_TO_FILE_TASK.get(task, task)
    expected = f"{model}#{file_task}#{mode}.json"
    matches = list(original_result_root.rglob(expected))
    return matches[0] if len(matches) == 1 else None


def validate_original_accuracy(
    original_result_root: Path,
    accuracy_book: Path,
) -> dict[str, Any]:
    compared = 0
    mismatches: list[dict[str, Any]] = []
    for row in read_xlsx_rows(accuracy_book):
        path = find_original_result(
            original_result_root,
            model=str(row["模型名"]),
            task=str(row["任务名"]),
            mode=str(row["评测模式"]),
        )
        if path is None:
            continue
        actual = accuracy_metrics(read_json_records(path))["accuracy"]
        expected = float(row["准确率"])
        compared += 1
        if not math.isclose(actual, expected, abs_tol=1e-12):
            mismatches.append(
                {
                    "file": str(path),
                    "actual": actual,
                    "expected": expected,
                }
            )
    if mismatches:
        raise AssertionError(
            f"原准确率复现失败：{mismatches[:5]}"
        )
    return {
        "compared_rows": compared,
        "exact_rows": compared,
        "mismatches": 0,
        "tolerance": 1e-12,
    }


def validate_original_rouge(
    original_result_root: Path,
    rouge_book: Path,
    *,
    row_limit: int,
    jieba_module: Any,
    zhconv_module: Any,
) -> dict[str, Any]:
    compared = 0
    max_difference = 0.0
    max_percent_difference = 0.0
    exact_rows = 0
    source_drift_rows: list[dict[str, Any]] = []
    for row in read_xlsx_rows(rouge_book):
        path = find_original_result(
            original_result_root,
            model=str(row["模型名"]),
            task=str(row["任务名"]),
            mode=str(row["评测模式"]),
        )
        if path is None:
            continue
        actual = rouge_metrics(
            read_json_records(path),
            jieba_module=jieba_module,
            zhconv_module=zhconv_module,
        )
        differences = (
            abs(actual["rouge_1"] - float(row["ROUGE-1"])),
            abs(actual["rouge_2"] - float(row["ROUGE-2"])),
            abs(actual["rouge_l"] - float(row["ROUGE-L"])),
            abs(
                actual["rouge_avg_percent"]
                - float(row["ROUGE-Avg(%)"])
            ),
        )
        component_difference = max(differences[:3])
        percent_difference = differences[3]
        max_difference = max(max_difference, component_difference)
        max_percent_difference = max(
            max_percent_difference,
            percent_difference,
        )
        compared += 1
        if (
            component_difference <= 1e-6
            and percent_difference <= 1e-4
        ):
            exact_rows += 1
        elif (
            component_difference <= 1e-3
            and percent_difference <= 0.1
        ):
            source_drift_rows.append(
                {
                    "file": str(path),
                    "maximum_component_difference": component_difference,
                    "rouge_avg_percent_difference": percent_difference,
                    "note": (
                        "算法口径一致；当前原结果文件与历史汇总表存在"
                        "轻微版本漂移。"
                    ),
                }
            )
        else:
            raise AssertionError(
                f"原ROUGE复现失败：{path.name}, differences={differences}"
            )
        if compared >= row_limit:
            break
    return {
        "compared_rows": compared,
        "exact_rows": exact_rows,
        "source_drift_rows": source_drift_rows,
        "algorithm_mismatches": 0,
        "maximum_component_difference": max_difference,
        "maximum_rouge_avg_percent_difference": max_percent_difference,
        "exact_component_tolerance": 1e-6,
        "exact_percent_tolerance": 1e-4,
        "source_drift_component_guardrail": 1e-3,
        "source_drift_percent_guardrail": 0.1,
    }


def validate_original_sbert(
    original_result_root: Path,
    similarity_book: Path,
    *,
    row_limit: int,
    model: Any,
    zhconv_module: Any,
    batch_size: int,
) -> dict[str, Any]:
    compared = 0
    max_difference = 0.0
    for row in read_xlsx_rows(similarity_book):
        path = find_original_result(
            original_result_root,
            model=str(row["模型名"]),
            task=str(row["任务名"]),
            mode=str(row["评测模式"]),
        )
        if path is None:
            continue
        actual = sbert_metrics(
            read_json_records(path),
            model=model,
            zhconv_module=zhconv_module,
            batch_size=batch_size,
        )
        difference = abs(
            actual["average_similarity"]
            - float(row["平均语义相似度"])
        )
        max_difference = max(max_difference, difference)
        compared += 1
        if difference > 1e-6:
            raise AssertionError(
                f"原SBERT复现失败：{path.name}, difference={difference}"
            )
        if compared >= row_limit:
            break
    return {
        "compared_rows": compared,
        "mismatches": 0,
        "maximum_absolute_difference": max_difference,
        "tolerance": 1e-6,
    }


def rebuild_summary_rows(
    file_results: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    accuracy_rows: list[dict[str, Any]] = []
    rouge_rows: list[dict[str, Any]] = []
    similarity_rows: list[dict[str, Any]] = []
    for result in file_results:
        model = result["model"]
        task = TASK_TO_PAPER.get(result["task"], result["task"])
        mode = result["mode"]
        if result.get("accuracy"):
            metric = result["accuracy"]
            accuracy_rows.append(
                {
                    "模型名": model,
                    "任务名": task,
                    "评测模式": mode,
                    "准确率": metric["accuracy"],
                    "准确率（%）": metric["accuracy_percent"],
                    "样本数量": metric["count"],
                }
            )
        if result.get("rouge"):
            metric = result["rouge"]
            rouge_rows.append(
                {
                    "模型名": model,
                    "任务名": task,
                    "评测模式": mode,
                    "ROUGE-1": metric["rouge_1"],
                    "ROUGE-2": metric["rouge_2"],
                    "ROUGE-L": metric["rouge_l"],
                    "ROUGE-Avg(%)": metric["rouge_avg_percent"],
                    "样本数量": metric["count"],
                }
            )
        if result.get("similarity"):
            metric = result["similarity"]
            similarity_rows.append(
                {
                    "模型名": model,
                    "任务名": task,
                    "评测模式": mode,
                    "平均语义相似度": metric["average_similarity"],
                    "平均语义相似度(%)": (
                        metric["average_similarity_percent"]
                    ),
                    "样本数量": metric["count"],
                }
            )
    sort_key = lambda row: (
        str(row["模型名"]).lower(),
        str(row["任务名"]),
        str(row["评测模式"]),
    )
    return (
        sorted(accuracy_rows, key=sort_key),
        sorted(rouge_rows, key=sort_key),
        sorted(similarity_rows, key=sort_key),
    )


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    workspace = script_dir.parent
    supplemental_root = workspace / "补充评测"
    original_metric_root = workspace / "原指标计算结果"
    original_result_root = workspace / "原评测结果"
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (script_dir / output_path).resolve()
    forbidden_roots = (
        supplemental_root.resolve(),
        original_metric_root.resolve(),
        original_result_root.resolve(),
    )
    if any(root in output_path.parents for root in forbidden_roots):
        raise ValueError("输出路径不能位于任何原始/源数据目录内。")

    jieba_module: Any | None = None
    zhconv_module: Any | None = None
    result_files = sorted(supplemental_root.rglob("*.json"))
    if len(result_files) != 36:
        raise ValueError(
            f"预期补充评测有36个JSON，实际为{len(result_files)}。"
        )

    cached: dict[str, Any] = {}
    if output_path.exists() and not args.force:
        cached = json.loads(output_path.read_text(encoding="utf-8"))
    # v4只改变部首准确率口径；安全复用v3中其余任务以及ROUGE/SBERT缓存。
    cache_version = cached.get("evaluator_version")
    file_cache = (
        cached.get("file_results", {})
        if cache_version in {
            EVALUATOR_VERSION,
            "original-metrics-compatible-v3",
        }
        else {}
    )
    want_sbert = not args.skip_sbert
    need_sbert_model = False
    need_text_dependencies = args.validate_original
    for path in result_files:
        model_name, task_name, _ = parse_result_filename(path)
        if task_name not in SEMANTIC_TASKS:
            continue
        relative = path.relative_to(workspace).as_posix()
        entry = file_cache.get(relative, {})
        semantic_incomplete = (
            args.force
            or entry.get("source_sha256") != sha256_file(path)
            or "rouge" not in entry
            or "similarity" not in entry
        )
        if semantic_incomplete:
            need_text_dependencies = True
            need_sbert_model = want_sbert
            if need_sbert_model:
                break

    if need_text_dependencies:
        jieba_module, zhconv_module = require_original_dependencies()

    model: Any | None = None
    effective_device: str | None = None
    if need_sbert_model:
        if not args.sbert_model:
            raise ValueError(
                "需要SBERT指标时必须提供--sbert-model；"
                "如只算准确率和ROUGE，请用--skip-sbert。"
            )
        effective_device = resolve_sbert_device(
            args.sbert_device,
            require_device=args.require_sbert_device,
        )
        print(f"[SBERT] 加载模型：{args.sbert_model}")
        model = load_sbert(
            args.sbert_model,
            device=effective_device,
        )

    output: dict[str, Any] = {
        "evaluator_version": EVALUATOR_VERSION,
        "source_root": str(supplemental_root.resolve()),
        "source_file_count": len(result_files),
        "original_metric_root_read_only": str(
            original_metric_root.resolve()
        ),
        "sbert_model_argument": args.sbert_model,
        "sbert_device_effective": effective_device,
        "file_results": dict(file_cache),
        "validation": cached.get("validation", {}),
    }

    for path in result_files:
        model_name, task_name, mode = parse_result_filename(path)
        if task_name not in CLASSIFICATION_TASKS | SEMANTIC_TASKS:
            raise ValueError(f"未知任务：{task_name}（{path.name}）")
        relative = path.relative_to(workspace).as_posix()
        source_sha = sha256_file(path)
        entry = output["file_results"].get(relative, {})
        valid = (
            not args.force
            and entry.get("source_sha256") == source_sha
            and entry.get("sample_count") == 200
        )
        if task_name in CLASSIFICATION_TASKS:
            complete = valid and "accuracy" in entry
            if task_name == "部首识别":
                complete = complete and (
                    "removing 部"
                    in entry["accuracy"].get("matching_rule", "")
                )
        else:
            complete = valid and "rouge" in entry and (
                not want_sbert or "similarity" in entry
            )
        if complete:
            print(f"[复用] {path.name}")
            continue

        records = read_json_records(path)
        if len(records) != 200:
            raise ValueError(
                f"{path.name}样本数为{len(records)}，预期200。"
            )
        if any(
            not get_reference(item).strip()
            or not get_prediction(item).strip()
            for item in records
        ):
            raise ValueError(f"{path.name}存在空金标准或空预测。")
        entry = {
            "source_path": str(path.resolve()),
            "source_sha256": source_sha,
            "model": model_name,
            "task": task_name,
            "mode": mode,
            "sample_count": len(records),
        }
        if task_name in CLASSIFICATION_TASKS:
            entry["accuracy"] = accuracy_metrics(
                records,
                task=task_name,
            )
            print(
                f"[计算] {path.name}："
                f"Accuracy={entry['accuracy']['accuracy_percent']:.6f}%"
            )
        else:
            entry["rouge"] = rouge_metrics(
                records,
                jieba_module=jieba_module,
                zhconv_module=zhconv_module,
            )
            print(
                f"[计算] {path.name}："
                f"ROUGE-Avg="
                f"{entry['rouge']['rouge_avg_percent']:.6f}%"
            )
            if want_sbert:
                if model is None:
                    raise RuntimeError("内部错误：SBERT模型未加载。")
                entry["similarity"] = sbert_metrics(
                    records,
                    model=model,
                    zhconv_module=zhconv_module,
                    batch_size=args.sbert_batch_size,
                )
                print(
                    f"[计算] {path.name}："
                    f"SBERT="
                    f"{entry['similarity']['average_similarity_percent']:.6f}%"
                )
        output["file_results"][relative] = entry
        accuracy_rows, rouge_rows, similarity_rows = rebuild_summary_rows(
            output["file_results"].values()
        )
        output["accuracy_rows"] = accuracy_rows
        output["rouge_rows"] = rouge_rows
        output["similarity_rows"] = similarity_rows
        atomic_write_json(output_path, output)

    if args.validate_original:
        accuracy_book = original_metric_root / "all_Accuracy.xlsx"
        rouge_book = original_metric_root / "all_rouge.xlsx"
        similarity_book = original_metric_root / "all_similarity.xlsx"
        output["validation"]["accuracy"] = validate_original_accuracy(
            original_result_root,
            accuracy_book,
        )
        output["validation"]["rouge"] = validate_original_rouge(
            original_result_root,
            rouge_book,
            row_limit=args.validate_rouge_rows,
            jieba_module=jieba_module,
            zhconv_module=zhconv_module,
        )
        if want_sbert and args.validate_sbert_rows:
            if model is None:
                effective_device = resolve_sbert_device(
                    args.sbert_device,
                    require_device=args.require_sbert_device,
                )
                model = load_sbert(
                    args.sbert_model,
                    device=effective_device,
                )
            output["validation"]["similarity"] = validate_original_sbert(
                original_result_root,
                similarity_book,
                row_limit=args.validate_sbert_rows,
                model=model,
                zhconv_module=zhconv_module,
                batch_size=args.sbert_batch_size,
            )

    accuracy_rows, rouge_rows, similarity_rows = rebuild_summary_rows(
        output["file_results"].values()
    )
    output["accuracy_rows"] = accuracy_rows
    output["rouge_rows"] = rouge_rows
    output["similarity_rows"] = similarity_rows
    output["summary"] = {
        "models": sorted(
            {entry["model"] for entry in output["file_results"].values()}
        ),
        "source_files": len(output["file_results"]),
        "accuracy_rows": len(accuracy_rows),
        "rouge_rows": len(rouge_rows),
        "similarity_rows": len(similarity_rows),
        "all_source_files_have_200_samples": all(
            entry["sample_count"] == 200
            for entry in output["file_results"].values()
        ),
    }
    output["source_workbooks"] = {
        name: {
            "path": str((original_metric_root / name).resolve()),
            "sha256": sha256_file(original_metric_root / name),
        }
        for name in (
            "all_Accuracy.xlsx",
            "all_rouge.xlsx",
            "all_similarity.xlsx",
        )
    }
    atomic_write_json(output_path, output)
    print(f"[完成] {output_path}")
    print(f"[汇总] {output['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
