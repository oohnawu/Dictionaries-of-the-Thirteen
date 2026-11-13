#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Evaluate completed definition-RAG predictions against the unchanged gold.

This program is intentionally separate from generation. It loads the
``所有含义``/``output`` gold answers only after a prediction file exists,
converts both predictions and references to Simplified Chinese, computes
transparent ROUGE-1/2/L F1 scores, and reports overall and retrieval-coverage
strata. Sentence-BERT cosine similarity is optional.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable

from baseline_common import (
    atomic_write_json,
    clean,
    compact_key,
    load_config,
    prediction_is_complete,
    read_json_list,
    resolve_path,
    to_simplified,
)


METRIC_FIELDS = (
    "rouge_1_f1",
    "rouge_2_f1",
    "rouge_l_f1",
    "rouge_avg_f1",
    "exact_match_simplified",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="baseline_config.json")
    parser.add_argument(
        "--predictions",
        required=True,
        help="run_qwen3_8b_rag.py生成的释义能力JSON结果。",
    )
    parser.add_argument(
        "--gold",
        default=None,
        help="覆盖现有释义测试JSON路径；通常无需设置。",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="汇总JSON路径；默认写在预测文件旁边。",
    )
    parser.add_argument(
        "--tokenizer",
        choices=("jieba", "character"),
        default="jieba",
        help="ROUGE分词方式；默认jieba，结果中会记录该设置。",
    )
    parser.add_argument(
        "--sbert-model",
        default=None,
        help=(
            "可选的本地Sentence-BERT路径或HF ID，例如"
            "paraphrase-multilingual-MiniLM-L12-v2。"
        ),
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="只评估已完成记录；正式实验不建议使用。",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_text(value: Any) -> str:
    return to_simplified(clean(value))


def character_tokens(value: Any) -> list[str]:
    return list(compact_key(normalized_text(value)))


def jieba_tokens(value: Any) -> list[str]:
    try:
        import jieba
    except ImportError as error:
        raise RuntimeError("jieba分词需要安装jieba。") from error
    tokens: list[str] = []
    for token in jieba.lcut(normalized_text(value), cut_all=False):
        normalized = compact_key(token)
        if normalized:
            tokens.append(normalized)
    return tokens


def make_tokenizer(name: str) -> Callable[[Any], list[str]]:
    return jieba_tokens if name == "jieba" else character_tokens


def ngrams(tokens: list[str], size: int) -> list[tuple[str, ...]]:
    if size < 1:
        raise ValueError("ngram size必须大于0。")
    return [
        tuple(tokens[index : index + size])
        for index in range(max(len(tokens) - size + 1, 0))
    ]


def overlap_f1(
    prediction_tokens: list[str],
    reference_tokens: list[str],
    *,
    ngram_size: int,
) -> float:
    prediction_counts = Counter(ngrams(prediction_tokens, ngram_size))
    reference_counts = Counter(ngrams(reference_tokens, ngram_size))
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


def score_pair(
    prediction: str,
    reference: str,
    *,
    tokenize: Callable[[Any], list[str]],
) -> dict[str, float]:
    prediction_tokens = tokenize(prediction)
    reference_tokens = tokenize(reference)
    rouge_1 = overlap_f1(
        prediction_tokens,
        reference_tokens,
        ngram_size=1,
    )
    rouge_2 = overlap_f1(
        prediction_tokens,
        reference_tokens,
        ngram_size=2,
    )
    rouge_l = rouge_l_f1(prediction_tokens, reference_tokens)
    return {
        "rouge_1_f1": rouge_1,
        "rouge_2_f1": rouge_2,
        "rouge_l_f1": rouge_l,
        "rouge_avg_f1": (rouge_1 + rouge_2 + rouge_l) / 3.0,
        "exact_match_simplified": float(
            compact_key(normalized_text(prediction))
            == compact_key(normalized_text(reference))
        ),
        "prediction_tokens": float(len(prediction_tokens)),
        "reference_tokens": float(len(reference_tokens)),
    }


def evidence_count_bin(count: int) -> str:
    if count == 0:
        return "0"
    if count <= 5:
        return "1-5"
    if count <= 10:
        return "6-10"
    if count <= 20:
        return "11-20"
    return "21+"


def average_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows_list = list(rows)
    summary: dict[str, Any] = {"count": len(rows_list)}
    for field in METRIC_FIELDS:
        summary[field] = (
            round(mean(float(row[field]) for row in rows_list), 6)
            if rows_list
            else None
        )
        summary[f"{field}_percent"] = (
            round(100.0 * summary[field], 3)
            if summary[field] is not None
            else None
        )
    if rows_list:
        summary["average_prediction_tokens"] = round(
            mean(float(row["prediction_tokens"]) for row in rows_list),
            3,
        )
        summary["average_reference_tokens"] = round(
            mean(float(row["reference_tokens"]) for row in rows_list),
            3,
        )
    else:
        summary["average_prediction_tokens"] = None
        summary["average_reference_tokens"] = None
    if any("sbert_cosine" in row for row in rows_list):
        values = [
            float(row["sbert_cosine"])
            for row in rows_list
            if "sbert_cosine" in row
        ]
        summary["sbert_cosine"] = (
            round(mean(values), 6) if values else None
        )
        summary["sbert_cosine_percent"] = (
            round(100.0 * summary["sbert_cosine"], 3)
            if summary["sbert_cosine"] is not None
            else None
        )
    return summary


def add_sbert_scores(
    rows: list[dict[str, Any]],
    *,
    model_path_or_id: str,
) -> None:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "Sentence-BERT评估需要安装sentence-transformers。"
        ) from error
    model = SentenceTransformer(model_path_or_id)
    predictions = [normalized_text(row["prediction"]) for row in rows]
    references = [normalized_text(row["reference"]) for row in rows]
    prediction_embeddings = model.encode(
        predictions,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    reference_embeddings = model.encode(
        references,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    for row, prediction_embedding, reference_embedding in zip(
        rows,
        prediction_embeddings,
        reference_embeddings,
    ):
        similarity = float(
            sum(
                float(left) * float(right)
                for left, right in zip(
                    prediction_embedding,
                    reference_embedding,
                )
            )
        )
        row["sbert_cosine"] = max(0.0, min(1.0, similarity))


def evaluate(
    predictions: list[dict[str, Any]],
    gold_items: list[dict[str, Any]],
    *,
    tokenizer_name: str,
    allow_incomplete: bool,
    sbert_model: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(predictions) != len(gold_items):
        raise ValueError(
            f"预测数({len(predictions)})与金标准数({len(gold_items)})不一致。"
        )
    tokenize = make_tokenizer(tokenizer_name)
    rows: list[dict[str, Any]] = []
    incomplete_indices: list[int] = []
    for index, (prediction_item, gold_item) in enumerate(
        zip(predictions, gold_items),
        start=1,
    ):
        if clean(prediction_item.get("input")) != clean(gold_item.get("input")):
            raise ValueError(f"第{index}条预测与金标准字头不一致。")
        if not prediction_is_complete(prediction_item):
            incomplete_indices.append(index)
            continue
        prediction = clean(prediction_item["prediction"])
        reference = clean(gold_item.get("output"))
        if not reference:
            raise ValueError(f"第{index}条金标准output为空。")
        retrieval = prediction_item.get("retrieval", [])
        if not isinstance(retrieval, list):
            raise ValueError(f"第{index}条retrieval不是列表。")
        row: dict[str, Any] = {
            "index": index,
            "headword": clean(gold_item.get("input")),
            "prediction": prediction,
            "reference": reference,
            "definition_mode": clean(
                prediction_item.get("definition_mode")
            ),
            "rag_method": clean(prediction_item.get("rag_method")),
            "retrieval_scope": clean(
                prediction_item.get("retrieval_scope")
            )
            or "unknown",
            "retrieval_hit": bool(retrieval),
            "evidence_count": len(retrieval),
            "evidence_count_bin": evidence_count_bin(len(retrieval)),
        }
        row.update(score_pair(prediction, reference, tokenize=tokenize))
        rows.append(row)

    if incomplete_indices and not allow_incomplete:
        preview = incomplete_indices[:10]
        raise ValueError(
            f"预测文件有{len(incomplete_indices)}条未完成，"
            f"前10个序号={preview}。正式评估前请先完成生成；"
            "如仅调试可使用--allow-incomplete。"
        )
    if not rows:
        raise ValueError("没有可评估的已完成预测。")
    if sbert_model:
        add_sbert_scores(rows, model_path_or_id=sbert_model)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped["all"].append(row)
        grouped[
            "retrieval_hit" if row["retrieval_hit"] else "no_evidence"
        ].append(row)
        grouped[f"scope:{row['retrieval_scope']}"].append(row)
        grouped[f"evidence_count:{row['evidence_count_bin']}"].append(row)
    summary = {
        "evaluated_items": len(rows),
        "incomplete_items": len(incomplete_indices),
        "retrieval_hit_items": sum(row["retrieval_hit"] for row in rows),
        "no_evidence_items": sum(not row["retrieval_hit"] for row in rows),
        "tokenizer": tokenizer_name,
        "traditional_to_simplified_before_scoring": True,
        "sbert_model": sbert_model,
        "groups": {
            name: average_metrics(group_rows)
            for name, group_rows in sorted(grouped.items())
        },
    }
    return summary, rows


def main() -> int:
    args = parse_args()
    config, config_base = load_config(Path(args.config))
    predictions_path = Path(args.predictions).resolve()
    if not predictions_path.exists():
        raise FileNotFoundError(f"预测文件不存在：{predictions_path}")
    eval_root = resolve_path(config_base, config["eval_root"])
    gold_path = (
        Path(args.gold).resolve()
        if args.gold
        else (
            eval_root
            / "释义能力"
            / "0-shot"
            / "释义能力_eval.json"
        ).resolve()
    )
    output_path = (
        Path(args.output).resolve()
        if args.output
        else predictions_path.with_name(
            predictions_path.stem + "_evaluation.json"
        )
    )
    per_item_path = output_path.with_name(
        output_path.stem + "_per_item.json"
    )

    # Gold answers are first read here, after the prediction file already exists.
    predictions = read_json_list(predictions_path)
    gold_items = read_json_list(gold_path)
    summary, rows = evaluate(
        predictions,
        gold_items,
        tokenizer_name=args.tokenizer,
        allow_incomplete=args.allow_incomplete,
        sbert_model=clean(args.sbert_model) or None,
    )
    report = {
        "evaluation_type": "cross-dictionary-definition-rag",
        "prediction_path": str(predictions_path),
        "prediction_sha256": sha256_file(predictions_path),
        "gold_path": str(gold_path),
        "gold_sha256": sha256_file(gold_path),
        **summary,
        "metric_definition": {
            "normalization": "NFKC then OpenCC t2s",
            "rouge_1_2": "multiset n-gram overlap F1",
            "rouge_l": "longest-common-subsequence F1",
            "rouge_avg": "mean(ROUGE-1 F1, ROUGE-2 F1, ROUGE-L F1)",
        },
    }
    atomic_write_json(output_path, report)
    atomic_write_json(per_item_path, rows)
    print(f"[完成] 汇总：{output_path}")
    print(f"[完成] 逐条：{per_item_path}")
    print(
        f"[总体] n={summary['groups']['all']['count']}; "
        f"ROUGE-Avg={summary['groups']['all']['rouge_avg_f1_percent']}%; "
        f"有证据={summary['retrieval_hit_items']}; "
        f"无证据={summary['no_evidence_items']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
