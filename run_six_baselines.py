#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run non-LLM baselines for the six tasks on the 0-shot test sets.

Baselines:
  1. 部首识别: exact dictionary lookup + global-majority fallback
  2. 拼音注音: exact dictionary lookup + global-majority fallback
  3. 反切注音: exact dictionary lookup + global-majority fallback
  4a. 词性识别: headword-majority POS after near-duplicate filtering
  4b. 词性识别: context BM25 with same-headword-first retrieval
  5. 释义能力: BM25 Top-1 after excluding every test headword
  6. 语境理解: near-duplicate-filtered, headword-first BM25 Top-1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from baseline_common import (
    BM25Index,
    HeadwordFirstBM25,
    atomic_write_json,
    build_context_documents,
    build_definition_documents,
    build_majority_lookup,
    build_pos_context_documents,
    clean,
    extract_example,
    filter_records_by_test_examples,
    headword_first_retrieval_results,
    load_config,
    read_json_list,
    read_xlsx_records,
    require_fields,
    resolve_path,
    retrieval_results,
    simplified_answer_key,
    surface_answer_key,
)


TASKS = (
    "部首识别",
    "拼音注音",
    "反切注音",
    "词性识别",
    "释义能力",
    "语境理解",
)
LOOKUP_TASKS = ("部首识别", "拼音注音", "反切注音")

OUTPUT_NAMES = {
    "部首识别": "dictionary-lookup",
    "拼音注音": "dictionary-lookup",
    "反切注音": "dictionary-lookup",
    "词性识别": "headword-majority-pos-strict",
    "释义能力": "bm25-top1-strict",
    "语境理解": "bm25-top1-strict",
}

POS_CONTEXT_OUTPUT_NAME = "bm25-context-pos-strict"

METHOD_NAMES = {
    "部首识别": "openbook-dictionary-majority",
    "拼音注音": "openbook-dictionary-majority",
    "反切注音": "openbook-dictionary-majority",
    "词性识别": "headword-majority-pos-strict",
    "释义能力": "bm25-top1-strict-headword-disjoint",
    "语境理解": "bm25-top1-nearduplicate-filtered-headword-first",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="baseline_config.json")
    parser.add_argument(
        "--task",
        action="append",
        choices=TASKS,
        help="只运行指定任务；可重复。默认运行全部六项。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="检查评测集、资源字段和严格排除后的语料规模，不写结果。",
    )
    return parser.parse_args()


def eval_path(eval_root: Path, task: str) -> Path:
    return eval_root / task / "0-shot" / f"{task}_eval.json"


def output_path(output_root: Path, task: str) -> Path:
    name = OUTPUT_NAMES[task]
    return output_root / task / "0-shot" / f"{name}#{task}#0-shot.json"


def pos_context_output_path(output_root: Path) -> Path:
    task = "词性识别"
    return (
        output_root
        / task
        / "0-shot"
        / f"{POS_CONTEXT_OUTPUT_NAME}#{task}#0-shot.json"
    )


def score_predictions(items: list[dict[str, Any]]) -> dict[str, Any]:
    simplified_exact = sum(
        simplified_answer_key(item.get("prediction"))
        == simplified_answer_key(item.get("output"))
        for item in items
    )
    surface_exact = sum(
        surface_answer_key(item.get("prediction"))
        == surface_answer_key(item.get("output"))
        for item in items
    )
    return {
        "items": len(items),
        "exact_match_count": simplified_exact,
        "exact_match_accuracy": (
            simplified_exact / len(items) if items else 0.0
        ),
        "surface_exact_match_count": surface_exact,
        "surface_exact_match_accuracy": (
            surface_exact / len(items) if items else 0.0
        ),
        "accuracy_normalization": (
            "Unicode NFKC + OpenCC t2s + whitespace removal"
        ),
    }


def extract_test_examples(
    items: list[dict[str, Any]],
    task: str,
) -> list[str]:
    examples: list[str] = []
    for index, item in enumerate(items, start=1):
        example = extract_example(clean(item.get("instruction")))
        if not example:
            raise ValueError(f"{task}第{index}条无法从instruction提取古文例句。")
        examples.append(example)
    return examples


def run_lookup(
    task: str,
    items: list[dict[str, Any]],
    resource: dict[str, Any],
    base: Path,
    dry_run: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = resolve_path(base, resource["path"])
    records = read_xlsx_records(path, resource.get("sheet_name"))
    key_field = resource["key_field"]
    value_field = resource["value_field"]
    require_fields(records, (key_field, value_field), path)
    lookup = build_majority_lookup(records, key_field, value_field)

    if dry_run:
        return [], {
            "resource": str(path),
            "resource_rows": len(records),
            "lookup_keys": len(lookup.prediction_by_key),
        }

    results: list[dict[str, Any]] = []
    hit_count = 0
    candidate_match_count = 0
    ambiguous_count = 0
    top_frequency_tie_count = 0
    for source in items:
        item = dict(source)
        key = clean(item.get("input"))
        hit = key in lookup.prediction_by_key
        if hit:
            hit_count += 1
        prediction = (
            lookup.prediction_by_key[key]
            if hit
            else lookup.global_prediction
        )
        candidate_pairs = lookup.candidates_by_key.get(key, [])
        candidate_contains_gold = any(
            simplified_answer_key(value)
            == simplified_answer_key(item.get("output"))
            for value, _ in candidate_pairs
        )
        ambiguous = len(candidate_pairs) > 1
        top_frequency_tie = (
            len(candidate_pairs) > 1
            and candidate_pairs[0][1] == candidate_pairs[1][1]
        )
        candidate_match_count += int(candidate_contains_gold)
        ambiguous_count += int(ambiguous)
        top_frequency_tie_count += int(top_frequency_tie)
        item.update(
            {
                "prediction": prediction,
                "baseline_method": "openbook-dictionary-majority",
                "lookup_hit": hit,
                "lookup_candidates": [
                    {"value": value, "count": count}
                    for value, count in candidate_pairs
                ],
                "candidate_contains_gold": candidate_contains_gold,
                "ambiguous_lookup": ambiguous,
                "top_frequency_tie": top_frequency_tie,
                "fallback_used": not hit,
                "leakage_control": (
                    "Full structured dictionary; report as an open-book lookup "
                    "baseline, not as a closed-book predictor."
                ),
            }
        )
        results.append(item)

    summary = {
        "resource": str(path),
        "resource_rows": len(records),
        "lookup_keys": len(lookup.prediction_by_key),
        "lookup_hits": hit_count,
        "lookup_coverage": hit_count / len(items) if items else 0.0,
        "candidate_set_match_count": candidate_match_count,
        "candidate_set_match_rate": (
            candidate_match_count / len(items) if items else 0.0
        ),
        "ambiguous_lookup_count": ambiguous_count,
        "ambiguous_lookup_rate": (
            ambiguous_count / len(items) if items else 0.0
        ),
        "top_frequency_tie_count": top_frequency_tie_count,
        "top_frequency_tie_rate": (
            top_frequency_tie_count / len(items) if items else 0.0
        ),
        **score_predictions(results),
    }
    return results, summary


def run_pos(
    items: list[dict[str, Any]],
    resource: dict[str, Any],
    leakage_filter_config: dict[str, Any],
    base: Path,
    dry_run: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = resolve_path(base, resource["path"])
    records = read_xlsx_records(path, resource.get("sheet_name"))
    key_field = resource["key_field"]
    value_field = resource["value_field"]
    example_field = resource["example_field"]
    require_fields(records, (key_field, value_field, example_field), path)

    test_examples = extract_test_examples(items, "词性识别")
    strict_records, filter_stats = filter_records_by_test_examples(
        records,
        example_field=example_field,
        test_examples=test_examples,
        similarity_threshold=float(
            leakage_filter_config.get("similarity_threshold", 0.90)
        ),
        containment_min_chars=int(
            leakage_filter_config.get("containment_min_chars", 6)
        ),
        ngram_size=int(leakage_filter_config.get("ngram_size", 3)),
    )
    lookup = build_majority_lookup(strict_records, key_field, value_field)

    if dry_run:
        return [], {
            "resource": str(path),
            "resource_rows": len(records),
            "strict_resource_rows": len(strict_records),
            "filter_stats": filter_stats,
            "lookup_keys": len(lookup.prediction_by_key),
        }

    results: list[dict[str, Any]] = []
    hit_count = 0
    for source in items:
        item = dict(source)
        key = clean(item.get("input"))
        hit = key in lookup.prediction_by_key
        if hit:
            hit_count += 1
        item.update(
            {
                "prediction": (
                    lookup.prediction_by_key[key]
                    if hit
                    else lookup.global_prediction
                ),
                "baseline_method": "headword-majority-pos-strict",
                "lookup_hit": hit,
                "lookup_candidates": [
                    {"value": value, "count": count}
                    for value, count in lookup.candidates_by_key.get(key, [])
                ],
                "fallback_used": not hit,
                "leakage_control": (
                    "Before POS counting, resource examples were removed when "
                    "they exactly matched, contained/were contained by, or had "
                    "character similarity above the configured threshold to "
                    "any test sentence."
                ),
            }
        )
        results.append(item)

    summary = {
        "resource": str(path),
        "resource_rows": len(records),
        "strict_resource_rows": len(strict_records),
        "filter_stats": filter_stats,
        "lookup_keys": len(lookup.prediction_by_key),
        "lookup_hits": hit_count,
        "lookup_coverage": hit_count / len(items) if items else 0.0,
        **score_predictions(results),
    }
    return results, summary


def run_pos_context_bm25(
    items: list[dict[str, Any]],
    resource: dict[str, Any],
    bm25_config: dict[str, Any],
    leakage_filter_config: dict[str, Any],
    base: Path,
    dry_run: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Retrieve a non-duplicate context and copy its POS label."""
    path = resolve_path(base, resource["path"])
    records = read_xlsx_records(path, resource.get("sheet_name"))
    key_field = resource["key_field"]
    pos_field = resource["value_field"]
    example_field = resource["example_field"]
    require_fields(records, (key_field, pos_field, example_field), path)

    test_examples = extract_test_examples(items, "词性识别")
    strict_records, filter_stats = filter_records_by_test_examples(
        records,
        example_field=example_field,
        test_examples=test_examples,
        similarity_threshold=float(
            leakage_filter_config.get("similarity_threshold", 0.90)
        ),
        containment_min_chars=int(
            leakage_filter_config.get("containment_min_chars", 6)
        ),
        ngram_size=int(leakage_filter_config.get("ngram_size", 3)),
    )
    documents = build_pos_context_documents(
        strict_records,
        key_field=key_field,
        pos_field=pos_field,
        example_field=example_field,
    )
    majority_lookup = build_majority_lookup(
        strict_records,
        key_field,
        pos_field,
    )

    if dry_run:
        return [], {
            "resource": str(path),
            "resource_rows": len(records),
            "strict_documents": len(documents),
            "filter_stats": filter_stats,
            "fallback_label": majority_lookup.global_prediction,
            "retrieval_text_fields": [key_field, example_field],
            "label_payload_field": pos_field,
        }

    index = HeadwordFirstBM25(
        documents,
        k1=float(bm25_config.get("k1", 1.5)),
        b=float(bm25_config.get("b", 0.75)),
    )
    top_k = int(bm25_config.get("top_k", 3))
    results: list[dict[str, Any]] = []
    scope_counts: dict[str, int] = {
        "same-headword": 0,
        "global-backoff": 0,
    }
    positive_retrievals = 0
    fallback_count = 0

    for source in items:
        item = dict(source)
        headword = clean(item.get("input"))
        example = extract_example(clean(item.get("instruction")))
        query = f"{headword} {example}"
        retrieval_scope, retrieved = headword_first_retrieval_results(
            index,
            query,
            headword=headword,
            top_k=top_k,
        )
        scope_counts[retrieval_scope] += 1
        if retrieved:
            positive_retrievals += 1
            prediction = retrieved[0]["answer"]
            fallback_used = False
        else:
            fallback_count += 1
            prediction = majority_lookup.global_prediction
            fallback_used = True
        item.update(
            {
                "prediction": prediction,
                "baseline_method": (
                    "bm25-top1-pos-nearduplicate-filtered-headword-first"
                ),
                "query_example": example,
                "retrieval_scope": retrieval_scope,
                "retrieval_hit": bool(retrieved),
                "retrieval": retrieved,
                "fallback_used": fallback_used,
                "fallback_label": majority_lookup.global_prediction,
                "leakage_control": (
                    "Exact, containment, and high-similarity variants of all "
                    "POS test sentences were removed before indexing. BM25 "
                    "indexed only headwords and examples; POS labels and "
                    "definitions were payload fields and did not affect "
                    "ranking. Same-headword examples were searched first, "
                    "followed by global backoff."
                ),
            }
        )
        results.append(item)

    summary = {
        "resource": str(path),
        "resource_rows": len(records),
        "strict_documents": len(documents),
        "filter_stats": filter_stats,
        "retrieval_text_fields": [key_field, example_field],
        "label_payload_field": pos_field,
        "retrieval_scope_counts": scope_counts,
        "positive_retrievals": positive_retrievals,
        "positive_retrieval_rate": (
            positive_retrievals / len(items) if items else 0.0
        ),
        "fallback_count": fallback_count,
        "fallback_label": majority_lookup.global_prediction,
        **score_predictions(results),
    }
    return results, summary


def run_definition_bm25(
    items: list[dict[str, Any]],
    resource: dict[str, Any],
    bm25_config: dict[str, Any],
    base: Path,
    dry_run: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = resolve_path(base, resource["path"])
    records = read_xlsx_records(path, resource.get("sheet_name"))
    key_field = resource["key_field"]
    answer_field = resource["answer_field"]
    require_fields(records, (key_field, answer_field), path)
    test_headwords = {
        clean(item.get("input")) for item in items if clean(item.get("input"))
    }
    documents = build_definition_documents(
        records,
        key_field=key_field,
        answer_field=answer_field,
        excluded_headwords=test_headwords,
    )

    if dry_run:
        return [], {
            "resource": str(path),
            "resource_rows": len(records),
            "strict_documents": len(documents),
            "excluded_test_headwords": len(test_headwords),
        }

    index = BM25Index(
        documents,
        k1=float(bm25_config.get("k1", 1.5)),
        b=float(bm25_config.get("b", 0.75)),
    )
    top_k = int(bm25_config.get("top_k", 3))
    empty_prediction = clean(
        bm25_config.get("empty_prediction", "未检索到相关释义")
    )
    results: list[dict[str, Any]] = []
    hit_count = 0
    for source in items:
        item = dict(source)
        retrieved = retrieval_results(
            index,
            clean(item.get("input")),
            top_k,
        )
        if retrieved:
            hit_count += 1
        item.update(
            {
                "prediction": (
                    retrieved[0]["answer"] if retrieved else empty_prediction
                ),
                "baseline_method": "bm25-top1-strict-headword-disjoint",
                "retrieval_hit": bool(retrieved),
                "retrieval": retrieved,
                "leakage_control": (
                    "All 释义能力 test headwords were removed from the "
                    "dictionary corpus before indexing."
                ),
            }
        )
        results.append(item)

    summary = {
        "resource": str(path),
        "resource_rows": len(records),
        "strict_documents": len(documents),
        "excluded_test_headwords": len(test_headwords),
        "positive_retrievals": hit_count,
        "positive_retrieval_rate": hit_count / len(items) if items else 0.0,
        **score_predictions(results),
    }
    return results, summary


def run_context_bm25(
    items: list[dict[str, Any]],
    resource: dict[str, Any],
    bm25_config: dict[str, Any],
    leakage_filter_config: dict[str, Any],
    base: Path,
    dry_run: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = resolve_path(base, resource["path"])
    records = read_xlsx_records(path, resource.get("sheet_name"))
    key_field = resource["key_field"]
    answer_field = resource["answer_field"]
    example_field = resource["example_field"]
    pos_field = resource["pos_field"]
    require_fields(
        records,
        (key_field, answer_field, example_field, pos_field),
        path,
    )
    test_examples = extract_test_examples(items, "语境理解")
    strict_records, filter_stats = filter_records_by_test_examples(
        records,
        example_field=example_field,
        test_examples=test_examples,
        similarity_threshold=float(
            leakage_filter_config.get("similarity_threshold", 0.90)
        ),
        containment_min_chars=int(
            leakage_filter_config.get("containment_min_chars", 6)
        ),
        ngram_size=int(leakage_filter_config.get("ngram_size", 3)),
    )
    documents = build_context_documents(
        strict_records,
        key_field=key_field,
        answer_field=answer_field,
        example_field=example_field,
        pos_field=pos_field,
        excluded_example_keys=set(),
    )

    if dry_run:
        return [], {
            "resource": str(path),
            "resource_rows": len(records),
            "strict_documents": len(documents),
            "filter_stats": filter_stats,
        }

    index = HeadwordFirstBM25(
        documents,
        k1=float(bm25_config.get("k1", 1.5)),
        b=float(bm25_config.get("b", 0.75)),
    )
    top_k = int(bm25_config.get("top_k", 3))
    empty_prediction = clean(
        bm25_config.get("empty_prediction", "未检索到相关释义")
    )
    results: list[dict[str, Any]] = []
    hit_count = 0
    scope_counts: dict[str, int] = {
        "same-headword": 0,
        "global-backoff": 0,
    }
    for source in items:
        item = dict(source)
        headword = clean(item.get("input"))
        example = extract_example(clean(item.get("instruction")))
        query = f"{headword} {example}"
        retrieval_scope, retrieved = headword_first_retrieval_results(
            index,
            query,
            headword=headword,
            top_k=top_k,
        )
        scope_counts[retrieval_scope] += 1
        if retrieved:
            hit_count += 1
        item.update(
            {
                "prediction": (
                    retrieved[0]["answer"] if retrieved else empty_prediction
                ),
                "baseline_method": (
                    "bm25-top1-nearduplicate-filtered-headword-first"
                ),
                "query_example": example,
                "retrieval_scope": retrieval_scope,
                "retrieval_hit": bool(retrieved),
                "retrieval": retrieved,
                "leakage_control": (
                    "Examples were removed when they exactly matched, "
                    "contained/were contained by, or exceeded the configured "
                    "character-similarity threshold to a test sentence. "
                    "Retrieval searched same-headword contexts first and used "
                    "the global corpus only when none were available."
                ),
            }
        )
        results.append(item)

    summary = {
        "resource": str(path),
        "resource_rows": len(records),
        "strict_documents": len(documents),
        "filter_stats": filter_stats,
        "retrieval_scope_counts": scope_counts,
        "positive_retrievals": hit_count,
        "positive_retrieval_rate": hit_count / len(items) if items else 0.0,
        **score_predictions(results),
    }
    return results, summary


def main() -> int:
    args = parse_args()
    config, config_base = load_config(Path(args.config))
    eval_root = resolve_path(config_base, config["eval_root"])
    output_root = resolve_path(config_base, config["output_root"])
    resources = config["resources"]
    bm25_config = config.get("bm25", {})
    leakage_filter_config = config.get("leakage_filter", {})
    selected = tuple(args.task) if args.task else TASKS

    summaries: dict[str, Any] = {}
    for task in selected:
        source_path = eval_path(eval_root, task)
        items = read_json_list(source_path)
        if task in LOOKUP_TASKS:
            results, summary = run_lookup(
                task, items, resources[task], config_base, args.dry_run
            )
        elif task == "词性识别":
            results, summary = run_pos(
                items,
                resources[task],
                leakage_filter_config,
                config_base,
                args.dry_run,
            )
        elif task == "释义能力":
            results, summary = run_definition_bm25(
                items,
                resources[task],
                bm25_config,
                config_base,
                args.dry_run,
            )
        else:
            results, summary = run_context_bm25(
                items,
                resources[task],
                bm25_config,
                leakage_filter_config,
                config_base,
                args.dry_run,
            )

        summary["eval_path"] = str(source_path)
        summary["method"] = METHOD_NAMES[task]
        summaries[task] = summary
        if not args.dry_run:
            destination = output_path(output_root, task)
            atomic_write_json(destination, results)
            print(f"[完成] {task}: {destination}")
        else:
            print(f"[检查] {task}: {json.dumps(summary, ensure_ascii=False)}")

        if task == "词性识别":
            context_pos_results, context_pos_summary = run_pos_context_bm25(
                items,
                resources[task],
                bm25_config,
                leakage_filter_config,
                config_base,
                args.dry_run,
            )
            context_pos_summary["eval_path"] = str(source_path)
            context_pos_summary["method"] = (
                "bm25-top1-pos-nearduplicate-filtered-headword-first"
            )
            summaries["词性识别（上下文BM25）"] = context_pos_summary
            if not args.dry_run:
                context_pos_destination = pos_context_output_path(output_root)
                atomic_write_json(
                    context_pos_destination,
                    context_pos_results,
                )
                print(
                    "[完成] 词性识别（上下文BM25）: "
                    f"{context_pos_destination}"
                )
            else:
                print(
                    "[检查] 词性识别（上下文BM25）: "
                    f"{json.dumps(context_pos_summary, ensure_ascii=False)}"
                )

    if not args.dry_run:
        summary_path = output_root / "baseline_run_summary.json"
        atomic_write_json(summary_path, summaries)
        print(f"[汇总] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
