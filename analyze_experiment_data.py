#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Audit the six-task evaluation data and candidate baseline resources.

The script is read-only. It checks:
  * JSON/Excel consistency for the 200-item evaluation sets;
  * 0-shot/5-shot target identity;
  * exact lookup coverage and answer identity;
  * leakage caused by using full dictionaries as retrieval corpora;
  * leave-test-sentence-out coverage for POS/context resources.

Output is a JSON report suitable for archiving with the experiment code.
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from baseline_common import (
    filter_records_by_test_examples,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="论文补充实验目录。")
    parser.add_argument(
        "--output",
        default="experiment_data_audit.json",
        help="审计报告JSON路径（相对于--root）。",
    )
    parser.add_argument(
        "--config",
        default="baseline_config.json",
        help="读取与正式实验一致的近重复过滤参数。",
    )
    return parser.parse_args()


def clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return unicodedata.normalize("NFKC", str(value)).strip()


def compact_key(value: Any) -> str:
    """Punctuation/whitespace-insensitive key for sentence matching."""
    return "".join(char for char in clean(value) if char.isalnum())


def read_json_list(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError(f"JSON顶层必须是对象列表：{path}")
    return data


def read_excel(path: Path, sheet_name: str | int = 0) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name=sheet_name, dtype=str)
    frame.columns = [clean(column) for column in frame.columns]
    return frame.fillna("")


def majority_lookup(
    rows: Iterable[tuple[str, str]],
) -> tuple[dict[str, str], str]:
    per_key: dict[str, Counter[str]] = defaultdict(Counter)
    global_counts: Counter[str] = Counter()
    for raw_key, raw_value in rows:
        key = clean(raw_key)
        value = clean(raw_value)
        if key and value:
            per_key[key][value] += 1
            global_counts[value] += 1

    def choose(counter: Counter[str]) -> str:
        maximum = max(counter.values())
        return sorted(label for label, count in counter.items() if count == maximum)[0]

    lookup = {key: choose(counts) for key, counts in per_key.items()}
    fallback = choose(global_counts) if global_counts else ""
    return lookup, fallback


def lookup_metrics(
    eval_items: list[dict[str, Any]],
    lookup: dict[str, str],
    fallback: str = "",
) -> dict[str, Any]:
    hits = 0
    correct = 0
    fallback_correct = 0
    for item in eval_items:
        key = clean(item.get("input"))
        gold = simplified_answer_key(item.get("output"))
        if key in lookup:
            hits += 1
            prediction = lookup[key]
        else:
            prediction = fallback
            if simplified_answer_key(prediction) == gold:
                fallback_correct += 1
        if simplified_answer_key(prediction) == gold:
            correct += 1
    total = len(eval_items)
    return {
        "total": total,
        "lookup_hits": hits,
        "lookup_coverage": hits / total if total else 0.0,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "fallback_correct": fallback_correct,
        "fallback_label": fallback,
        "accuracy_normalization": (
            "Unicode NFKC + OpenCC t2s + whitespace removal"
        ),
    }


def excel_json_consistency(
    frame: pd.DataFrame,
    eval_items: list[dict[str, Any]],
    input_column: str,
    output_column: str,
) -> dict[str, Any]:
    count = min(len(frame), len(eval_items))
    input_matches = 0
    output_matches = 0
    for index in range(count):
        item = eval_items[index]
        row = frame.iloc[index]
        input_matches += clean(row[input_column]) == clean(item.get("input"))
        output_matches += (
            surface_answer_key(row[output_column])
            == surface_answer_key(item.get("output"))
        )
    return {
        "excel_rows": len(frame),
        "json_rows": len(eval_items),
        "aligned_rows_checked": count,
        "input_matches": input_matches,
        "output_matches": output_matches,
        "fully_consistent": (
            len(frame) == len(eval_items)
            and input_matches == count
            and output_matches == count
        ),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    output_path = root / args.output
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    leakage_filter_config = config.get("leakage_filter", {})

    eval_sets: dict[str, dict[str, list[dict[str, Any]]]] = {}
    task_summary: dict[str, Any] = {}
    for task in TASKS:
        modes: dict[str, list[dict[str, Any]]] = {}
        for mode in ("0-shot", "5-shot"):
            path = root / "eval_data" / "json" / task / mode / f"{task}_eval.json"
            modes[mode] = read_json_list(path)
        eval_sets[task] = modes
        zero = modes["0-shot"]
        five = modes["5-shot"]
        task_summary[task] = {
            "zero_shot_count": len(zero),
            "five_shot_count": len(five),
            "unique_inputs": len({clean(item.get("input")) for item in zero}),
            "duplicate_input_count": len(zero)
            - len({clean(item.get("input")) for item in zero}),
            "zero_five_inputs_identical": [
                clean(item.get("input")) for item in zero
            ]
            == [clean(item.get("input")) for item in five],
            "zero_five_outputs_identical": [
                surface_answer_key(item.get("output")) for item in zero
            ]
            == [surface_answer_key(item.get("output")) for item in five],
        }

    radical_eval = read_excel(root / "eval_data" / "excel" / "部首_eval.xlsx")
    pronunciation_eval = read_excel(
        root / "eval_data" / "excel" / "pinyin&fanqie_eval_labeled.xlsx"
    )
    pos_eval = read_excel(root / "eval_data" / "excel" / "词性_eval.xlsx")
    semantic_eval = read_excel(root / "eval_data" / "excel" / "释义&语境_eval.xlsx")

    consistency = {
        "部首识别": excel_json_consistency(
            radical_eval,
            eval_sets["部首识别"]["0-shot"],
            "字词",
            "部首",
        ),
        "拼音注音": excel_json_consistency(
            pronunciation_eval,
            eval_sets["拼音注音"]["0-shot"],
            "字词",
            "拼音",
        ),
        "反切注音": excel_json_consistency(
            pronunciation_eval,
            eval_sets["反切注音"]["0-shot"],
            "字词",
            "反切音",
        ),
        "词性识别": excel_json_consistency(
            pos_eval,
            eval_sets["词性识别"]["0-shot"],
            "字词",
            "词性",
        ),
        "释义能力": excel_json_consistency(
            semantic_eval,
            eval_sets["释义能力"]["0-shot"],
            "字词",
            "所有含义",
        ),
        "语境理解": excel_json_consistency(
            semantic_eval,
            eval_sets["语境理解"]["0-shot"],
            "字词",
            "释义",
        ),
    }

    radical_resource = read_excel(root / "合并数据" / "merged_部首.xlsx")
    pronunciation_resource = read_excel(
        root / "合并数据" / "merge_拼音&反切.xlsx"
    )
    full_definition_resource = read_excel(
        root / "合并数据" / "all_字词及释义.xlsx"
    )
    context_resource = read_excel(
        root / "合并数据" / "merged_词性&例句&释义.xlsx"
    )

    radical_lookup, radical_fallback = majority_lookup(
        zip(radical_resource["字词"], radical_resource["部首"])
    )
    pinyin_lookup, pinyin_fallback = majority_lookup(
        zip(pronunciation_resource["字词"], pronunciation_resource["拼音"])
    )
    fanqie_lookup, fanqie_fallback = majority_lookup(
        zip(pronunciation_resource["字词"], pronunciation_resource["反切音"])
    )
    definition_lookup, _ = majority_lookup(
        zip(full_definition_resource["字词"], full_definition_resource["释义"])
    )

    context_resource = context_resource.copy()
    context_records = context_resource.to_dict(orient="records")
    common_filter_arguments = {
        "example_field": "例句_校准",
        "similarity_threshold": float(
            leakage_filter_config.get("similarity_threshold", 0.90)
        ),
        "containment_min_chars": int(
            leakage_filter_config.get("containment_min_chars", 6)
        ),
        "ngram_size": int(leakage_filter_config.get("ngram_size", 3)),
    }
    pos_train_records, pos_filter_stats = filter_records_by_test_examples(
        context_records,
        test_examples=pos_eval["例句"].tolist(),
        **common_filter_arguments,
    )
    context_train_records, context_filter_stats = filter_records_by_test_examples(
        context_records,
        test_examples=semantic_eval["例句"].tolist(),
        **common_filter_arguments,
    )
    pos_train = pd.DataFrame(pos_train_records)
    context_train = pd.DataFrame(context_train_records)
    pos_lookup, pos_fallback = majority_lookup(
        zip(pos_train["字词"], pos_train["词性"])
    )
    context_headword_lookup, _ = majority_lookup(
        zip(context_train["字词"], context_train["词性"])
    )

    definition_test_words = {
        clean(item.get("input"))
        for item in eval_sets["释义能力"]["0-shot"]
        if clean(item.get("input"))
    }
    definition_strict_train = full_definition_resource[
        ~full_definition_resource["字词"].map(clean).isin(definition_test_words)
    ]

    context_test_pairs = {
        (clean(row["字词"]), compact_key(row["例句"]))
        for _, row in semantic_eval.iterrows()
    }
    exact_context_rows = 0
    exact_context_answer_rows = 0
    context_index: dict[tuple[str, str], list[str]] = defaultdict(list)
    for _, row in context_resource.iterrows():
        key = (clean(row["字词"]), compact_key(row["例句_校准"]))
        if key[0] and key[1]:
            context_index[key].append(clean(row["释义"]))
    for _, row in semantic_eval.iterrows():
        key = (clean(row["字词"]), compact_key(row["例句"]))
        candidates = context_index.get(key, [])
        if candidates:
            exact_context_rows += 1
        if any(
            simplified_answer_key(candidate)
            == simplified_answer_key(row["释义"])
            for candidate in candidates
        ):
            exact_context_answer_rows += 1

    contamination_groups = (
        pronunciation_eval["contamination_group"]
        .map(clean)
        .value_counts()
        .sort_index()
        .to_dict()
    )
    source_books = (
        semantic_eval["书名"].map(clean).value_counts().sort_index().to_dict()
    )

    report = {
        "root": str(root),
        "task_summary": task_summary,
        "excel_json_consistency": consistency,
        "eval_distributions": {
            "pinyin_fanqie_contamination_group": contamination_groups,
            "semantic_source_book": source_books,
        },
        "resource_rows": {
            "merged_部首": len(radical_resource),
            "merge_拼音&反切": len(pronunciation_resource),
            "all_字词及释义": len(full_definition_resource),
            "merged_词性&例句&释义": len(context_resource),
        },
        "openbook_lookup": {
            "部首识别": lookup_metrics(
                eval_sets["部首识别"]["0-shot"],
                radical_lookup,
                radical_fallback,
            ),
            "拼音注音": lookup_metrics(
                eval_sets["拼音注音"]["0-shot"],
                pinyin_lookup,
                pinyin_fallback,
            ),
            "反切注音": lookup_metrics(
                eval_sets["反切注音"]["0-shot"],
                fanqie_lookup,
                fanqie_fallback,
            ),
            "释义能力_exact_dictionary": lookup_metrics(
                eval_sets["释义能力"]["0-shot"],
                definition_lookup,
            ),
        },
        "leakage_audit": {
            "definition_test_headwords_in_full_dictionary": sum(
                word in definition_lookup for word in definition_test_words
            ),
            "definition_test_headword_count": len(definition_test_words),
            "definition_full_dictionary_rows": len(full_definition_resource),
            "definition_strict_corpus_rows_after_test_headword_exclusion": len(
                definition_strict_train
            ),
            "context_exact_test_pairs_found_in_full_resource": exact_context_rows,
            "context_exact_test_pairs_with_identical_gold": exact_context_answer_rows,
            "context_test_pair_count": len(context_test_pairs),
            "context_full_resource_rows": len(context_resource),
            "context_filter_stats": context_filter_stats,
            "pos_filter_stats": pos_filter_stats,
        },
        "leave_test_sentence_out_pos_baseline": lookup_metrics(
            eval_sets["词性识别"]["0-shot"],
            pos_lookup,
            pos_fallback,
        ),
        "leave_test_sentence_out_headword_coverage": {
            "语境理解": sum(
                clean(item.get("input")) in context_headword_lookup
                for item in eval_sets["语境理解"]["0-shot"]
            ),
            "total": len(eval_sets["语境理解"]["0-shot"]),
        },
        "recommended_protocol": {
            "closed_retrieval_definition": (
                "BM25 corpus = all_字词及释义 minus every 释义能力 test headword."
            ),
            "closed_retrieval_context": (
                "BM25 corpus removes exact, containment, and >=0.90 "
                "character-similarity variants of every test sentence, then "
                "retrieves same-headword contexts before global backoff."
            ),
            "pos_prior": (
                "Headword-majority POS after the same near-duplicate filtering; "
                "global-majority fallback for unseen headwords."
            ),
            "pos_context_bm25": (
                "After the same near-duplicate filtering, BM25 indexes only "
                "headwords and examples, retrieves same-headword examples "
                "first, and copies the Top-1 candidate POS label. POS labels "
                "and definitions are not indexed."
            ),
            "openbook": (
                "Exact lookup in full dictionaries must be labeled as an "
                "open-book upper bound, because some resources contain gold answers."
            ),
        },
    }

    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
