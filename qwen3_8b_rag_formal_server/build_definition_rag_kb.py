#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build the read-only Dictionary-B index for definition RAG.

The builder reads the derived Thirteen-Classics dictionary workbook and the
existing context-evaluation JSON only. It removes every record whose example
is an exact, containment, or near-duplicate variant of any test sentence.
It then creates two isolated retrieval views:

* usage_entries_by_headword: D2 example-only ablation (no POS/definition keys)
* definition_entries_by_headword: D3/D4 cross-dictionary knowledge evidence

The Hanyu Da Cidian gold answers are never read by this program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from baseline_common import (
    atomic_write_json,
    build_usage_evidence_documents,
    clean,
    compact_key,
    extract_example,
    filter_records_by_test_examples,
    load_config,
    read_json_list,
    read_xlsx_records,
    require_fields,
    resolve_path,
    to_simplified,
)


SCHEMA_VERSION = "definition-rag-kb-b-v1"
FORBIDDEN_EVIDENCE_KEYS = {
    "所有含义",
    "answer",
    "output",
    "gold",
    "reference_answer",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="baseline_config.json")
    parser.add_argument(
        "--output",
        default=None,
        help="覆盖definition_rag.index_path；只能写入正式实验目录中的派生位置。",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def optional_text(record: dict[str, str], field: str) -> str:
    value = clean(record.get(field))
    return "" if value.lower() in {"", "0", "nan", "none", "null"} else value


def sense_identity(pos: str, definition: str) -> str:
    """Return a script- and punctuation-insensitive exact sense identity."""
    normalized_pos = compact_key(to_simplified(pos))
    normalized_definition = compact_key(to_simplified(definition))
    return f"{normalized_pos}|{normalized_definition}"


def representative_priority(entry: dict[str, Any]) -> tuple[int, int, int, int]:
    """Prefer a record with richer usable evidence; break ties deterministically."""
    return (
        int(bool(entry["book"])) + int(bool(entry["chapter"])),
        len(compact_key(entry["paragraph"])),
        len(compact_key(entry["example"])),
        -int(entry["source_row"]),
    )


def evidence_key_union(entries: Iterable[dict[str, Any]]) -> set[str]:
    return set().union(*(entry.keys() for entry in entries))


def assert_safe_evidence(
    entries: Iterable[dict[str, Any]],
    *,
    allowed_fields: set[str],
) -> None:
    entries_list = list(entries)
    retrieval_evidence = evidence_key_union(entries_list)
    # Required hard guards: ``retrieval_evidence`` is the complete key set.
    assert "所有含义" not in retrieval_evidence
    assert "answer" not in retrieval_evidence
    exposed = retrieval_evidence & FORBIDDEN_EVIDENCE_KEYS
    assert not exposed, f"检索证据包含禁用字段：{sorted(exposed)}"
    unexpected = retrieval_evidence - allowed_fields
    assert not unexpected, f"检索证据包含未声明字段：{sorted(unexpected)}"


def build_definition_entries(
    records: list[dict[str, str]],
    *,
    key_field: str,
    definition_field: str,
    pos_field: str,
    example_field: str,
    paragraph_field: str,
    book_field: str,
    chapter_field: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    rows_with_definition = 0
    duplicate_sense_rows = 0

    for fallback_row, record in enumerate(records, start=2):
        headword = clean(record.get(key_field))
        definition = clean(record.get(definition_field))
        if not headword or not definition:
            continue
        rows_with_definition += 1
        pos = optional_text(record, pos_field)
        entry = {
            "headword": headword,
            "pos": pos,
            "definition": definition,
            "example": optional_text(record, example_field),
            "paragraph": optional_text(record, paragraph_field),
            "book": optional_text(record, book_field),
            "chapter": optional_text(record, chapter_field),
            "source_row": int(record.get("__source_row__", fallback_row)),
        }
        identity = sense_identity(pos, definition)
        existing = grouped[headword].get(identity)
        if existing is None:
            grouped[headword][identity] = entry
        else:
            duplicate_sense_rows += 1
            if representative_priority(entry) > representative_priority(existing):
                grouped[headword][identity] = entry

    entries_by_headword = {
        headword: sorted(
            identities.values(),
            key=lambda item: (
                compact_key(to_simplified(item["pos"])),
                compact_key(to_simplified(item["definition"])),
                item["source_row"],
            ),
        )
        for headword, identities in sorted(grouped.items())
    }
    all_entries = [
        entry
        for entries in entries_by_headword.values()
        for entry in entries
    ]
    assert_safe_evidence(
        all_entries,
        allowed_fields={
            "headword",
            "pos",
            "definition",
            "example",
            "paragraph",
            "book",
            "chapter",
            "source_row",
        },
    )
    stats = {
        "rows_with_definition": rows_with_definition,
        "unique_sense_entries": len(all_entries),
        "headwords": len(entries_by_headword),
        "deduplicated_same_pos_definition_rows": duplicate_sense_rows,
    }
    return entries_by_headword, stats


def build_usage_entries(
    records: list[dict[str, str]],
    *,
    key_field: str,
    example_field: str,
    paragraph_field: str,
    book_field: str,
    chapter_field: str,
    similarity_threshold: float,
    containment_min_chars: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    visible_fields = (
        key_field,
        example_field,
        paragraph_field,
        book_field,
        chapter_field,
    )
    safe_records = [
        {field: record.get(field, "") for field in visible_fields}
        for record in records
    ]
    documents, stats = build_usage_evidence_documents(
        safe_records,
        key_field=key_field,
        example_field=example_field,
        paragraph_field=paragraph_field,
        book_field=book_field,
        chapter_field=chapter_field,
        similarity_threshold=similarity_threshold,
        containment_min_chars=containment_min_chars,
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for document in documents:
        entry = {
            "headword": document.headword,
            "example": document.example,
            "paragraph": document.paragraph,
            "book": document.book,
            "chapter": document.chapter,
            "source_row": document.source_row,
        }
        grouped[document.headword].append(entry)
    entries_by_headword = {
        headword: sorted(entries, key=lambda item: item["source_row"])
        for headword, entries in sorted(grouped.items())
    }
    all_entries = [
        entry
        for entries in entries_by_headword.values()
        for entry in entries
    ]
    assert_safe_evidence(
        all_entries,
        allowed_fields={
            "headword",
            "example",
            "paragraph",
            "book",
            "chapter",
            "source_row",
        },
    )
    return entries_by_headword, stats


def count_test_coverage(
    test_headwords: list[str],
    entries_by_headword: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    simplified_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for headword, entries in entries_by_headword.items():
        simplified_groups[to_simplified(headword)].extend(entries)

    scopes: Counter[str] = Counter()
    candidate_counts: list[int] = []
    for headword in test_headwords:
        candidates = entries_by_headword.get(headword, [])
        scope = "same-headword"
        if not candidates:
            simplified = to_simplified(headword)
            if simplified != headword:
                candidates = simplified_groups.get(simplified, [])
                if candidates:
                    scope = "simplified-headword"
        if not candidates:
            scope = "no-evidence"
        scopes[scope] += 1
        candidate_counts.append(len(candidates))
    sorted_counts = sorted(candidate_counts)
    median = (
        0.0
        if not sorted_counts
        else (
            float(sorted_counts[len(sorted_counts) // 2])
            if len(sorted_counts) % 2
            else (
                sorted_counts[len(sorted_counts) // 2 - 1]
                + sorted_counts[len(sorted_counts) // 2]
            )
            / 2.0
        )
    )
    return {
        "items": len(test_headwords),
        "scope_counts": dict(scopes),
        "covered_items": sum(count > 0 for count in candidate_counts),
        "zero_candidate_items": sum(count == 0 for count in candidate_counts),
        "average_candidates": round(
            sum(candidate_counts) / max(len(candidate_counts), 1),
            3,
        ),
        "median_candidates": median,
        "maximum_candidates": max(candidate_counts, default=0),
        "items_with_at_most_20_candidates": sum(
            count <= 20 for count in candidate_counts
        ),
    }


def main() -> int:
    args = parse_args()
    config, config_base = load_config(Path(args.config))
    resource = config["resources"]["释义能力"]
    definition_config = config["definition_rag"]
    leakage_config = config.get("leakage_filter", {})

    retrieval_source_path = str(
        resolve_path(config_base, resource["path"])
    )
    # Required hard guard against accidentally indexing the gold dictionary.
    assert "汉语大词典" not in retrieval_source_path
    source_path = Path(retrieval_source_path)
    context_test_path = resolve_path(
        config_base,
        definition_config["test_context_path"],
    )
    output_path = resolve_path(
        config_base,
        args.output or definition_config["index_path"],
    )
    try:
        output_path.relative_to(config_base)
    except ValueError as error:
        raise ValueError(
            "释义RAG索引只能写入正式实验目录内部，不能覆盖原始数据目录。"
        ) from error

    records = read_xlsx_records(source_path, resource.get("sheet_name"))
    fields = (
        resource["key_field"],
        resource["definition_field"],
        resource["pos_field"],
        resource["example_field"],
        resource["paragraph_field"],
        resource["book_field"],
        resource["chapter_field"],
    )
    require_fields(records, fields, source_path)
    records_with_rows = [
        {**record, "__source_row__": row_number}
        for row_number, record in enumerate(records, start=2)
    ]

    # Project the context-test file to instruction/input immediately. Gold
    # ``output`` values are deliberately ignored by the builder.
    raw_context_items = read_json_list(context_test_path)
    test_examples: list[str] = []
    test_headwords: list[str] = []
    for item_number, raw_item in enumerate(raw_context_items, start=1):
        safe_item = {
            "instruction": clean(raw_item.get("instruction")),
            "input": clean(raw_item.get("input")),
        }
        example = extract_example(safe_item["instruction"])
        if not safe_item["input"] or not example:
            raise ValueError(
                f"语境测试集第{item_number}条缺少字头或无法提取例句。"
            )
        test_headwords.append(safe_item["input"])
        test_examples.append(example)

    similarity_threshold = float(
        leakage_config.get("similarity_threshold", 0.90)
    )
    containment_min_chars = int(
        leakage_config.get("containment_min_chars", 6)
    )
    strict_records, filter_stats = filter_records_by_test_examples(
        records_with_rows,
        example_field=resource["example_field"],
        test_examples=test_examples,
        similarity_threshold=similarity_threshold,
        containment_min_chars=containment_min_chars,
        ngram_size=int(leakage_config.get("ngram_size", 3)),
    )

    usage_entries, usage_stats = build_usage_entries(
        strict_records,
        key_field=resource["key_field"],
        example_field=resource["example_field"],
        paragraph_field=resource["paragraph_field"],
        book_field=resource["book_field"],
        chapter_field=resource["chapter_field"],
        similarity_threshold=similarity_threshold,
        containment_min_chars=containment_min_chars,
    )
    definition_entries, definition_stats = build_definition_entries(
        strict_records,
        key_field=resource["key_field"],
        definition_field=resource["definition_field"],
        pos_field=resource["pos_field"],
        example_field=resource["example_field"],
        paragraph_field=resource["paragraph_field"],
        book_field=resource["book_field"],
        chapter_field=resource["chapter_field"],
    )

    source_dictionary = clean(definition_config["source_dictionary"])
    reference_dictionary = clean(definition_config["reference_dictionary"])
    if source_dictionary == reference_dictionary:
        raise ValueError("检索辞书与金标准辞书必须相互独立。")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "index_type": "cross-dictionary-definition-rag",
        "provenance": {
            "source_dictionary": source_dictionary,
            "reference_dictionary": reference_dictionary,
            "retrieval_source_path": str(source_path),
            "retrieval_source_sha256": sha256_file(source_path),
            "test_context_path": str(context_test_path),
            "test_context_sha256": sha256_file(context_test_path),
            "gold_dictionary_read_by_builder": False,
        },
        "leakage_filter": filter_stats,
        "stats": {
            "usage_view": usage_stats,
            "definition_view": definition_stats,
            "usage_test_coverage": count_test_coverage(
                test_headwords,
                usage_entries,
            ),
            "definition_test_coverage": count_test_coverage(
                test_headwords,
                definition_entries,
            ),
        },
        "usage_entries_by_headword": usage_entries,
        "definition_entries_by_headword": definition_entries,
    }
    atomic_write_json(output_path, payload)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "schema_version": SCHEMA_VERSION,
                "filter_stats": filter_stats,
                "usage_stats": usage_stats,
                "definition_stats": definition_stats,
                "definition_coverage": payload["stats"][
                    "definition_test_coverage"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
