#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run Qwen3-8B cross-dictionary RAG experiments.

The retrievers use task-appropriate leakage controls:
  * 释义能力 reads a prebuilt Dictionary-B index and supports D2 example-only,
    D3 definition/POS, and D4 full-evidence prompts. Retrieval is exact
    same-headword with a one-way Traditional-to-Simplified fallback; unrelated
    headwords are never retrieved.
  * 语境理解 removes exact, containment, and high-similarity variants of
    every test sentence, then retrieves same-headword contexts first.

For 释义能力 the gold ``output`` is projected out immediately and is not
written to prediction files. ``evaluate_definition_rag.py`` reads it only
after generation has finished. Results are atomically checkpointed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from baseline_common import (
    HeadwordFirstBM25,
    atomic_write_json,
    build_context_documents,
    clean,
    compact_key,
    extract_example,
    filter_records_by_test_examples,
    headword_first_retrieval_results,
    load_config,
    prediction_is_complete,
    read_json_list,
    read_xlsx_records,
    require_fields,
    resolve_path,
    to_simplified,
    validate_resume,
)


TASKS = ("释义能力", "语境理解")
DEFINITION_MODES = ("D2", "D3", "D4")
DEFINITION_SCHEMA_VERSION = "definition-rag-kb-b-v1"
FORBIDDEN_DEFINITION_KEYS = {
    "所有含义",
    "answer",
    "output",
    "gold",
    "reference_answer",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _character_ngrams(text: str, size: int) -> set[str]:
    normalized = compact_key(text)
    if len(normalized) < size:
        return {normalized} if normalized else set()
    return {
        normalized[index : index + size]
        for index in range(len(normalized) - size + 1)
    }


def _ngram_jaccard(left: str, right: str, size: int) -> float:
    left_ngrams = _character_ngrams(left, size)
    right_ngrams = _character_ngrams(right, size)
    if not left_ngrams and not right_ngrams:
        return 1.0
    union = left_ngrams | right_ngrams
    return len(left_ngrams & right_ngrams) / len(union) if union else 0.0


class DefinitionRAGIndex:
    """Strict exact-headword index over the independently built Dictionary B."""

    def __init__(
        self,
        payload: dict[str, Any],
        *,
        index_path: Path,
        diversity_ngram_size: int,
    ) -> None:
        if payload.get("schema_version") != DEFINITION_SCHEMA_VERSION:
            raise ValueError(
                f"释义索引版本不匹配：{index_path}；"
                f"需要{DEFINITION_SCHEMA_VERSION}。"
            )
        if diversity_ngram_size < 1:
            raise ValueError("diversity_ngram_size必须大于0。")
        provenance = payload.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError("释义索引缺少provenance。")
        retrieval_source_path = clean(
            provenance.get("retrieval_source_path")
        )
        assert "汉语大词典" not in retrieval_source_path
        if provenance.get("gold_dictionary_read_by_builder") is not False:
            raise ValueError("释义索引未声明gold_dictionary_read_by_builder=false。")
        if provenance.get("source_dictionary") == provenance.get(
            "reference_dictionary"
        ):
            raise ValueError("检索辞书和参考辞书不能相同。")

        usage = payload.get("usage_entries_by_headword")
        definitions = payload.get("definition_entries_by_headword")
        if not isinstance(usage, dict) or not isinstance(definitions, dict):
            raise ValueError("释义索引缺少D2或D3/D4检索视图。")
        self.index_path = index_path
        self.index_sha256 = sha256_file(index_path)
        self.provenance = provenance
        self.stats = payload.get("stats", {})
        self.diversity_ngram_size = diversity_ngram_size
        self.usage_by_headword = self._validate_view(
            usage,
            allowed_fields={
                "headword",
                "example",
                "paragraph",
                "book",
                "chapter",
                "source_row",
            },
            view_name="D2",
        )
        self.definition_by_headword = self._validate_view(
            definitions,
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
            view_name="D3/D4",
        )
        self.usage_by_simplified = self._simplified_view(
            self.usage_by_headword
        )
        self.definition_by_simplified = self._simplified_view(
            self.definition_by_headword
        )

    @staticmethod
    def _validate_view(
        raw_view: dict[str, Any],
        *,
        allowed_fields: set[str],
        view_name: str,
    ) -> dict[str, list[dict[str, Any]]]:
        validated: dict[str, list[dict[str, Any]]] = {}
        for raw_headword, raw_entries in raw_view.items():
            headword = clean(raw_headword)
            if not headword or not isinstance(raw_entries, list):
                raise ValueError(f"{view_name}索引包含非法字头或候选列表。")
            entries: list[dict[str, Any]] = []
            for raw_entry in raw_entries:
                if not isinstance(raw_entry, dict):
                    raise ValueError(f"{view_name}索引候选不是JSON对象。")
                retrieval_evidence = set(raw_entry)
                # Required hard guards over the complete candidate key set.
                assert "所有含义" not in retrieval_evidence
                assert "answer" not in retrieval_evidence
                exposed = retrieval_evidence & FORBIDDEN_DEFINITION_KEYS
                assert not exposed, f"{view_name}暴露禁用字段：{sorted(exposed)}"
                unexpected = retrieval_evidence - allowed_fields
                if unexpected:
                    raise ValueError(
                        f"{view_name}索引包含未声明字段：{sorted(unexpected)}"
                    )
                if clean(raw_entry.get("headword")) != headword:
                    raise ValueError(f"{view_name}索引字头与候选内容不一致。")
                entries.append(dict(raw_entry))
            validated[headword] = entries
        return validated

    @staticmethod
    def _simplified_view(
        view: dict[str, list[dict[str, Any]]],
    ) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        seen: dict[str, set[tuple[str, int]]] = defaultdict(set)
        for headword, entries in view.items():
            simplified = to_simplified(headword)
            for entry in entries:
                identity = (
                    clean(entry.get("headword")),
                    int(entry.get("source_row", 0)),
                )
                if identity not in seen[simplified]:
                    grouped[simplified].append(entry)
                    seen[simplified].add(identity)
        return dict(grouped)

    def _select_diverse(
        self,
        candidates: list[dict[str, Any]],
        *,
        mode: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        selected: list[tuple[float, dict[str, Any]]] = []
        remaining = list(candidates)
        selected_books: set[str] = set()
        selected_sources: set[tuple[str, str]] = set()
        content_field = "example" if mode == "D2" else "definition"

        while remaining and len(selected) < top_k:
            scored: list[tuple[float, int, dict[str, Any]]] = []
            selected_texts = [
                clean(entry.get(content_field))
                for _, entry in selected
            ]
            for candidate in remaining:
                content = clean(candidate.get(content_field))
                if selected_texts:
                    novelty = 1.0 - max(
                        _ngram_jaccard(
                            content,
                            selected_text,
                            self.diversity_ngram_size,
                        )
                        for selected_text in selected_texts
                    )
                else:
                    novelty = 1.0
                book = clean(candidate.get("book"))
                chapter = clean(candidate.get("chapter"))
                unseen_book = float(bool(book) and book not in selected_books)
                source = (book, chapter)
                unseen_source = float(
                    bool(book or chapter) and source not in selected_sources
                )
                informativeness = min(len(compact_key(content)), 80) / 80.0
                score = (
                    2.0 * unseen_book
                    + unseen_source
                    + 1.5 * novelty
                    + 0.25 * informativeness
                )
                scored.append(
                    (
                        score,
                        -int(candidate.get("source_row", 0)),
                        candidate,
                    )
                )
            score, _, chosen = max(
                scored,
                key=lambda item: (item[0], item[1]),
            )
            selected.append((score, chosen))
            book = clean(chosen.get("book"))
            chapter = clean(chosen.get("chapter"))
            if book:
                selected_books.add(book)
            if book or chapter:
                selected_sources.add((book, chapter))
            remaining.remove(chosen)

        results: list[dict[str, Any]] = []
        for rank, (score, entry) in enumerate(selected, start=1):
            result = dict(entry)
            result["rank"] = rank
            result["score"] = round(score, 6)
            results.append(result)
        return results

    def search(
        self,
        headword: str,
        *,
        mode: str,
        top_k: int,
    ) -> tuple[str, list[dict[str, Any]]]:
        if mode not in DEFINITION_MODES:
            raise ValueError(f"未知释义实验模式：{mode}")
        view = (
            self.usage_by_headword
            if mode == "D2"
            else self.definition_by_headword
        )
        simplified_view = (
            self.usage_by_simplified
            if mode == "D2"
            else self.definition_by_simplified
        )
        normalized = clean(headword)
        candidates = list(view.get(normalized, []))
        scope = f"same-headword-{mode.lower()}"
        if not candidates:
            simplified = to_simplified(normalized)
            # One-way fallback only when the query itself changes under T2S.
            if simplified != normalized:
                candidates = list(simplified_view.get(simplified, []))
                if candidates:
                    scope = f"simplified-headword-{mode.lower()}"
        if not candidates:
            return "no-evidence", []
        return scope, self._select_diverse(
            candidates,
            mode=mode,
            top_k=top_k,
        )


def load_definition_index(
    path: Path,
    *,
    diversity_ngram_size: int,
) -> DefinitionRAGIndex:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"释义索引顶层必须为JSON对象：{path}")
    return DefinitionRAGIndex(
        payload,
        index_path=path,
        diversity_ngram_size=diversity_ngram_size,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="baseline_config.json")
    parser.add_argument(
        "--task",
        action="append",
        choices=TASKS,
        help="只运行指定任务；可重复。默认运行两个任务。",
    )
    parser.add_argument(
        "--definition-mode",
        choices=DEFINITION_MODES,
        default="D4",
        help=(
            "释义能力证据模式：D2=例句消融，D3=词性+释义，"
            "D4=词性+释义+例句+出处；默认D4。"
        ),
    )
    parser.add_argument(
        "--definition-index",
        default=None,
        help="覆盖definition_rag.index_path。运行前须先构建该派生索引。",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="仅local后端使用：覆盖Qwen3-8B本地路径或HF模型ID。",
    )
    parser.add_argument(
        "--backend",
        choices=("local", "bailian"),
        default=None,
        help="推理后端；覆盖配置中的rag.backend。",
    )
    parser.add_argument(
        "--bailian-model",
        default=None,
        help="仅bailian后端使用：覆盖百炼模型ID，默认qwen3-8b。",
    )
    parser.add_argument(
        "--bailian-base-url",
        default=None,
        help=(
            "仅bailian后端使用：覆盖OpenAI兼容Base URL。"
            "命令行优先，其次DASHSCOPE_BASE_URL，最后读取配置文件。"
        ),
    )
    parser.add_argument(
        "--output-model-name",
        default=None,
        help="覆盖结果文件名中使用的模型名称。",
    )
    parser.add_argument(
        "--max-new-items-per-task",
        type=int,
        default=None,
        help="每项任务本次最多新生成多少条；用于先做1条烟雾测试。",
    )
    parser.add_argument(
        "--load-in-4bit",
        choices=("true", "false"),
        default=None,
        help="覆盖配置文件中的4-bit量化开关。",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="transformers device_map，默认auto。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只构建严格检索库并显示候选，不加载Qwen、不写结果。",
    )
    return parser.parse_args()


def eval_path(eval_root: Path, task: str) -> Path:
    return eval_root / task / "0-shot" / f"{task}_eval.json"


def load_generation_items(path: Path, task: str) -> list[dict[str, Any]]:
    raw_items = read_json_list(path)
    if task != "释义能力":
        return raw_items
    projected: list[dict[str, Any]] = []
    for item_number, raw_item in enumerate(raw_items, start=1):
        item = {
            "instruction": clean(raw_item.get("instruction")),
            "input": clean(raw_item.get("input")),
        }
        if not item["instruction"] or not item["input"]:
            raise ValueError(f"释义测试集第{item_number}条缺少instruction/input。")
        assert "output" not in item
        assert "所有含义" not in item
        projected.append(item)
    return projected


def output_path(
    output_root: Path,
    output_model_name: str,
    task: str,
    *,
    definition_mode: str,
) -> Path:
    method_suffix = f"-{definition_mode.lower()}" if task == "释义能力" else ""
    return (
        output_root
        / task
        / "0-shot"
        / f"{output_model_name}{method_suffix}#{task}#0-shot.json"
    )


def build_retriever(
    task: str,
    items: list[dict[str, Any]],
    resource: dict[str, Any],
    bm25_config: dict[str, Any],
    leakage_filter_config: dict[str, Any],
    config_base: Path,
    *,
    definition_index_path: Path,
    definition_mode: str,
) -> tuple[DefinitionRAGIndex | HeadwordFirstBM25, dict[str, Any]]:

    if task == "释义能力":
        if not definition_index_path.exists():
            raise FileNotFoundError(
                f"释义RAG索引不存在：{definition_index_path}；"
                "请先运行build_definition_rag_kb.py。"
            )
        index = load_definition_index(
            definition_index_path,
            diversity_ngram_size=int(
                leakage_filter_config.get("diversity_ngram_size", 2)
            ),
        )
        visible_fields_by_mode = {
            "D2": ["字词", "例句_校准", "包含例句的段落", "书名", "章节"],
            "D3": ["字词", "词性", "释义", "书名", "章节"],
            "D4": [
                "字词",
                "词性",
                "释义",
                "例句_校准",
                "包含例句的段落",
                "书名",
                "章节",
            ],
        }
        stats = {
            "resource": str(definition_index_path),
            "resource_sha256": index.index_sha256,
            "source_dictionary": index.provenance["source_dictionary"],
            "reference_dictionary": index.provenance["reference_dictionary"],
            "index_stats": index.stats,
            "definition_mode": definition_mode,
            "visible_fields": visible_fields_by_mode[definition_mode],
            "gold_visible_during_generation": False,
            "leakage_control": (
                "test-example-exclusion-exact-headword-t2s-only-no-global"
            ),
        }
    else:
        path = resolve_path(config_base, resource["path"])
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
        extracted_examples: list[str] = []
        for index, item in enumerate(items, start=1):
            example = extract_example(clean(item.get("instruction")))
            if not example:
                raise ValueError(f"语境理解第{index}条无法提取目标例句。")
            extracted_examples.append(example)
        strict_records, filter_stats = filter_records_by_test_examples(
            records,
            example_field=example_field,
            test_examples=extracted_examples,
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
        stats = {
            "resource": str(path),
            "resource_rows": len(records),
            "strict_documents": len(documents),
            "filter_stats": filter_stats,
            "leakage_control": (
                "near-duplicate-filtered-headword-first"
            ),
        }

    if task == "语境理解":
        index = HeadwordFirstBM25(
            documents,
            k1=float(bm25_config.get("k1", 1.5)),
            b=float(bm25_config.get("b", 0.75)),
        )
    return index, stats


def retrieve_for_item(
    task: str,
    item: dict[str, Any],
    index: DefinitionRAGIndex | HeadwordFirstBM25,
    top_k: int,
    *,
    definition_mode: str,
) -> tuple[str, list[dict[str, Any]], str]:
    headword = clean(item.get("input"))
    if task == "释义能力":
        query = headword
        if not isinstance(index, DefinitionRAGIndex):
            raise TypeError("释义能力检索器类型错误。")
        scope, retrieval = index.search(
            headword,
            mode=definition_mode,
            top_k=top_k,
        )
    else:
        example = extract_example(clean(item.get("instruction")))
        query = f"{headword} {example}"
        if not isinstance(index, HeadwordFirstBM25):
            raise TypeError("语境理解检索器类型错误。")
        scope, retrieval = headword_first_retrieval_results(
            index,
            query,
            headword=headword,
            top_k=top_k,
        )
    return query, retrieval, scope


def truncate(value: str, maximum: int) -> str:
    text = clean(value)
    return text if len(text) <= maximum else text[: maximum - 1] + "…"


def build_prompt(
    task: str,
    item: dict[str, Any],
    retrieval: list[dict[str, Any]],
    max_reference_chars: int,
    *,
    definition_mode: str,
    definition_config: dict[str, Any],
) -> str:
    reference_blocks: list[str] = []
    for candidate in retrieval:
        if task == "释义能力":
            source = "，".join(
                value
                for value in (
                    candidate.get("book", ""),
                    candidate.get("chapter", ""),
                )
                if value
            )
            max_definition_chars = int(
                definition_config.get("max_definition_chars", 300)
            )
            max_example_chars = int(
                definition_config.get("max_example_chars", 260)
            )
            max_paragraph_chars = int(
                definition_config.get("max_paragraph_chars", 420)
            )
            if definition_mode == "D2":
                block = (
                    f"[例证{candidate['rank']}]\n"
                    f"出处：{source or '未标注'}\n"
                    f"例句："
                    f"{truncate(candidate.get('example', ''), max_example_chars)}"
                )
                paragraph = candidate.get("paragraph", "")
                if paragraph:
                    block += (
                        "\n上下文："
                        f"{truncate(paragraph, max_paragraph_chars)}"
                    )
            elif definition_mode == "D3":
                block = (
                    f"[辞书义项{candidate['rank']}]\n"
                    f"词性：{candidate.get('pos', '') or '未标注'}\n"
                    f"释义："
                    f"{truncate(candidate.get('definition', ''), max_definition_chars)}\n"
                    f"出处：{source or '未标注'}"
                )
            else:
                block = (
                    f"[辞书证据{candidate['rank']}]\n"
                    f"词性：{candidate.get('pos', '') or '未标注'}\n"
                    f"释义："
                    f"{truncate(candidate.get('definition', ''), max_definition_chars)}\n"
                    f"例句："
                    f"{truncate(candidate.get('example', ''), max_example_chars)}\n"
                    f"出处：{source or '未标注'}"
                )
                paragraph = candidate.get("paragraph", "")
                if paragraph:
                    block += (
                        "\n上下文："
                        f"{truncate(paragraph, max_paragraph_chars)}"
                    )
        else:
            block = (
                f"[候选{candidate['rank']}]\n"
                f"词头：{candidate['headword']}\n"
                f"词性：{candidate.get('pos', '')}\n"
                f"例句：{truncate(candidate.get('example', ''), max_reference_chars)}\n"
                f"语境义：{truncate(candidate['answer'], max_reference_chars)}"
            )
        reference_blocks.append(block)

    if reference_blocks:
        references = "\n\n".join(reference_blocks)
    elif task == "释义能力":
        references = (
            "（独立辞书知识库中没有找到该词头的可用证据，"
            "请依靠可靠的古汉语知识作答。）"
        )
    else:
        references = "（BM25未检索到正分候选，请依靠模型自身知识作答。）"
    headword = clean(item.get("input"))

    if task == "释义能力":
        target = (
            f"目标词：{headword}\n"
            "请编号输出该古汉语词汇的完整释义。只输出最终义项，"
            "不要输出分析过程、检索编号、例句或出处。"
        )
        if definition_mode == "D2":
            control = (
                "这是D2例证增强消融：材料只包含《十三经辞典》的例句、"
                "上下文和出处，不包含词性或释义。请从实际用法归纳义项；"
                "不要把例句直接改写成释义。例证未覆盖的义项可以依据"
                "可靠知识补充，但不确定时不要臆造。"
            )
        elif definition_mode == "D3":
            control = (
                "这是D3跨辞书释义知识RAG：材料来自《十三经辞典》，"
                "不是《汉语大词典》金标准，且可能不完整或重复。"
                "请合并近义或重复记录、区分不同词性和义项，并在可靠时"
                "补充证据未覆盖的常见义项；不要机械拼接或臆造。"
            )
        else:
            control = (
                "这是D4完整跨辞书RAG：材料来自《十三经辞典》，"
                "不是《汉语大词典》金标准，且可能不完整或重复。"
                "请综合词性、释义、例句与出处，合并重复义项并区分"
                "不同用法；可以依据可靠知识补充缺失的常见义项，"
                "但不得机械抄录例句或臆造冷僻义项。"
            )
    else:
        example = extract_example(clean(item.get("instruction")))
        target = (
            f"目标词：{headword}\n"
            f"目标例句：{example}\n"
            "请直接输出目标词在该例句中的准确含义。"
        )
        control = (
            "检索库已经排除了与测试句完全相同、互相包含或高度相似的"
            "例句，并优先检索目标词在其他语境中的记录。"
            "请比较词义和上下文，不要机械复制不相关义项。"
        )

    # Put the target at the end so left-side tokenizer truncation discards
    # older references before it can discard the actual query.
    return (
        "你是一名古汉语词典编纂专家。以下是检索到的参考证据：\n\n"
        f"{references}\n\n"
        f"{control}\n\n"
        f"{target}"
    )


def remove_thinking(text: str) -> str:
    value = clean(text)
    value = re.sub(r"<think>.*?</think>", "", value, flags=re.DOTALL).strip()
    if "</think>" in value:
        value = value.rsplit("</think>", 1)[-1].strip()
    return value


def load_qwen(
    model_path: str,
    *,
    load_in_4bit: bool,
    device_map: str,
):
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
    except ImportError as error:
        raise RuntimeError(
            "缺少Qwen本地推理依赖；请安装torch、transformers、accelerate，"
            "4-bit模式另需bitsandbytes。"
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "device_map": device_map,
        "torch_dtype": "auto",
        "trust_remote_code": False,
    }
    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()
    return torch, tokenizer, model


def generation_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "你是专业的古汉语词典编纂助手。只输出最终答案，不输出思考过程。",
        },
        {"role": "user", "content": prompt},
    ]


def generate_answer_local(
    torch_module,
    tokenizer,
    model,
    prompt: str,
    *,
    max_input_tokens: int,
    max_new_tokens: int,
) -> tuple[str, dict[str, Any]]:
    messages = generation_messages(prompt)
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = tokenizer(
        rendered,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )
    input_device = next(model.parameters()).device
    encoded = {key: value.to(input_device) for key, value in encoded.items()}
    input_length = encoded["input_ids"].shape[1]

    with torch_module.inference_mode():
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    answer_ids = generated[0, input_length:]
    answer = remove_thinking(
        tokenizer.decode(answer_ids, skip_special_tokens=True)
    )
    if not answer:
        raise RuntimeError("Qwen3-8B生成结果为空。")
    return answer, {
        "provider": "local-transformers",
        "finish_reason": "eos_or_max_tokens",
        "reasoning_chars_discarded": 0,
    }


def create_bailian_client(
    *,
    api_key_env: str,
    base_url: str,
    timeout_seconds: float,
):
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError(
            "百炼后端需要OpenAI Python SDK；请运行：pip install openai"
        ) from error
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(
            f"没有读取到{api_key_env}环境变量。"
            f"请先设置百炼API Key，且不要把密钥写入代码。"
        )
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout_seconds,
        max_retries=0,
    )


def _reasoning_piece(value: Any) -> str:
    direct = getattr(value, "reasoning_content", None)
    if direct:
        return str(direct)
    extra = getattr(value, "model_extra", None)
    if isinstance(extra, dict) and extra.get("reasoning_content"):
        return str(extra["reasoning_content"])
    return ""


def _usage_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else None
    fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    result = {
        field: getattr(value, field)
        for field in fields
        if getattr(value, field, None) is not None
    }
    return result or None


def generate_answer_bailian(
    client,
    prompt: str,
    *,
    model_id: str,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
    stream: bool,
    max_attempts: int,
    retry_wait_seconds: float,
) -> tuple[str, dict[str, Any]]:
    """Call Bailian through its OpenAI-compatible Chat Completions endpoint."""
    if max_attempts < 1:
        raise ValueError("bailian.max_attempts必须大于0。")
    messages = generation_messages(prompt)
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                extra_body={"enable_thinking": enable_thinking},
            )
            content_parts: list[str] = []
            reasoning_chars = 0
            finish_reason = ""
            request_id = ""
            response_model = ""
            system_fingerprint = ""
            usage: dict[str, Any] | None = None

            if stream:
                for chunk in response:
                    request_id = clean(getattr(chunk, "id", request_id))
                    response_model = clean(
                        getattr(chunk, "model", response_model)
                    )
                    system_fingerprint = clean(
                        getattr(
                            chunk,
                            "system_fingerprint",
                            system_fingerprint,
                        )
                    )
                    chunk_usage = _usage_dict(getattr(chunk, "usage", None))
                    if chunk_usage:
                        usage = chunk_usage
                    choices = getattr(chunk, "choices", None)
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.delta
                    content_piece = getattr(delta, "content", None)
                    if content_piece:
                        content_parts.append(str(content_piece))
                    reasoning_chars += len(_reasoning_piece(delta))
                    if choice.finish_reason:
                        finish_reason = str(choice.finish_reason)
            else:
                request_id = clean(getattr(response, "id", ""))
                response_model = clean(getattr(response, "model", ""))
                system_fingerprint = clean(
                    getattr(response, "system_fingerprint", "")
                )
                usage = _usage_dict(getattr(response, "usage", None))
                choices = getattr(response, "choices", None)
                if not choices:
                    raise RuntimeError("百炼非流式响应没有choices。")
                choice = choices[0]
                message = choice.message
                content_piece = getattr(message, "content", None)
                if content_piece:
                    content_parts.append(str(content_piece))
                reasoning_chars = len(_reasoning_piece(message))
                finish_reason = clean(choice.finish_reason)

            answer = remove_thinking("".join(content_parts))
            if not answer:
                raise RuntimeError(
                    "百炼返回的最终content为空；"
                    f"finish_reason={finish_reason!r}；"
                    f"reasoning_chars={reasoning_chars}"
                )
            return answer, {
                "provider": "aliyun-bailian-openai-compatible",
                "request_id": request_id,
                "response_model": response_model,
                "system_fingerprint": system_fingerprint,
                "finish_reason": finish_reason,
                "usage": usage,
                "reasoning_chars_discarded": reasoning_chars,
                "attempts": attempt,
                "stream": stream,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as error:
            last_error = error
            if attempt >= max_attempts:
                break
            print(
                f"[百炼重试] 第{attempt}/{max_attempts}次失败："
                f"{type(error).__name__}: {error}",
                file=sys.stderr,
            )
            time.sleep(retry_wait_seconds)

    raise RuntimeError(
        f"百炼API连续{max_attempts}次调用失败：{last_error}"
    ) from last_error


def load_or_initialize(
    source_items: list[dict[str, Any]],
    destination: Path,
) -> list[dict[str, Any]]:
    if destination.exists():
        saved = read_json_list(destination)
        validate_resume(source_items, saved, destination)
        return saved
    result = [dict(item) for item in source_items]
    atomic_write_json(destination, result)
    return result


def task_setting(
    rag_config: dict[str, Any],
    task: str,
    name: str,
    default: int,
) -> int:
    by_task = rag_config.get(f"{name}_by_task", {})
    if isinstance(by_task, dict) and task in by_task:
        return int(by_task[task])
    return int(rag_config.get(name, default))


def rag_method_for_task(task: str, *, definition_mode: str) -> str:
    if task == "释义能力":
        methods = {
            "D2": "qwen3-8b-cross-dictionary-d2-example-only",
            "D3": "qwen3-8b-cross-dictionary-d3-definition-pos",
            "D4": "qwen3-8b-cross-dictionary-d4-full-evidence",
        }
        return methods[definition_mode]
    return "qwen3-8b-bm25-context-strict"


def audit_retriever(
    task: str,
    items: list[dict[str, Any]],
    index: DefinitionRAGIndex | HeadwordFirstBM25,
    *,
    top_k: int,
    definition_mode: str,
) -> dict[str, Any]:
    """Audit coverage and fail if forbidden gold-answer fields are exposed."""

    scope_counts: Counter[str] = Counter()
    candidate_counts: list[int] = []
    for item_number, item in enumerate(items, start=1):
        _, retrieved, scope = retrieve_for_item(
            task,
            item,
            index,
            top_k,
            definition_mode=definition_mode,
        )
        scope_counts[scope] += 1
        candidate_counts.append(len(retrieved))
        if task == "释义能力":
            exposed = [
                sorted(FORBIDDEN_DEFINITION_KEYS & set(candidate))
                for candidate in retrieved
                if FORBIDDEN_DEFINITION_KEYS & set(candidate)
            ]
            if exposed:
                raise RuntimeError(
                    f"释义能力第{item_number}条检索结果暴露gold字段："
                    f"{exposed}"
                )
            if definition_mode == "D2":
                d2_forbidden = {"definition", "pos", "释义", "词性"}
                d2_exposed = [
                    sorted(d2_forbidden & set(candidate))
                    for candidate in retrieved
                    if d2_forbidden & set(candidate)
                ]
                if d2_exposed:
                    raise RuntimeError(
                        f"D2第{item_number}条暴露释义/词性：{d2_exposed}"
                    )

    return {
        "items": len(items),
        "scope_counts": dict(scope_counts),
        "zero_candidate_items": sum(count == 0 for count in candidate_counts),
        "average_candidates": round(
            sum(candidate_counts) / max(len(candidate_counts), 1),
            3,
        ),
        "maximum_candidates": max(candidate_counts, default=0),
        "gold_key_violations": 0 if task == "释义能力" else None,
        "definition_mode": definition_mode if task == "释义能力" else None,
    }


def validate_completed_configuration(
    results: list[dict[str, Any]],
    *,
    destination: Path,
    task: str,
    top_k: int,
    max_reference_chars: int,
    model_path_or_id: str,
    inference_backend: str,
    api_base_url: str | None,
    definition_mode: str,
    definition_index_sha256: str | None,
) -> None:
    """Prevent resume files from mixing incompatible experiment settings."""

    expected_method = rag_method_for_task(
        task,
        definition_mode=definition_mode,
    )
    for index, item in enumerate(results, start=1):
        if not prediction_is_complete(item):
            continue
        actual = {
            "rag_method": item.get("rag_method"),
            "retrieval_top_k": item.get("retrieval_top_k"),
            "max_reference_chars": item.get("max_reference_chars"),
            "model_path_or_id": item.get("model_path_or_id"),
            "definition_mode": item.get("definition_mode"),
            "definition_index_sha256": item.get("definition_index_sha256"),
            "inference_backend": item.get("inference_backend"),
            "api_base_url": item.get("api_base_url"),
        }
        expected = {
            "rag_method": expected_method,
            "retrieval_top_k": top_k,
            "max_reference_chars": max_reference_chars,
            "model_path_or_id": model_path_or_id,
            "definition_mode": (
                definition_mode if task == "释义能力" else None
            ),
            "definition_index_sha256": (
                definition_index_sha256 if task == "释义能力" else None
            ),
            "inference_backend": inference_backend,
            "api_base_url": api_base_url,
        }
        if actual != expected:
            raise ValueError(
                f"断点文件实验配置不一致：{destination}第{index}条；"
                f"已有={actual}，当前={expected}。"
                "请更换--output-model-name，避免混合实验条件。"
            )


def main() -> int:
    args = parse_args()
    config, config_base = load_config(Path(args.config))
    eval_root = resolve_path(config_base, config["eval_root"])
    output_root = resolve_path(config_base, config["output_root"])
    resources = config["resources"]
    bm25_config = config.get("bm25", {})
    leakage_filter_config = config.get("leakage_filter", {})
    definition_config = config.get("definition_rag", {})
    rag_config = config.get("rag", {})
    bailian_config = rag_config.get("bailian", {})
    selected = tuple(args.task) if args.task else TASKS
    definition_mode = args.definition_mode
    definition_index_path = resolve_path(
        config_base,
        args.definition_index or definition_config["index_path"],
    )

    inference_backend = clean(
        args.backend or rag_config.get("backend", "local")
    ).lower()
    if inference_backend not in {"local", "bailian"}:
        raise ValueError("rag.backend必须是local或bailian。")
    model_path = clean(args.model_path or rag_config.get("model_path"))
    bailian_model = clean(
        args.bailian_model
        or bailian_config.get("model")
        or "qwen3-8b"
    )
    bailian_base_url = clean(
        args.bailian_base_url
        or os.getenv("DASHSCOPE_BASE_URL")
        or bailian_config.get("base_url")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ).rstrip("/")
    if bailian_base_url.endswith("/chat/completions"):
        raise ValueError(
            "bailian.base_url应填写到/compatible-mode/v1，"
            "不要包含/chat/completions。"
        )
    model_path_or_id = (
        model_path if inference_backend == "local" else bailian_model
    )
    api_base_url = (
        bailian_base_url if inference_backend == "bailian" else None
    )
    output_model_name = clean(
        args.output_model_name or rag_config.get("output_model_name")
    )
    if not output_model_name or any(
        character in output_model_name for character in ("/", "\\", "\0")
    ):
        raise ValueError("RAG输出模型名为空或含有非法路径字符。")
    load_in_4bit = (
        args.load_in_4bit == "true"
        if args.load_in_4bit is not None
        else bool(rag_config.get("load_in_4bit", False))
    )
    max_input_tokens = int(rag_config.get("max_input_tokens", 8192))
    max_new_tokens = int(rag_config.get("max_new_tokens", 512))
    api_key_env = clean(
        bailian_config.get("api_key_env") or "DASHSCOPE_API_KEY"
    )
    bailian_timeout_seconds = float(
        bailian_config.get("timeout_seconds", 600)
    )
    bailian_max_attempts = int(
        bailian_config.get("max_attempts", 5)
    )
    bailian_retry_wait_seconds = float(
        bailian_config.get("retry_wait_seconds", 10)
    )
    bailian_request_interval_seconds = float(
        bailian_config.get("request_interval_seconds", 1)
    )
    bailian_stream = bool(bailian_config.get("stream", True))
    bailian_enable_thinking = bool(
        bailian_config.get("enable_thinking", False)
    )
    bailian_temperature = float(
        bailian_config.get("temperature", 0)
    )
    if inference_backend == "bailian" and bailian_enable_thinking:
        raise ValueError(
            "正式对照实验要求bailian.enable_thinking=false。"
        )
    top_k_by_task = {
        task: (
            int(definition_config.get("max_unique_senses", 20))
            if task == "释义能力"
            else task_setting(
                rag_config,
                task,
                "top_k",
                int(bm25_config.get("top_k", 3)),
            )
        )
        for task in selected
    }
    max_reference_chars_by_task = {
        task: task_setting(
            rag_config,
            task,
            "max_reference_chars",
            1200,
        )
        for task in selected
    }

    prepared: dict[
        str,
        tuple[
            list[dict[str, Any]],
            DefinitionRAGIndex | HeadwordFirstBM25,
            dict[str, Any],
        ],
    ] = {}
    for task in selected:
        source_items = load_generation_items(eval_path(eval_root, task), task)
        index, stats = build_retriever(
            task,
            source_items,
            resources[task],
            bm25_config,
            leakage_filter_config,
            config_base,
            definition_index_path=definition_index_path,
            definition_mode=definition_mode,
        )
        prepared[task] = (source_items, index, stats)
        audit = audit_retriever(
            task,
            source_items,
            index,
            top_k=top_k_by_task[task],
            definition_mode=definition_mode,
        )
        first_query, first_retrieval, first_scope = retrieve_for_item(
            task,
            source_items[0],
            index,
            top_k_by_task[task],
            definition_mode=definition_mode,
        )
        print(
            f"[检索库] {task}: {stats}; "
            f"首条query={first_query!r}; "
            f"scope={first_scope}; "
            f"top_k={top_k_by_task[task]}; "
            f"候选数={len(first_retrieval)}; "
            f"候选来源="
            f"{[item.get('book', item['headword']) for item in first_retrieval]}; "
            f"全量审计={audit}"
        )

    if args.dry_run:
        print(
            f"dry-run完成：backend={inference_backend}; "
            f"model={model_path_or_id}; 未调用模型、未写结果。"
        )
        return 0

    local_runtime: tuple[Any, Any, Any] | None = None
    bailian_client = None
    if inference_backend == "local":
        if not model_path or model_path == "/path/to/Qwen3-8B":
            raise ValueError(
                "local后端需要在baseline_config.json中设置rag.model_path，"
                "或通过--model-path传入Qwen3-8B路径。"
            )
        local_runtime = load_qwen(
            model_path,
            load_in_4bit=load_in_4bit,
            device_map=args.device_map,
        )
    else:
        if not bailian_model:
            raise ValueError("百炼模型ID不能为空。")
        bailian_client = create_bailian_client(
            api_key_env=api_key_env,
            base_url=bailian_base_url,
            timeout_seconds=bailian_timeout_seconds,
        )
        print(
            f"[百炼] model={bailian_model}; "
            f"base_url={bailian_base_url}; "
            f"api_key_env={api_key_env}; "
            f"stream={bailian_stream}; enable_thinking=false"
        )

    incomplete = False
    for task in selected:
        source_items, index, stats = prepared[task]
        task_top_k = top_k_by_task[task]
        task_max_reference_chars = max_reference_chars_by_task[task]
        destination = output_path(
            output_root,
            output_model_name,
            task,
            definition_mode=definition_mode,
        )
        results = load_or_initialize(source_items, destination)
        definition_index_sha256 = (
            index.index_sha256
            if isinstance(index, DefinitionRAGIndex)
            else None
        )
        validate_completed_configuration(
            results,
            destination=destination,
            task=task,
            top_k=task_top_k,
            max_reference_chars=task_max_reference_chars,
            model_path_or_id=model_path_or_id,
            inference_backend=inference_backend,
            api_base_url=api_base_url,
            definition_mode=definition_mode,
            definition_index_sha256=definition_index_sha256,
        )
        unfinished = [
            index_number
            for index_number, item in enumerate(results)
            if not prediction_is_complete(item)
        ]
        selected_indices = (
            unfinished
            if args.max_new_items_per_task is None
            else unfinished[: args.max_new_items_per_task]
        )
        print(
            f"[开始] {task}: 总数={len(results)}, "
            f"已完成={len(results) - len(unfinished)}, "
            f"本次处理={len(selected_indices)}"
        )

        for index_number in selected_indices:
            item = results[index_number]
            query, retrieved, retrieval_scope = retrieve_for_item(
                task,
                item,
                index,
                task_top_k,
                definition_mode=definition_mode,
            )
            prompt = build_prompt(
                task,
                item,
                retrieved,
                task_max_reference_chars,
                definition_mode=definition_mode,
                definition_config=definition_config,
            )
            generation_metadata: dict[str, Any] = {}
            try:
                if inference_backend == "local":
                    assert local_runtime is not None
                    torch_module, tokenizer, model = local_runtime
                    answer, generation_metadata = generate_answer_local(
                        torch_module,
                        tokenizer,
                        model,
                        prompt,
                        max_input_tokens=max_input_tokens,
                        max_new_tokens=max_new_tokens,
                    )
                else:
                    assert bailian_client is not None
                    answer, generation_metadata = generate_answer_bailian(
                        bailian_client,
                        prompt,
                        model_id=bailian_model,
                        max_tokens=max_new_tokens,
                        temperature=bailian_temperature,
                        enable_thinking=bailian_enable_thinking,
                        stream=bailian_stream,
                        max_attempts=bailian_max_attempts,
                        retry_wait_seconds=bailian_retry_wait_seconds,
                    )
                item["prediction"] = answer
                item.pop("prediction_error", None)
            except Exception as error:
                item.pop("prediction", None)
                item["prediction_error"] = str(error)
                print(
                    f"[失败] {task}第{index_number + 1}条：{error}",
                    file=sys.stderr,
                )
            metadata = {
                "rag_method": rag_method_for_task(
                    task,
                    definition_mode=definition_mode,
                ),
                "retrieval_query": query,
                "retrieval_scope": retrieval_scope,
                "retrieval": retrieved,
                "retrieval_hit": bool(retrieved),
                "retrieval_top_k": task_top_k,
                "max_reference_chars": task_max_reference_chars,
                "leakage_control": stats["leakage_control"],
                "reference_fields": (
                    stats["visible_fields"]
                    if task == "释义能力"
                    else ["字词", "词性", "例句_校准", "释义"]
                ),
                "model_path_or_id": model_path_or_id,
                "inference_backend": inference_backend,
                "api_base_url": api_base_url,
                "thinking_enabled": False,
                "do_sample": False,
                "temperature": (
                    bailian_temperature
                    if inference_backend == "bailian"
                    else None
                ),
                "generation_provider": generation_metadata.get("provider"),
                "generation_finish_reason": generation_metadata.get(
                    "finish_reason"
                ),
                "reasoning_chars_discarded": generation_metadata.get(
                    "reasoning_chars_discarded",
                    0,
                ),
            }
            if inference_backend == "bailian":
                metadata.update(
                    {
                        "api_request_id": generation_metadata.get(
                            "request_id"
                        ),
                        "api_usage": generation_metadata.get("usage"),
                        "api_response_model": generation_metadata.get(
                            "response_model"
                        ),
                        "api_system_fingerprint": generation_metadata.get(
                            "system_fingerprint"
                        ),
                        "api_completed_at_utc": generation_metadata.get(
                            "completed_at_utc"
                        ),
                        "api_attempts": generation_metadata.get("attempts"),
                        "api_stream": bailian_stream,
                        "api_key_env": api_key_env,
                    }
                )
            if task == "释义能力":
                metadata.update(
                    {
                        "definition_mode": definition_mode,
                        "definition_index_sha256": definition_index_sha256,
                        "source_dictionary": stats["source_dictionary"],
                        "reference_dictionary": stats[
                            "reference_dictionary"
                        ],
                        "cross_dictionary": True,
                        "gold_visible_during_generation": False,
                        "definition_fields_exposed": (
                            definition_mode in {"D3", "D4"}
                        ),
                    }
                )
            item.update(metadata)
            atomic_write_json(destination, results)
            print(f"[保存] {task} {index_number + 1}/{len(results)}")
            if (
                inference_backend == "bailian"
                and bailian_request_interval_seconds > 0
                and index_number != selected_indices[-1]
            ):
                time.sleep(bailian_request_interval_seconds)

        remaining = sum(not prediction_is_complete(item) for item in results)
        if remaining:
            incomplete = True
            print(f"[未完成] {task}尚余{remaining}条；再次运行将断点续跑。")
        else:
            print(f"[完成] {task}: {destination}")

    if incomplete and args.max_new_items_per_task is None:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
