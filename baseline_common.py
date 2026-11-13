#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared utilities for the supplementary baseline and RAG experiments."""

from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Sequence

from openpyxl import load_workbook


_T2S_CONVERTER: Any | None = None


def clean(value: Any) -> str:
    if value is None:
        return ""
    return unicodedata.normalize("NFKC", str(value)).strip()


def compact_key(value: Any) -> str:
    """Return a punctuation/whitespace-insensitive matching key."""
    return "".join(char for char in clean(value) if char.isalnum())


def to_simplified(value: Any) -> str:
    """Convert normalized Chinese text to Simplified Chinese with OpenCC."""
    global _T2S_CONVERTER
    if _T2S_CONVERTER is None:
        try:
            from opencc import OpenCC
        except ImportError as error:
            raise RuntimeError(
                "准确率计算需要繁转简依赖。请运行："
                "pip install opencc-python-reimplemented"
            ) from error
        _T2S_CONVERTER = OpenCC("t2s")
    return _T2S_CONVERTER.convert(clean(value))


def surface_answer_key(value: Any) -> str:
    """Whitespace-insensitive key that preserves the original script."""
    return re.sub(r"\s+", "", clean(value))


def simplified_answer_key(value: Any) -> str:
    """Whitespace-insensitive scoring key after Traditional-to-Simplified conversion."""
    return re.sub(r"\s+", "", to_simplified(value))


def read_json_list(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"JSON顶层必须是列表：{path}")
    if not all(isinstance(item, dict) for item in data):
        raise ValueError(f"JSON列表中包含非对象条目：{path}")
    return data


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def load_config(path: Path) -> tuple[dict[str, Any], Path]:
    config_path = path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("配置文件顶层必须是JSON对象。")
    return config, config_path.parent


def resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def read_xlsx_records(path: Path, sheet_name: str | None = None) -> list[dict[str, str]]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = workbook[sheet_name] if sheet_name else workbook.worksheets[0]
        rows = sheet.iter_rows(values_only=True)
        try:
            raw_headers = next(rows)
        except StopIteration:
            return []
        headers = [clean(value) for value in raw_headers]
        records: list[dict[str, str]] = []
        for raw_row in rows:
            record = {
                header: clean(value)
                for header, value in zip(headers, raw_row)
                if header
            }
            if any(record.values()):
                records.append(record)
        return records
    finally:
        workbook.close()


def require_fields(
    records: Sequence[dict[str, str]],
    required: Iterable[str],
    source: Path,
) -> None:
    available = set().union(*(record.keys() for record in records[:10]))
    missing = [field for field in required if field not in available]
    if missing:
        raise ValueError(
            f"资源缺少字段{missing}：{source}；可用字段={sorted(available)}"
        )


def extract_example(instruction: str) -> str:
    text = clean(instruction)
    quoted = re.search(
        r"古文例句\s*[:：]?\s*[‘'“\"](?P<example>.*?)[’'”\"]"
        r"中的(?:词性|准确含义)",
        text,
        flags=re.DOTALL,
    )
    if quoted:
        return clean(quoted.group("example"))

    fallback = re.search(
        r"古文例句\s*[:：]?\s*(?P<example>.*?)"
        r"中的(?:词性|准确含义)",
        text,
        flags=re.DOTALL,
    )
    if fallback:
        return clean(fallback.group("example")).strip("'‘’\"“”")
    return ""


def _character_ngrams(text: str, size: int) -> set[str]:
    if len(text) < size:
        return {text} if text else set()
    return {
        text[index : index + size]
        for index in range(len(text) - size + 1)
    }


class ExampleLeakageFilter:
    """Detect exact, containment, and high-similarity test-sentence variants."""

    def __init__(
        self,
        test_examples: Iterable[str],
        *,
        similarity_threshold: float = 0.90,
        containment_min_chars: int = 6,
        ngram_size: int = 3,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold必须在0到1之间。")
        if containment_min_chars < 1:
            raise ValueError("containment_min_chars必须大于0。")
        if ngram_size < 1:
            raise ValueError("ngram_size必须大于0。")

        self.test_keys = tuple(
            sorted(
                {
                    compact_key(example)
                    for example in test_examples
                    if compact_key(example)
                }
            )
        )
        self.test_key_set = set(self.test_keys)
        self.similarity_threshold = similarity_threshold
        self.containment_min_chars = containment_min_chars
        self.ngram_size = ngram_size
        self._cache: dict[str, tuple[str | None, str, float]] = {}
        self._ngram_to_test_indices: dict[str, set[int]] = defaultdict(set)
        for index, test_key in enumerate(self.test_keys):
            for ngram in _character_ngrams(test_key, ngram_size):
                self._ngram_to_test_indices[ngram].add(index)

    def classify(self, example: str) -> tuple[str | None, str, float]:
        """Return (reason, matched_test_key, similarity)."""
        key = compact_key(example)
        if not key:
            return None, "", 0.0
        if key in self._cache:
            return self._cache[key]

        if key in self.test_key_set:
            result = ("exact", key, 1.0)
            self._cache[key] = result
            return result

        for test_key in self.test_keys:
            if min(len(key), len(test_key)) < self.containment_min_chars:
                continue
            if key in test_key or test_key in key:
                similarity = (
                    2.0 * min(len(key), len(test_key)) / (len(key) + len(test_key))
                )
                result = ("containment", test_key, similarity)
                self._cache[key] = result
                return result

        candidate_indices: set[int] = set()
        for ngram in _character_ngrams(key, self.ngram_size):
            candidate_indices.update(self._ngram_to_test_indices.get(ngram, set()))
        if not candidate_indices and len(key) < self.ngram_size:
            candidate_indices.update(range(len(self.test_keys)))

        best_key = ""
        best_similarity = 0.0
        for index in candidate_indices:
            test_key = self.test_keys[index]
            maximum_possible = (
                2.0 * min(len(key), len(test_key)) / (len(key) + len(test_key))
            )
            if maximum_possible < self.similarity_threshold:
                continue
            similarity = SequenceMatcher(
                None,
                key,
                test_key,
                autojunk=False,
            ).ratio()
            if similarity > best_similarity:
                best_key = test_key
                best_similarity = similarity

        if best_similarity >= self.similarity_threshold:
            result = ("similarity", best_key, best_similarity)
        else:
            result = (None, "", best_similarity)
        self._cache[key] = result
        return result


def filter_records_by_test_examples(
    records: Sequence[dict[str, str]],
    *,
    example_field: str,
    test_examples: Iterable[str],
    similarity_threshold: float = 0.90,
    containment_min_chars: int = 6,
    ngram_size: int = 3,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    detector = ExampleLeakageFilter(
        test_examples,
        similarity_threshold=similarity_threshold,
        containment_min_chars=containment_min_chars,
        ngram_size=ngram_size,
    )
    kept: list[dict[str, str]] = []
    reason_counts: Counter[str] = Counter()
    for record in records:
        reason, _, _ = detector.classify(record.get(example_field, ""))
        if reason is None:
            kept.append(record)
        else:
            reason_counts[reason] += 1

    stats = {
        "input_rows": len(records),
        "kept_rows": len(kept),
        "excluded_rows": len(records) - len(kept),
        "excluded_exact": reason_counts["exact"],
        "excluded_containment": reason_counts["containment"],
        "excluded_similarity": reason_counts["similarity"],
        "test_example_keys": len(detector.test_keys),
        "similarity_threshold": similarity_threshold,
        "containment_min_chars": containment_min_chars,
        "ngram_size": ngram_size,
    }
    return kept, stats


def tokenize(text: str) -> list[str]:
    """Character unigrams + adjacent CJK bigrams + Latin/number tokens."""
    normalized = clean(text).lower()
    tokens: list[str] = []
    cjk_run: list[str] = []

    def flush_cjk() -> None:
        if not cjk_run:
            return
        tokens.extend(cjk_run)
        tokens.extend(
            cjk_run[index] + cjk_run[index + 1]
            for index in range(len(cjk_run) - 1)
        )
        cjk_run.clear()

    for match in re.finditer(r"[\u3400-\u9fff]|[a-z0-9]+|.", normalized):
        token = match.group(0)
        if re.fullmatch(r"[\u3400-\u9fff]", token):
            cjk_run.append(token)
        else:
            flush_cjk()
            if re.fullmatch(r"[a-z0-9]+", token):
                tokens.append(token)
    flush_cjk()
    return tokens


@dataclass(frozen=True)
class RetrievalDocument:
    headword: str
    answer: str
    text: str
    example: str = ""
    pos: str = ""
    source_row: int = 0

    def as_result(self, rank: int, score: float) -> dict[str, Any]:
        result: dict[str, Any] = {
            "rank": rank,
            "score": round(score, 6),
            "headword": self.headword,
            "answer": self.answer,
            "source_row": self.source_row,
        }
        if self.example:
            result["example"] = self.example
        if self.pos:
            result["pos"] = self.pos
        return result


@dataclass(frozen=True)
class UsageEvidence:
    """Unlabelled corpus evidence used for definition induction."""

    headword: str
    example: str
    paragraph: str = ""
    book: str = ""
    chapter: str = ""
    source_row: int = 0

    def as_result(self, rank: int, score: float) -> dict[str, Any]:
        result: dict[str, Any] = {
            "rank": rank,
            "score": round(score, 6),
            "headword": self.headword,
            "example": self.example,
            "source_row": self.source_row,
        }
        if self.paragraph and compact_key(self.paragraph) != compact_key(self.example):
            result["paragraph"] = self.paragraph
        if self.book:
            result["book"] = self.book
        if self.chapter:
            result["chapter"] = self.chapter
        return result


def _ngram_jaccard(left: str, right: str, size: int) -> float:
    left_ngrams = _character_ngrams(compact_key(left), size)
    right_ngrams = _character_ngrams(compact_key(right), size)
    if not left_ngrams and not right_ngrams:
        return 1.0
    union = left_ngrams | right_ngrams
    return len(left_ngrams & right_ngrams) / len(union) if union else 0.0


def build_usage_evidence_documents(
    records: Sequence[dict[str, str]],
    *,
    key_field: str,
    example_field: str,
    paragraph_field: str,
    book_field: str,
    chapter_field: str,
    similarity_threshold: float = 0.90,
    containment_min_chars: int = 6,
) -> tuple[list[UsageEvidence], dict[str, int | float]]:
    """Build a label-free, within-headword deduplicated evidence corpus.

    Only the five explicitly supplied evidence fields are accessed. In
    particular, definition and POS labels cannot enter the retrieval index.
    When two examples are near duplicates, the longer example is retained.
    """

    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold必须在0到1之间。")
    if containment_min_chars < 1:
        raise ValueError("containment_min_chars必须大于0。")

    def optional_metadata(record: dict[str, str], field: str) -> str:
        value = clean(record.get(field))
        if value.lower() in {"0", "nan", "none", "null"}:
            return ""
        return value

    grouped: dict[str, list[UsageEvidence]] = defaultdict(list)
    valid_rows = 0
    for row_number, record in enumerate(records, start=2):
        headword = clean(record.get(key_field))
        example = clean(record.get(example_field))
        if not headword or not compact_key(example):
            continue
        valid_rows += 1
        grouped[headword].append(
            UsageEvidence(
                headword=headword,
                example=example,
                paragraph=optional_metadata(record, paragraph_field),
                book=optional_metadata(record, book_field),
                chapter=optional_metadata(record, chapter_field),
                source_row=row_number,
            )
        )

    reason_counts: Counter[str] = Counter()
    documents: list[UsageEvidence] = []
    for headword in sorted(grouped):
        # Keep the more informative (longer) form when one variant contains
        # another. The row number gives deterministic tie-breaking.
        candidates = sorted(
            grouped[headword],
            key=lambda item: (-len(compact_key(item.example)), item.source_row),
        )
        kept: list[UsageEvidence] = []
        kept_keys: list[str] = []
        for candidate in candidates:
            candidate_key = compact_key(candidate.example)
            duplicate_reason = ""
            for kept_key in kept_keys:
                if candidate_key == kept_key:
                    duplicate_reason = "exact"
                    break
                if (
                    min(len(candidate_key), len(kept_key))
                    >= containment_min_chars
                    and (
                        candidate_key in kept_key
                        or kept_key in candidate_key
                    )
                ):
                    duplicate_reason = "containment"
                    break
                if (
                    SequenceMatcher(None, candidate_key, kept_key).ratio()
                    >= similarity_threshold
                ):
                    duplicate_reason = "similarity"
                    break
            if duplicate_reason:
                reason_counts[duplicate_reason] += 1
                continue
            kept.append(candidate)
            kept_keys.append(candidate_key)
        documents.extend(sorted(kept, key=lambda item: item.source_row))

    stats: dict[str, int | float] = {
        "input_rows": len(records),
        "valid_evidence_rows": valid_rows,
        "evidence_documents": len(documents),
        "evidence_headwords": len({item.headword for item in documents}),
        "excluded_exact": reason_counts["exact"],
        "excluded_containment": reason_counts["containment"],
        "excluded_similarity": reason_counts["similarity"],
        "similarity_threshold": similarity_threshold,
        "containment_min_chars": containment_min_chars,
    }
    return documents, stats


class UsageEvidenceIndex:
    """Select diverse same-headword examples without using hidden labels."""

    def __init__(
        self,
        documents: Sequence[UsageEvidence],
        *,
        diversity_ngram_size: int = 2,
    ) -> None:
        if diversity_ngram_size < 1:
            raise ValueError("diversity_ngram_size必须大于0。")
        self.diversity_ngram_size = diversity_ngram_size
        grouped: dict[str, list[UsageEvidence]] = defaultdict(list)
        for document in documents:
            grouped[clean(document.headword)].append(document)
        self.documents_by_headword = {
            headword: sorted(items, key=lambda item: item.source_row)
            for headword, items in grouped.items()
        }
        simplified_grouped: dict[str, list[UsageEvidence]] = defaultdict(list)
        for headword, items in self.documents_by_headword.items():
            simplified_grouped[to_simplified(headword)].extend(items)
        self.documents_by_simplified_headword = {
            headword: sorted(
                {item.source_row: item for item in items}.values(),
                key=lambda item: item.source_row,
            )
            for headword, items in simplified_grouped.items()
        }

    def has_evidence(self, headword: str) -> bool:
        normalized = clean(headword)
        if normalized in self.documents_by_headword:
            return True
        simplified = to_simplified(normalized)
        return bool(
            simplified != normalized
            and simplified in self.documents_by_simplified_headword
        )

    def search(
        self,
        headword: str,
        *,
        top_k: int = 10,
    ) -> tuple[str, list[tuple[float, UsageEvidence]]]:
        normalized = clean(headword)
        candidates = list(self.documents_by_headword.get(normalized, []))
        scope = "same-headword-usage-evidence"
        if not candidates:
            simplified = to_simplified(normalized)
            if simplified != normalized:
                candidates = list(
                    self.documents_by_simplified_headword.get(
                        simplified,
                        [],
                    )
                )
                if candidates:
                    scope = "simplified-headword-usage-evidence"
        if not candidates or top_k <= 0:
            return "no-evidence", []

        selected: list[tuple[float, UsageEvidence]] = []
        selected_books: set[str] = set()
        selected_sources: set[tuple[str, str]] = set()
        remaining = candidates
        while remaining and len(selected) < top_k:
            scored: list[tuple[float, int, UsageEvidence]] = []
            selected_examples = [item.example for _, item in selected]
            for candidate in remaining:
                if selected_examples:
                    maximum_similarity = max(
                        _ngram_jaccard(
                            candidate.example,
                            selected_example,
                            self.diversity_ngram_size,
                        )
                        for selected_example in selected_examples
                    )
                    novelty = 1.0 - maximum_similarity
                else:
                    novelty = 1.0
                unseen_book = float(
                    bool(candidate.book)
                    and candidate.book not in selected_books
                )
                source = (candidate.book, candidate.chapter)
                unseen_source = float(
                    bool(candidate.book or candidate.chapter)
                    and source not in selected_sources
                )
                informativeness = min(
                    len(compact_key(candidate.example)),
                    80,
                ) / 80.0
                score = (
                    2.0 * unseen_book
                    + unseen_source
                    + 1.5 * novelty
                    + 0.25 * informativeness
                )
                scored.append((score, -candidate.source_row, candidate))

            score, _, chosen = max(
                scored,
                key=lambda item: (item[0], item[1]),
            )
            selected.append((score, chosen))
            if chosen.book:
                selected_books.add(chosen.book)
            if chosen.book or chosen.chapter:
                selected_sources.add((chosen.book, chosen.chapter))
            remaining.remove(chosen)

        return scope, selected


class BM25Index:
    def __init__(
        self,
        documents: Sequence[RetrievalDocument],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not documents:
            raise ValueError("BM25语料库为空。")
        self.documents = list(documents)
        self.k1 = k1
        self.b = b
        self.term_frequencies: list[Counter[str]] = []
        self.document_lengths: list[int] = []
        document_frequency: Counter[str] = Counter()

        for document in self.documents:
            frequencies = Counter(tokenize(document.text))
            self.term_frequencies.append(frequencies)
            length = sum(frequencies.values())
            self.document_lengths.append(length)
            document_frequency.update(frequencies.keys())

        self.average_length = (
            sum(self.document_lengths) / len(self.document_lengths)
        )
        count = len(self.documents)
        self.idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str, top_k: int = 3) -> list[tuple[float, RetrievalDocument]]:
        query_terms = set(tokenize(query))
        if not query_terms:
            return []
        scores: list[tuple[float, int]] = []
        for index, frequencies in enumerate(self.term_frequencies):
            score = 0.0
            length = self.document_lengths[index]
            normalization = self.k1 * (
                1.0 - self.b + self.b * length / max(self.average_length, 1e-12)
            )
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                score += self.idf.get(term, 0.0) * (
                    frequency * (self.k1 + 1.0) / (frequency + normalization)
                )
            if score > 0:
                scores.append((score, index))
        scores.sort(key=lambda item: (-item[0], item[1]))
        return [
            (score, self.documents[index])
            for score, index in scores[: max(0, top_k)]
        ]


class HeadwordFirstBM25:
    """Search same-headword contexts first, then use the full corpus as fallback."""

    def __init__(
        self,
        documents: Sequence[RetrievalDocument],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.global_index = BM25Index(documents, k1=k1, b=b)
        grouped: dict[str, list[RetrievalDocument]] = defaultdict(list)
        for document in documents:
            grouped[clean(document.headword)].append(document)
        self.documents_by_headword = dict(grouped)
        self._headword_indices: dict[str, BM25Index] = {}

    def search(
        self,
        query: str,
        *,
        headword: str,
        top_k: int = 3,
    ) -> tuple[str, list[tuple[float, RetrievalDocument]]]:
        normalized_headword = clean(headword)
        same_headword_documents = self.documents_by_headword.get(
            normalized_headword,
            [],
        )
        if same_headword_documents:
            index = self._headword_indices.get(normalized_headword)
            if index is None:
                index = BM25Index(
                    same_headword_documents,
                    k1=self.k1,
                    b=self.b,
                )
                self._headword_indices[normalized_headword] = index
            results = index.search(query, top_k=top_k)
            if results:
                return "same-headword", results

        return (
            "global-backoff",
            self.global_index.search(query, top_k=top_k),
        )


@dataclass
class MajorityLookup:
    prediction_by_key: dict[str, str]
    candidates_by_key: dict[str, list[tuple[str, int]]]
    global_prediction: str


def build_majority_lookup(
    records: Sequence[dict[str, str]],
    key_field: str,
    value_field: str,
) -> MajorityLookup:
    per_key: dict[str, Counter[str]] = defaultdict(Counter)
    global_counts: Counter[str] = Counter()
    for record in records:
        key = clean(record.get(key_field))
        value = clean(record.get(value_field))
        if key and value:
            per_key[key][value] += 1
            global_counts[value] += 1

    def choose(counter: Counter[str]) -> str:
        maximum = max(counter.values())
        return sorted(value for value, count in counter.items() if count == maximum)[0]

    prediction_by_key = {key: choose(counter) for key, counter in per_key.items()}
    candidates_by_key = {
        key: sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        for key, counter in per_key.items()
    }
    global_prediction = choose(global_counts) if global_counts else ""
    return MajorityLookup(
        prediction_by_key=prediction_by_key,
        candidates_by_key=candidates_by_key,
        global_prediction=global_prediction,
    )


def build_definition_documents(
    records: Sequence[dict[str, str]],
    *,
    key_field: str,
    answer_field: str,
    excluded_headwords: set[str],
) -> list[RetrievalDocument]:
    documents: list[RetrievalDocument] = []
    seen: set[tuple[str, str]] = set()
    for row_number, record in enumerate(records, start=2):
        headword = clean(record.get(key_field))
        answer = clean(record.get(answer_field))
        if not headword or not answer or headword in excluded_headwords:
            continue
        identity = (headword, answer)
        if identity in seen:
            continue
        seen.add(identity)
        documents.append(
            RetrievalDocument(
                headword=headword,
                answer=answer,
                text=f"{headword} {answer}",
                source_row=row_number,
            )
        )
    return documents


def build_pos_context_documents(
    records: Sequence[dict[str, str]],
    *,
    key_field: str,
    pos_field: str,
    example_field: str,
) -> list[RetrievalDocument]:
    """Build POS retrieval documents without indexing labels or definitions."""
    documents: list[RetrievalDocument] = []
    seen: set[tuple[str, str, str]] = set()
    for row_number, record in enumerate(records, start=2):
        headword = clean(record.get(key_field))
        pos = clean(record.get(pos_field))
        example = clean(record.get(example_field))
        example_key = compact_key(example)
        if not headword or not pos or not example_key:
            continue
        identity = (headword, example_key, pos)
        if identity in seen:
            continue
        seen.add(identity)
        documents.append(
            RetrievalDocument(
                headword=headword,
                answer=pos,
                example=example,
                pos=pos,
                # POS is payload only. It must not influence BM25 ranking.
                text=f"{headword} {example}",
                source_row=row_number,
            )
        )
    return documents


def build_context_documents(
    records: Sequence[dict[str, str]],
    *,
    key_field: str,
    answer_field: str,
    example_field: str,
    pos_field: str,
    excluded_example_keys: set[str],
) -> list[RetrievalDocument]:
    documents: list[RetrievalDocument] = []
    seen: set[tuple[str, str, str]] = set()
    for row_number, record in enumerate(records, start=2):
        headword = clean(record.get(key_field))
        answer = clean(record.get(answer_field))
        example = clean(record.get(example_field))
        pos = clean(record.get(pos_field))
        example_key = compact_key(example)
        if (
            not headword
            or not answer
            or not example
            or example_key in excluded_example_keys
        ):
            continue
        identity = (headword, example_key, answer)
        if identity in seen:
            continue
        seen.add(identity)
        documents.append(
            RetrievalDocument(
                headword=headword,
                answer=answer,
                example=example,
                pos=pos,
                text=f"{headword} {example} {pos} {answer}",
                source_row=row_number,
            )
        )
    return documents


def retrieval_results(
    index: BM25Index,
    query: str,
    top_k: int,
) -> list[dict[str, Any]]:
    return [
        document.as_result(rank, score)
        for rank, (score, document) in enumerate(
            index.search(query, top_k=top_k),
            start=1,
        )
    ]


def headword_first_retrieval_results(
    index: HeadwordFirstBM25,
    query: str,
    *,
    headword: str,
    top_k: int,
) -> tuple[str, list[dict[str, Any]]]:
    scope, results = index.search(
        query,
        headword=headword,
        top_k=top_k,
    )
    return (
        scope,
        [
            document.as_result(rank, score)
            for rank, (score, document) in enumerate(results, start=1)
        ],
    )


def prediction_is_complete(item: dict[str, Any]) -> bool:
    value = item.get("prediction")
    if not isinstance(value, str) or not value.strip():
        return False
    normalized = value.strip().upper()
    return not (
        normalized.startswith("ERROR:")
        or normalized.startswith("[ERROR]")
    )


def validate_resume(
    source: Sequence[dict[str, Any]],
    saved: Sequence[dict[str, Any]],
    path: Path,
) -> None:
    if len(source) != len(saved):
        raise ValueError(f"断点文件条数与评测集不一致：{path}")
    for index, (left, right) in enumerate(zip(source, saved), start=1):
        for field in ("instruction", "input", "output"):
            if left.get(field) != right.get(field):
                raise ValueError(
                    f"断点文件第{index}条的{field}与评测集不一致：{path}"
                )
