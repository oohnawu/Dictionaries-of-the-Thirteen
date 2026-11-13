#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""分析真实或模拟专家评分：一致性、Spearman相关、95% CI及案例。"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

import numpy as np
from openpyxl import load_workbook
from scipy.stats import spearmanr


DIMENSIONS = (
    "semantic_correctness",
    "completeness",
    "conciseness",
    "contextual_appropriateness",
    "lexicographical_style",
    "hallucination_severity",
)
SCORE_LABELS = {
    "语义正确性": "semantic_correctness",
    "完整性": "completeness",
    "简洁性": "conciseness",
    "语境适切性": "contextual_appropriateness",
    "辞书体例": "lexicographical_style",
    "幻觉严重度": "hallucination_severity",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="./derived_data/human_eval_manifest.json",
    )
    parser.add_argument(
        "--ratings",
        default="./derived_data/real_human_ratings.json",
    )
    parser.add_argument(
        "--output",
        default="./derived_data/real_human_analysis.json",
    )
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260731)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_ratings_json(path: Path) -> tuple[str, list[dict[str, Any]]]:
    data = read_json(path)
    if isinstance(data, list):
        return "UNSPECIFIED", data
    return str(data.get("data_status", "UNSPECIFIED")), data["ratings"]


def read_ratings_xlsx(path: Path) -> tuple[str, list[dict[str, Any]]]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    ratings: list[dict[str, Any]] = []
    try:
        for sheet in workbook.worksheets:
            if not sheet.title.startswith("专家"):
                continue
            rows = list(sheet.iter_rows(values_only=True))
            header_index = next(
                (
                    index
                    for index, row in enumerate(rows[:12])
                    if row and "盲评编号" in row
                ),
                None,
            )
            if header_index is None:
                continue
            headers = [str(value or "").strip() for value in rows[header_index]]
            for row in rows[header_index + 1:]:
                record = dict(zip(headers, row))
                unit_id = str(record.get("盲评编号") or "").strip()
                if not unit_id:
                    continue
                if all(record.get(label) in (None, "") for label in SCORE_LABELS):
                    continue
                rating = {
                    "data_status": "REAL_HUMAN_RATING",
                    "unit_id": unit_id,
                    "expert_id": sheet.title,
                    "comment": str(record.get("评语") or "").strip(),
                }
                for label, key in SCORE_LABELS.items():
                    value = record.get(label)
                    if value in (None, ""):
                        raise ValueError(
                            f"{sheet.title} {unit_id} 的{label}为空。"
                        )
                    rating[key] = int(value)
                ratings.append(rating)
    finally:
        workbook.close()
    return "REAL_HUMAN_RATING", ratings


def validate_ratings(
    ratings: list[dict[str, Any]],
    units_by_id: dict[str, dict[str, Any]],
) -> None:
    seen: set[tuple[str, str]] = set()
    counts: Counter[str] = Counter()
    for row in ratings:
        unit_id = row["unit_id"]
        expert = row["expert_id"]
        if unit_id not in units_by_id:
            raise ValueError(f"未知盲评编号：{unit_id}")
        pair = (unit_id, expert)
        if pair in seen:
            raise ValueError(f"重复评分：{pair}")
        seen.add(pair)
        counts[unit_id] += 1
        for dimension in DIMENSIONS[:-1]:
            value = int(row[dimension])
            if not 1 <= value <= 5:
                raise ValueError(f"{pair} {dimension}={value}超出1—5。")
        hallucination = int(row["hallucination_severity"])
        if not 0 <= hallucination <= 2:
            raise ValueError(f"{pair}幻觉严重度超出0—2。")
    bad = {unit_id: count for unit_id, count in counts.items() if count != 3}
    if len(counts) != len(units_by_id) or bad:
        raise ValueError(
            f"应为每个输出3份评分；已评分单元={len(counts)}/"
            f"{len(units_by_id)}，异常={list(bad.items())[:10]}"
        )


def ordinal_distance(
    left: int,
    right: int,
    marginals: dict[int, float],
) -> float:
    if left == right:
        return 0.0
    low, high = sorted((left, right))
    interior = sum(
        count
        for value, count in marginals.items()
        if low < value < high
    )
    distance = (
        marginals.get(low, 0.0) / 2
        + interior
        + marginals.get(high, 0.0) / 2
    )
    return distance * distance


def krippendorff_alpha_ordinal(
    ratings_by_unit: Iterable[list[int]],
) -> float | None:
    coincidence: defaultdict[tuple[int, int], float] = defaultdict(float)
    for ratings in ratings_by_unit:
        values = [int(value) for value in ratings]
        n = len(values)
        if n < 2:
            continue
        counts = Counter(values)
        for left, left_count in counts.items():
            for right, right_count in counts.items():
                numerator = left_count * (
                    right_count - (1 if left == right else 0)
                )
                coincidence[(left, right)] += numerator / (n - 1)
    total = sum(coincidence.values())
    if total <= 1:
        return None
    marginals: defaultdict[int, float] = defaultdict(float)
    for (left, _), value in coincidence.items():
        marginals[left] += value
    observed = sum(
        value * ordinal_distance(left, right, marginals)
        for (left, right), value in coincidence.items()
    ) / total
    expected_numerator = 0.0
    for left, left_count in marginals.items():
        for right, right_count in marginals.items():
            pair_count = left_count * (
                right_count - (1 if left == right else 0)
            ) / (total - 1)
            expected_numerator += (
                pair_count * ordinal_distance(left, right, marginals)
            )
    expected = expected_numerator / total
    if expected == 0:
        return 1.0 if observed == 0 else None
    return 1.0 - observed / expected


def bootstrap_spearman(
    x: list[float],
    y: list[float],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    result = spearmanr(x, y)
    rho = float(result.statistic)
    p_value = float(result.pvalue)
    rng = random.Random(seed)
    estimates: list[float] = []
    n = len(x)
    for _ in range(iterations):
        indices = [rng.randrange(n) for _ in range(n)]
        sample_x = [x[index] for index in indices]
        sample_y = [y[index] for index in indices]
        estimate = spearmanr(sample_x, sample_y).statistic
        if not math.isnan(estimate):
            estimates.append(float(estimate))
    low, high = np.percentile(estimates, [2.5, 97.5])
    return {
        "n": n,
        "spearman_rho": rho,
        "p_value": p_value,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_iterations": len(estimates),
    }


def summarize_unit_ratings(
    units: list[dict[str, Any]],
    ratings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for rating in ratings:
        grouped[rating["unit_id"]].append(rating)
    summaries: list[dict[str, Any]] = []
    for unit in units:
        rows = grouped[unit["unit_id"]]
        averages = {
            dimension: mean(float(row[dimension]) for row in rows)
            for dimension in DIMENSIONS
        }
        semantic_composite = mean(
            averages[key]
            for key in (
                "semantic_correctness",
                "completeness",
                "contextual_appropriateness",
            )
        )
        overall_quality = mean(
            averages[key] for key in DIMENSIONS[:-1]
        )
        summaries.append(
            {
                **unit,
                **{f"mean_{key}": value for key, value in averages.items()},
                "human_semantic_composite": semantic_composite,
                "human_overall_quality": overall_quality,
                "rating_count": len(rows),
            }
        )
    return summaries


def consistency_table(
    units: list[dict[str, Any]],
    ratings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    unit_map = {unit["unit_id"]: unit for unit in units}
    grouped: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
    for row in ratings:
        task = unit_map[row["unit_id"]]["task"]
        for dimension in DIMENSIONS:
            grouped[(task, dimension)].append(
                (row["unit_id"], int(row[dimension]))
            )
    output: list[dict[str, Any]] = []
    for task in sorted({unit["task"] for unit in units}):
        task_rows = [row for row in ratings if unit_map[row["unit_id"]]["task"] == task]
        for dimension in DIMENSIONS:
            by_unit: defaultdict[str, list[int]] = defaultdict(list)
            for row in task_rows:
                by_unit[row["unit_id"]].append(int(row[dimension]))
            alpha = krippendorff_alpha_ordinal(by_unit.values())
            values = [int(row[dimension]) for row in task_rows]
            output.append(
                {
                    "task": task,
                    "dimension": dimension,
                    "krippendorff_alpha_ordinal": alpha,
                    "rating_count": len(values),
                    "mean": mean(values),
                    "std": pstdev(values),
                }
            )
    return output


def correlation_table(
    unit_summaries: list[dict[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    human_metrics = (
        "human_semantic_composite",
        "human_overall_quality",
        "mean_hallucination_severity",
    )
    automatic_metrics = ("rouge_avg_percent", "sbert_similarity")
    groups = [("合并", unit_summaries)] + [
        (task, [row for row in unit_summaries if row["task"] == task])
        for task in sorted({row["task"] for row in unit_summaries})
    ]
    output: list[dict[str, Any]] = []
    counter = 0
    for group_name, rows in groups:
        for human_metric in human_metrics:
            for automatic_metric in automatic_metrics:
                counter += 1
                result = bootstrap_spearman(
                    [float(row[human_metric]) for row in rows],
                    [float(row[automatic_metric]) for row in rows],
                    iterations=iterations,
                    seed=seed + counter * 97,
                )
                output.append(
                    {
                        "group": group_name,
                        "human_metric": human_metric,
                        "automatic_metric": automatic_metric,
                        **result,
                    }
                )
    return output


def model_summary(unit_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in unit_summaries:
        grouped[(row["task"], row["model"])].append(row)
    output: list[dict[str, Any]] = []
    for (task, model), rows in sorted(grouped.items()):
        output.append(
            {
                "task": task,
                "model": model,
                "model_role": rows[0]["model_role"],
                "n": len(rows),
                "human_semantic_mean": mean(
                    row["human_semantic_composite"] for row in rows
                ),
                "human_semantic_std": pstdev(
                    row["human_semantic_composite"] for row in rows
                ),
                "human_overall_mean": mean(
                    row["human_overall_quality"] for row in rows
                ),
                "human_overall_std": pstdev(
                    row["human_overall_quality"] for row in rows
                ),
                "hallucination_mean": mean(
                    row["mean_hallucination_severity"] for row in rows
                ),
                "rouge_avg_percent": mean(
                    row["rouge_avg_percent"] for row in rows
                ),
                "sbert_similarity_percent": mean(
                    row["sbert_similarity_percent"] for row in rows
                ),
            }
        )
    return output


def qualitative_cases(unit_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for task in sorted({row["task"] for row in unit_summaries}):
        rows = [row for row in unit_summaries if row["task"] == task]
        rouge_low, rouge_high = np.percentile(
            [row["rouge_avg_percent"] for row in rows],
            [25, 75],
        )
        sbert_high = float(np.percentile(
            [row["sbert_similarity"] for row in rows],
            75,
        ))
        human_low = float(np.percentile(
            [row["human_overall_quality"] for row in rows],
            25,
        ))
        definitions = (
            (
                "ROUGE低、SBERT高、专家评价高",
                lambda row: row["rouge_avg_percent"] <= rouge_low
                and row["sbert_similarity"] >= sbert_high
                and row["human_overall_quality"] >= 3.5,
                lambda row: (
                    row["human_overall_quality"]
                    + row["sbert_similarity"]
                    - row["rouge_avg_percent"] / 100
                ),
            ),
            (
                "ROUGE高、专家评价低",
                lambda row: row["rouge_avg_percent"] >= rouge_high
                and row["human_overall_quality"] <= human_low,
                lambda row: (
                    row["rouge_avg_percent"] / 100
                    - row["human_overall_quality"] / 5
                ),
            ),
            (
                "SBERT高但存在古汉语细微误解或幻觉",
                lambda row: row["sbert_similarity"] >= sbert_high
                and (
                    row["mean_semantic_correctness"] <= 3.0
                    or row["mean_hallucination_severity"] >= 0.67
                ),
                lambda row: (
                    row["sbert_similarity"]
                    + row["mean_hallucination_severity"] / 2
                    - row["mean_semantic_correctness"] / 5
                ),
            ),
            (
                "输出流畅但出现文化或辞书学幻觉",
                lambda row: (
                    row["mean_conciseness"] >= 4.0
                    and row["mean_lexicographical_style"] >= 4.0
                    and row["mean_hallucination_severity"] >= 0.67
                ),
                lambda row: (
                    row["mean_conciseness"]
                    + row["mean_lexicographical_style"]
                    + row["mean_hallucination_severity"]
                ),
            ),
        )
        for case_type, predicate, ranking in definitions:
            matches = [row for row in rows if predicate(row)]
            status = "满足预设条件"
            if not matches:
                matches = rows
                status = "无严格命中；列出最接近候选供人工复核"
            selected = sorted(matches, key=ranking, reverse=True)[:2]
            for row in selected:
                output.append(
                    {
                        "task": task,
                        "case_type": case_type,
                        "selection_status": status,
                        "unit_id": row["unit_id"],
                        "model": row["model"],
                        "headword": row["headword"],
                        "context": row["context"],
                        "reference": row["reference"],
                        "prediction": row["prediction"],
                        "rouge_avg_percent": row["rouge_avg_percent"],
                        "sbert_similarity_percent": row["sbert_similarity_percent"],
                        "human_semantic_composite": row["human_semantic_composite"],
                        "human_overall_quality": row["human_overall_quality"],
                        "hallucination_mean": row["mean_hallucination_severity"],
                    }
                )
    return output


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    manifest_path = Path(args.manifest)
    ratings_path = Path(args.ratings)
    output_path = Path(args.output)
    if not manifest_path.is_absolute():
        manifest_path = (script_dir / manifest_path).resolve()
    if not ratings_path.is_absolute():
        ratings_path = (script_dir / ratings_path).resolve()
    if not output_path.is_absolute():
        output_path = (script_dir / output_path).resolve()

    manifest = read_json(manifest_path)
    units = manifest["units"]
    units_by_id = {unit["unit_id"]: unit for unit in units}
    if ratings_path.suffix.lower() == ".xlsx":
        data_status, ratings = read_ratings_xlsx(ratings_path)
    else:
        data_status, ratings = read_ratings_json(ratings_path)
    validate_ratings(ratings, units_by_id)
    unit_summaries = summarize_unit_ratings(units, ratings)
    output = {
        "data_status": data_status,
        "warning": (
            "若data_status含SYNTHETIC，本结果只能用于流程测试，"
            "不能作为真实专家实验证据。"
        ),
        "design": manifest["design"],
        "consistency": consistency_table(units, ratings),
        "correlations": correlation_table(
            unit_summaries,
            iterations=args.bootstrap,
            seed=args.seed,
        ),
        "model_summary": model_summary(unit_summaries),
        "qualitative_cases": qualitative_cases(unit_summaries),
        "unit_summaries": unit_summaries,
        "ratings": ratings,
    }
    atomic_json(output_path, output)
    print(
        "[完成]",
        {
            "data_status": data_status,
            "units": len(unit_summaries),
            "ratings": len(ratings),
            "output": str(output_path),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

