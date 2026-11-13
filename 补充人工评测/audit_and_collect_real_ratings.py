"""审计五份专家评分表并汇总真实人工评分，不修改任何源工作簿。"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from openpyxl import load_workbook


EXPERTS = ["专家A", "专家B", "专家C", "专家D", "专家E"]
SCORE_FIELDS = [
    (8, "semantic_correctness", "语义正确性", 1, 5),
    (9, "completeness", "完整性", 1, 5),
    (10, "conciseness", "简洁性", 1, 5),
    (11, "contextual_appropriateness", "语境适切性", 1, 5),
    (12, "lexicographical_style", "辞书体例", 1, 5),
    (13, "hallucination_severity", "幻觉严重度", 0, 2),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    return value


def cell_value(sheet, row: int, column: int) -> Any:
    return clean(sheet.cell(row=row, column=column).value)


def compare_cells(
    expected_sheet,
    actual_sheet,
    coordinates: list[tuple[int, int]],
    *,
    limit: int = 50,
) -> tuple[int, list[dict[str, Any]]]:
    count = 0
    examples: list[dict[str, Any]] = []
    for row, column in coordinates:
        expected = cell_value(expected_sheet, row, column)
        actual = cell_value(actual_sheet, row, column)
        if expected != actual:
            count += 1
            if len(examples) < limit:
                examples.append(
                    {
                        "cell": actual_sheet.cell(row=row, column=column).coordinate,
                        "expected": expected,
                        "actual": actual,
                    }
                )
    return count, examples


def integer_score(value: Any, low: int, high: int) -> int | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not numeric.is_integer():
        return None
    result = int(numeric)
    if not low <= result <= high:
        return None
    return result


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    result_dir = script_dir / "结果文件"
    derived_dir = script_dir / "derived_data"
    manifest_path = derived_dir / "human_eval_manifest.json"
    master_path = result_dir / "人工专家评测_正式待评分.xlsx"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_units = {
        expert: sorted(
            unit["unit_id"]
            for unit in manifest["units"]
            if expert in unit["assigned_experts"]
        )
        for expert in EXPERTS
    }

    audit: dict[str, Any] = {
        "data_status": "REAL_HUMAN_RATING_AUDIT",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256(manifest_path),
        "master_blank_workbook": str(master_path),
        "master_blank_sha256": sha256(master_path),
        "audit_definition": {
            "editable_data_columns": "H:N（H:M数值评分，N评语）",
            "protected_data_columns": "A:G及O；另核对表头、评测说明、行序、盲评编号、公式和工作表结构",
            "score_ranges": "H:L为1—5；M为0—2",
            "required_comment_rule": "语义正确性≤2或幻觉严重度>0时应填写评语",
        },
        "experts": [],
    }
    ratings: list[dict[str, Any]] = []
    blocking_errors: list[str] = []
    warning_count = 0

    master = load_workbook(master_path, read_only=False, data_only=False)
    try:
        for expert in EXPERTS:
            blank_path = result_dir / f"人工专家评测_正式待评分_{expert}.xlsx"
            scored_path = result_dir / f"人工专家评测_正式待评分_{expert}_已评分.xlsx"
            expert_report: dict[str, Any] = {
                "expert": expert,
                "blank_file": str(blank_path),
                "blank_sha256": sha256(blank_path),
                "scored_file": str(scored_path),
                "scored_sha256": sha256(scored_path),
                "blocking_errors": [],
                "warnings": [],
            }
            blank = load_workbook(blank_path, read_only=False, data_only=False)
            scored = load_workbook(scored_path, read_only=False, data_only=False)
            try:
                expected_sheet_names = ["评测说明", expert]
                for label, workbook in (("分发空白表", blank), ("已评分表", scored)):
                    if workbook.sheetnames != expected_sheet_names:
                        expert_report["blocking_errors"].append(
                            f"{label}工作表结构异常：{workbook.sheetnames!r}"
                        )

                master_sheet = master[expert]
                blank_sheet = blank[expert]
                scored_sheet = scored[expert]

                # 先确认分发空白表本身与总空白表一致。
                all_instruction_cells = [
                    (row, column) for row in range(1, 21) for column in range(1, 7)
                ]
                header_cells = [
                    (row, column) for row in range(1, 4) for column in range(1, 16)
                ]
                protected_data_cells = [
                    (row, column)
                    for row in range(4, 604)
                    for column in (*range(1, 8), 15)
                ]
                master_count, master_examples = compare_cells(
                    master_sheet,
                    blank_sheet,
                    header_cells + protected_data_cells,
                )
                if master_count:
                    expert_report["blocking_errors"].append(
                        f"分发空白表与总空白表有{master_count}个非评分单元格不一致"
                    )
                    expert_report["blank_vs_master_examples"] = master_examples

                instruction_count, instruction_examples = compare_cells(
                    blank["评测说明"],
                    scored["评测说明"],
                    all_instruction_cells,
                )
                protected_count, protected_examples = compare_cells(
                    blank_sheet,
                    scored_sheet,
                    header_cells + protected_data_cells,
                )
                expert_report["instruction_modified_cells"] = instruction_count
                expert_report["protected_modified_cells"] = protected_count
                expert_report["protected_modification_examples"] = (
                    instruction_examples + protected_examples
                )[:50]
                if instruction_count or protected_count:
                    expert_report["blocking_errors"].append(
                        "已评分表修改了评测说明、表头或A:G/O非评分内容"
                    )

                expected_merges = {
                    "评测说明": sorted(str(item) for item in blank["评测说明"].merged_cells.ranges),
                    expert: sorted(str(item) for item in blank_sheet.merged_cells.ranges),
                }
                actual_merges = {
                    "评测说明": sorted(str(item) for item in scored["评测说明"].merged_cells.ranges),
                    expert: sorted(str(item) for item in scored_sheet.merged_cells.ranges),
                }
                expert_report["merged_ranges_unchanged"] = expected_merges == actual_merges
                if expected_merges != actual_merges:
                    expert_report["blocking_errors"].append("合并单元格结构被修改")

                actual_unit_ids = [cell_value(scored_sheet, row, 2) for row in range(4, 604)]
                expert_report["expected_rows"] = len(expected_units[expert])
                expert_report["actual_rows"] = len(actual_unit_ids)
                expert_report["unit_ids_match_manifest"] = actual_unit_ids == expected_units[expert]
                expert_report["duplicate_unit_ids"] = [
                    unit_id
                    for unit_id, count in Counter(actual_unit_ids).items()
                    if count > 1
                ]
                if not expert_report["unit_ids_match_manifest"]:
                    expert_report["blocking_errors"].append(
                        "盲评编号集合或行序与预设分配不一致"
                    )

                invalid_scores: list[dict[str, Any]] = []
                missing_required_comments: list[dict[str, Any]] = []
                comment_count = 0
                for row in range(4, 604):
                    rating: dict[str, Any] = {
                        "data_status": "REAL_HUMAN_RATING",
                        "unit_id": cell_value(scored_sheet, row, 2),
                        "expert_id": expert,
                    }
                    for column, key, label, low, high in SCORE_FIELDS:
                        raw_value = cell_value(scored_sheet, row, column)
                        value = integer_score(raw_value, low, high)
                        if value is None:
                            invalid_scores.append(
                                {
                                    "cell": scored_sheet.cell(row, column).coordinate,
                                    "field": label,
                                    "value": raw_value,
                                    "expected": f"{low}—{high}的整数",
                                }
                            )
                        rating[key] = value
                    comment = cell_value(scored_sheet, row, 14)
                    rating["comment"] = "" if comment in (None, "") else str(comment)
                    if rating["comment"]:
                        comment_count += 1
                    if (
                        rating["semantic_correctness"] is not None
                        and rating["hallucination_severity"] is not None
                        and (
                            rating["semantic_correctness"] <= 2
                            or rating["hallucination_severity"] > 0
                        )
                        and not rating["comment"]
                    ):
                        missing_required_comments.append(
                            {"row": row, "unit_id": rating["unit_id"]}
                        )
                    ratings.append(rating)

                expert_report["valid_score_rows"] = 600 - len(
                    {item["cell"][1:] for item in invalid_scores}
                )
                expert_report["invalid_score_cells"] = len(invalid_scores)
                expert_report["invalid_score_examples"] = invalid_scores[:50]
                expert_report["comment_count"] = comment_count
                expert_report["missing_required_comment_count"] = len(
                    missing_required_comments
                )
                expert_report["missing_required_comment_examples"] = (
                    missing_required_comments[:50]
                )
                if invalid_scores:
                    expert_report["blocking_errors"].append(
                        f"有{len(invalid_scores)}个评分为空或超出允许范围"
                    )
                if missing_required_comments:
                    warning_count += len(missing_required_comments)
                    expert_report["warnings"].append(
                        f"有{len(missing_required_comments)}条低正确性/幻觉样本未填写评语"
                    )

                if expert_report["blocking_errors"]:
                    blocking_errors.extend(
                        f"{expert}: {message}"
                        for message in expert_report["blocking_errors"]
                    )
                audit["experts"].append(expert_report)
            finally:
                blank.close()
                scored.close()
    finally:
        master.close()

    audit["summary"] = {
        "expert_files": len(audit["experts"]),
        "expected_ratings": 3000,
        "collected_ratings": len(ratings),
        "blocking_error_count": len(blocking_errors),
        "required_comment_warning_count": warning_count,
        "non_score_columns_unchanged": not any(
            report["instruction_modified_cells"]
            or report["protected_modified_cells"]
            or not report["merged_ranges_unchanged"]
            for report in audit["experts"]
        ),
        "numerical_scores_complete_and_valid": not any(
            report["invalid_score_cells"] for report in audit["experts"]
        ),
        "unit_assignments_match_manifest": all(
            report["unit_ids_match_manifest"] for report in audit["experts"]
        ),
        "data_compliant_for_statistical_analysis": not blocking_errors,
    }
    unit_task = {unit["unit_id"]: unit["task"] for unit in manifest["units"]}
    ratings_by_unit: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    ratings_by_expert: defaultdict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for rating in ratings:
        ratings_by_unit[rating["unit_id"]].append(rating)
        ratings_by_expert[rating["expert_id"]][rating["unit_id"]] = rating
    dimension_keys = [item[1] for item in SCORE_FIELDS]
    exact_agreement: list[dict[str, Any]] = []
    for task in sorted(set(unit_task.values())):
        unit_ids = [unit_id for unit_id, value in unit_task.items() if value == task]
        for key in dimension_keys:
            exact = sum(
                len({row[key] for row in ratings_by_unit[unit_id]}) == 1
                for unit_id in unit_ids
            )
            exact_agreement.append(
                {
                    "task": task,
                    "dimension": key,
                    "units": len(unit_ids),
                    "three_rater_exact_agreement_units": exact,
                    "three_rater_exact_agreement_rate": exact / len(unit_ids),
                }
            )
    pairwise_vector_agreement: list[dict[str, Any]] = []
    for left, right in itertools.combinations(EXPERTS, 2):
        shared = sorted(
            set(ratings_by_expert[left]) & set(ratings_by_expert[right])
        )
        exact = sum(
            all(
                ratings_by_expert[left][unit_id][key]
                == ratings_by_expert[right][unit_id][key]
                for key in dimension_keys
            )
            for unit_id in shared
        )
        pairwise_vector_agreement.append(
            {
                "expert_left": left,
                "expert_right": right,
                "shared_units": len(shared),
                "six_dimension_exact_match_units": exact,
                "six_dimension_exact_match_rate": exact / len(shared),
            }
        )
    unusually_high_pairs = [
        row
        for row in pairwise_vector_agreement
        if row["shared_units"] >= 100
        and row["six_dimension_exact_match_rate"] >= 0.90
    ]
    audit["agreement_diagnostics"] = {
        "purpose": (
            "用于识别需由研究者确认的异常高一致性；该诊断不能证明或否定评分独立性"
        ),
        "three_rater_exact_agreement": exact_agreement,
        "pairwise_six_dimension_exact_match": pairwise_vector_agreement,
        "unusually_high_pair_count": len(unusually_high_pairs),
        "unusually_high_pairs": unusually_high_pairs,
        "independence_confirmation_required": bool(unusually_high_pairs),
    }
    audit["summary"]["independence_confirmation_required"] = bool(
        unusually_high_pairs
    )
    audit["blocking_errors"] = blocking_errors
    derived_dir.mkdir(parents=True, exist_ok=True)
    audit_path = derived_dir / "real_human_rating_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if blocking_errors:
        print(
            "[审计未通过]",
            json.dumps(audit["summary"], ensure_ascii=False),
        )
        for message in blocking_errors[:20]:
            print(" -", message)
        return 2

    ratings_payload = {
        "data_status": "REAL_HUMAN_RATING",
        "source_audit": str(audit_path),
        "ratings": ratings,
    }
    ratings_path = derived_dir / "real_human_ratings.json"
    ratings_path.write_text(
        json.dumps(ratings_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "[审计通过]",
        json.dumps(audit["summary"], ensure_ascii=False),
    )
    print("[汇总完成]", {"ratings": len(ratings), "output": str(ratings_path)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
