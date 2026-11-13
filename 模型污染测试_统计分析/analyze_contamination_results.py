from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
from opencc import OpenCC


PROMPT_ORDER = ["general", "dictionary", "source"]
GROUP_LABELS = {
    "A": "DTC entries",
    "B": "HDC entries",
    "C": "Public classical texts",
}


def clean_scalar(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def normalize_text(value: object, converter: OpenCC) -> str:
    text = unicodedata.normalize("NFKC", clean_scalar(value))
    text = converter.convert(text).lower()
    return "".join(ch for ch in text if unicodedata.category(ch)[0] in {"L", "N"})


def levenshtein_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, char_left in enumerate(left, start=1):
        current = [i]
        for j, char_right in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (char_left != char_right),
                )
            )
        previous = current
    return previous[-1]


def similarity_metrics(reference: str, generated: str) -> dict[str, object]:
    if not reference or not generated:
        return {
            "exact_reproduction": False,
            "reference_contained": False,
            "sequence_ratio": 0.0,
            "reference_coverage": 0.0,
            "matched_f1": 0.0,
            "edit_similarity": 0.0,
            "longest_block_ratio": 0.0,
        }

    matcher = SequenceMatcher(None, reference, generated, autojunk=False)
    blocks = matcher.get_matching_blocks()
    matched = sum(block.size for block in blocks)
    longest = max((block.size for block in blocks), default=0)
    max_len = max(len(reference), len(generated))
    edit_similarity = 1.0 - levenshtein_distance(reference, generated) / max_len
    return {
        "exact_reproduction": reference == generated,
        "reference_contained": reference in generated,
        "sequence_ratio": matcher.ratio(),
        "reference_coverage": matched / len(reference),
        "matched_f1": 2 * matched / (len(reference) + len(generated)),
        "edit_similarity": edit_similarity,
        "longest_block_ratio": longest / len(reference),
    }


def model_name_from_frame(frame: pd.DataFrame, path: Path) -> str:
    if "model" in frame.columns:
        values = [clean_scalar(value) for value in frame["model"]]
        values = [value for value in values if value]
        if values:
            return Counter(values).most_common(1)[0][0]
    return re.sub(r"_contamination_results$", "", path.stem, flags=re.I)


def validate_result_frame(
    frame: pd.DataFrame,
    sample_lookup: pd.DataFrame,
    path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    required = {
        "sample_id",
        "prompt_type",
        "reference_suffix",
        "generated_text",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path.name}: missing required columns {missing}")

    data = frame.copy()
    data["sample_id"] = data["sample_id"].map(clean_scalar)
    data["prompt_type"] = data["prompt_type"].map(clean_scalar).str.lower()
    data = data.merge(
        sample_lookup,
        how="left",
        on="sample_id",
        validate="many_to_one",
        suffixes=("", "_sample"),
    )
    model_name = model_name_from_frame(frame, path)
    data["model_display"] = model_name
    data["result_file"] = path.name

    duplicate_pairs = int(data.duplicated(["sample_id", "prompt_type"]).sum())
    missing_samples = int(data["group_sample"].isna().sum()) if "group_sample" in data else 0
    invalid_prompts = sorted(set(data["prompt_type"]).difference(PROMPT_ORDER))
    blank_outputs = int(data["generated_text"].map(clean_scalar).eq("").sum())
    expected_pairs = len(sample_lookup) * len(PROMPT_ORDER)
    validation = {
        "result_file": path.name,
        "model": model_name,
        "rows": len(data),
        "expected_rows": expected_pairs,
        "unique_samples": int(data["sample_id"].nunique()),
        "duplicate_sample_prompt_pairs": duplicate_pairs,
        "missing_sample_metadata": missing_samples,
        "invalid_prompt_types": ", ".join(invalid_prompts),
        "blank_generated_outputs": blank_outputs,
        "reference_suffix_mismatches": None,
        "status": (
            "PASS"
            if len(data) == expected_pairs
            and data["sample_id"].nunique() == len(sample_lookup)
            and duplicate_pairs == 0
            and missing_samples == 0
            and not invalid_prompts
            and blank_outputs == 0
            else "CHECK"
        ),
    }
    return data, validation


def summarize(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(keys, dropna=False, sort=False)
    result = grouped.agg(
        outputs=("sample_id", "size"),
        blank_outputs=("blank_output", "sum"),
        unique_samples=("sample_id", "nunique"),
        exact_reproductions=("exact_reproduction", "sum"),
        contained_reproductions=("reference_contained", "sum"),
        near_verbatim_outputs=("near_verbatim", "sum"),
        high_overlap_unique_samples=("high_overlap_sample", "nunique"),
        mean_reference_coverage=("reference_coverage", "mean"),
        median_reference_coverage=("reference_coverage", "median"),
        p95_reference_coverage=("reference_coverage", lambda values: values.quantile(0.95)),
        maximum_reference_coverage=("reference_coverage", "max"),
        mean_edit_similarity=("edit_similarity", "mean"),
        maximum_edit_similarity=("edit_similarity", "max"),
    ).reset_index()
    result["usable_outputs"] = result["outputs"] - result["blank_outputs"]
    result["exact_rate"] = result["exact_reproductions"] / result["usable_outputs"]
    result["near_verbatim_rate"] = result["near_verbatim_outputs"] / result["usable_outputs"]
    return result


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze continuation-based contamination probes.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("\u6a21\u578b\u6c61\u67d3\u6d4b\u8bd5"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("\u6a21\u578b\u6c61\u67d3\u6d4b\u8bd5_\u7edf\u8ba1\u5206\u6790") / "\u7ed3\u679c\u6587\u4ef6",
    )
    parser.add_argument("--threshold", type=float, default=0.90)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_path = input_dir / "contamination_samples_final_\u6821\u5bf9.xlsx"
    sample_frame = pd.read_excel(sample_path, sheet_name="selected_100")
    sample_frame["sample_id"] = sample_frame["sample_id"].map(clean_scalar)
    if len(sample_frame) != 100 or sample_frame["sample_id"].nunique() != 100:
        raise ValueError("selected_100 must contain 100 unique sample_id values")

    sample_columns = [
        "sample_id",
        "group",
        "source_name",
        "volume_name",
        "headword",
        "core_visible_prefix",
        "core_reference_suffix",
        "core_prefix_length",
        "core_suffix_length",
        "manual_verified",
    ]
    sample_lookup = sample_frame[sample_columns].copy()
    sample_lookup = sample_lookup.rename(columns={column: f"{column}_sample" for column in sample_columns if column != "sample_id"})

    converter = OpenCC("t2s")
    all_frames: list[pd.DataFrame] = []
    validations: list[dict[str, object]] = []
    result_paths = sorted((input_dir / "results").glob("*.xlsx"))
    if not result_paths:
        raise FileNotFoundError(f"No result workbooks found under {input_dir / 'results'}")

    for result_path in result_paths:
        raw = pd.read_excel(result_path)
        data, validation = validate_result_frame(raw, sample_lookup, result_path)
        validations.append(validation)
        data["group"] = data["group_sample"]
        data["group_label"] = data["group"].map(GROUP_LABELS)
        data["source_name"] = data["source_name_sample"]
        data["headword"] = data["headword_sample"].map(clean_scalar)
        data["visible_prefix"] = data["core_visible_prefix_sample"].map(clean_scalar)
        data["reference_suffix_master"] = data["core_reference_suffix_sample"].map(clean_scalar)
        data["reference_suffix_result"] = data["reference_suffix"].map(clean_scalar)
        data["generated_text"] = data["generated_text"].map(clean_scalar)
        data["blank_output"] = data["generated_text"].eq("")
        data["normalized_reference"] = data["reference_suffix_master"].map(lambda value: normalize_text(value, converter))
        data["normalized_generated"] = data["generated_text"].map(lambda value: normalize_text(value, converter))
        data["reference_field_matches_master"] = [
            normalize_text(left, converter) == normalize_text(right, converter)
            for left, right in zip(data["reference_suffix_result"], data["reference_suffix_master"])
        ]
        validations[-1]["reference_suffix_mismatches"] = int(
            (~data["reference_field_matches_master"]).sum()
        )
        if validations[-1]["reference_suffix_mismatches"]:
            validations[-1]["status"] = "CHECK"

        metric_rows = [
            similarity_metrics(reference, generated)
            for reference, generated in zip(data["normalized_reference"], data["normalized_generated"])
        ]
        metrics = pd.DataFrame(metric_rows, index=data.index)
        data = pd.concat([data, metrics], axis=1)
        data["near_verbatim"] = data["reference_coverage"].ge(args.threshold)
        data["risk_band"] = pd.cut(
            data["reference_coverage"],
            bins=[-0.001, 0.50, 0.70, args.threshold, 1.001],
            labels=["low", "limited", "moderate", "high"],
            right=False,
        ).astype(str)
        data["high_overlap_sample"] = data["sample_id"].where(data["near_verbatim"])
        all_frames.append(data)

    detail = pd.concat(all_frames, ignore_index=True)
    validation_frame = pd.DataFrame(validations)

    source_summary = summarize(detail, ["group", "group_label"])
    model_source_summary = summarize(detail, ["model_display", "group", "group_label"])
    model_prompt_summary = summarize(detail, ["model_display", "prompt_type"])
    source_prompt_summary = summarize(detail, ["group", "group_label", "prompt_type"])

    cases = detail.loc[detail["near_verbatim"]].copy()
    cases = cases.sort_values(
        ["group", "reference_coverage", "exact_reproduction", "model_display"],
        ascending=[True, False, False, True],
    )

    sensitivity_rows: list[dict[str, object]] = []
    usable_detail = detail.loc[~detail["blank_output"]].copy()
    for threshold in [0.80, 0.85, 0.90, 0.95, 1.00]:
        for (group, group_label), subset in usable_detail.groupby(
            ["group", "group_label"], sort=False
        ):
            matches = subset["reference_coverage"].ge(threshold)
            sensitivity_rows.append(
                {
                    "group": group,
                    "group_label": group_label,
                    "threshold": threshold,
                    "usable_outputs": len(subset),
                    "matching_outputs": int(matches.sum()),
                    "matching_rate": float(matches.mean()),
                    "unique_matching_samples": int(
                        subset.loc[matches, "sample_id"].nunique()
                    ),
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)

    detail_columns = [
        "model_display",
        "result_file",
        "sample_id",
        "group",
        "group_label",
        "source_name",
        "headword",
        "prompt_type",
        "visible_prefix",
        "reference_suffix_master",
        "generated_text",
        "exact_reproduction",
        "reference_contained",
        "near_verbatim",
        "risk_band",
        "sequence_ratio",
        "reference_coverage",
        "matched_f1",
        "edit_similarity",
        "longest_block_ratio",
        "reference_field_matches_master",
    ]

    save_csv(validation_frame, output_dir / "00_\u7ed3\u679c\u6587\u4ef6\u5b8c\u6574\u6027\u5ba1\u8ba1.csv")
    save_csv(source_summary, output_dir / "01_\u6765\u6e90\u7ec4\u603b\u4f53\u7edf\u8ba1.csv")
    save_csv(model_source_summary, output_dir / "02_\u6a21\u578b_\u6765\u6e90\u7ec4\u7edf\u8ba1.csv")
    save_csv(model_prompt_summary, output_dir / "03_\u6a21\u578b_\u63d0\u793a\u7c7b\u578b\u7edf\u8ba1.csv")
    save_csv(source_prompt_summary, output_dir / "04_\u6765\u6e90\u7ec4_\u63d0\u793a\u7c7b\u578b\u7edf\u8ba1.csv")
    save_csv(detail[detail_columns], output_dir / "05_\u9010\u6761\u7edf\u8ba1\u660e\u7ec6.csv")
    save_csv(cases[detail_columns], output_dir / "06_\u9ad8\u91cd\u5408\u7eed\u5199\u6848\u4f8b.csv")
    save_csv(sensitivity, output_dir / "07_\u9608\u503c\u654f\u611f\u6027\u5206\u6790.csv")

    metadata = {
        "input_directory": str(input_dir),
        "sample_file": str(sample_path),
        "result_files": [path.name for path in result_paths],
        "models": int(detail["model_display"].nunique()),
        "samples": int(sample_frame["sample_id"].nunique()),
        "outputs": int(len(detail)),
        "prompt_types": sorted(detail["prompt_type"].unique().tolist()),
        "normalization": "Unicode NFKC, traditional-to-simplified conversion, lowercase, retain Unicode letters and numbers",
        "near_verbatim_rule": f"reference character coverage >= {args.threshold:.2f}",
        "near_verbatim_includes_exact": True,
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print("\nSource summary")
    print(source_summary.to_string(index=False))
    print("\nValidation")
    print(validation_frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
