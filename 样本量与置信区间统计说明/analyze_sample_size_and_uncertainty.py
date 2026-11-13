#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate read-only sample-size, sampling-audit, and CI summaries.

This script is intentionally conservative: it does not alter evaluation
results, source datasets, contamination tests, or audit outputs.  It only reads
existing derived files and writes reviewer-facing statistical supplements into
the current directory.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / "结果文件"

TASK_FILES = {
    "部首识别": ROOT / "eval_data" / "excel" / "部首_eval.xlsx",
    "拼音/反切注音": ROOT / "eval_data" / "excel" / "pinyin&fanqie_eval_labeled.xlsx",
    "词性识别": ROOT / "eval_data" / "excel" / "词性_eval.xlsx",
    "释义生成/语境理解": ROOT / "eval_data" / "excel" / "释义&语境_eval.xlsx",
}

INTEGRATED_METRICS = ROOT / "补充评测_整合结果" / "原始与新增模型指标整合_修正版.xlsx"
SUPPLEMENTARY_BASELINES = (
    ROOT
    / "qwen3_8b_rag_formal_server"
    / "evaluation_results"
    / "supplementary_baseline_metrics.csv"
)
CROSS_TASK_AUDIT = ROOT / "audit_output_cross_task" / "跨任务字词与内容重叠审计.xlsx"
CONTAMINATION_DIR = ROOT / "模型污染测试"


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def qbin(series: pd.Series, labels: list[str]) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    try:
        return pd.qcut(values, q=len(labels), labels=labels, duplicates="drop")
    except ValueError:
        return pd.Series(["未分箱"] * len(series), index=series.index)


def sense_count(text: Any) -> int:
    value = clean_text(text)
    matches = re.findall(r"(?:^|\s)(\d+)[.．、]", value)
    return max(len(matches), 1 if value else 0)


def sense_bin(count: int) -> str:
    if count <= 1:
        return "单义或未显式编号"
    if count <= 3:
        return "2-3义项"
    if count <= 8:
        return "4-8义项"
    return "9义项及以上"


def wilson_ci(correct: float, total: float, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (float("nan"), float("nan"))
    p = correct / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def bootstrap_ci(values: list[float], iterations: int = 10000) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    # Deterministic bootstrap without importing heavy dependencies.
    import random

    rng = random.Random(20260801)
    n = len(values)
    means = []
    for _ in range(iterations):
        sample_sum = 0.0
        for _ in range(n):
            sample_sum += values[rng.randrange(n)]
        means.append(sample_sum / n)
    means.sort()
    return means[int(0.025 * iterations)], means[int(0.975 * iterations) - 1]


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def count_rows(label: str, df: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    counts = Counter(clean_text(value) or "空值" for value in df[column])
    total = len(df)
    return [
        {
            "任务": label,
            "分布变量": column,
            "类别": key,
            "样本数": value,
            "比例": value / total if total else 0,
        }
        for key, value in counts.most_common()
    ]


def build_sample_summaries() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    distributions: list[dict[str, Any]] = []

    radical = pd.read_excel(TASK_FILES["部首识别"])
    summaries.append(
        {
            "任务": "部首识别",
            "样本数": len(radical),
            "唯一字词数": radical["字词"].nunique(),
            "主要分层说明": "按检字部首覆盖不同部首；报告部首频次分布与低频部首覆盖。",
        }
    )
    distributions.extend(count_rows("部首识别", radical, "部首"))

    pinyin_fanqie = pd.read_excel(TASK_FILES["拼音/反切注音"])
    summaries.extend(
        [
            {
                "任务": "拼音注音",
                "样本数": len(pinyin_fanqie),
                "唯一字词数": pinyin_fanqie["字词"].nunique(),
                "主要分层说明": "与反切使用同一批测试字，保证音读任务可比；保留污染构造分组。",
            },
            {
                "任务": "反切注音",
                "样本数": len(pinyin_fanqie),
                "唯一字词数": pinyin_fanqie["字词"].nunique(),
                "主要分层说明": "与拼音使用同一批测试字，保证音读任务可比；保留污染构造分组。",
            },
        ]
    )
    distributions.extend(count_rows("拼音注音", pinyin_fanqie, "contamination_group"))
    distributions.extend(count_rows("反切注音", pinyin_fanqie, "contamination_group"))

    pos = pd.read_excel(TASK_FILES["词性识别"])
    pos["例句长度分箱"] = qbin(pos["例句"].map(lambda x: len(clean_text(x))), ["短", "中短", "中长", "长"])
    summaries.append(
        {
            "任务": "词性识别",
            "样本数": len(pos),
            "唯一字词数": pos["字词"].nunique(),
            "主要分层说明": "按词性、书名、章节和例句长度描述样本分布。",
        }
    )
    for col in ["词性", "书名", "例句长度分箱"]:
        distributions.extend(count_rows("词性识别", pos, col))

    semantic = pd.read_excel(TASK_FILES["释义生成/语境理解"])
    semantic["例句长度分箱"] = qbin(semantic["例句"].map(lambda x: len(clean_text(x))), ["短", "中短", "中长", "长"])
    semantic["释义长度分箱"] = qbin(semantic["所有含义"].map(lambda x: len(clean_text(x))), ["短", "中短", "中长", "长"])
    semantic["义项数"] = semantic["所有含义"].map(sense_count)
    semantic["义项数分箱"] = semantic["义项数"].map(sense_bin)
    for task in ["释义生成", "语境理解"]:
        summaries.append(
            {
                "任务": task,
                "样本数": len(semantic),
                "唯一字词数": semantic["字词"].nunique(),
                "主要分层说明": "同一批词头分别用于完整释义生成和语境义辨析；按书名、词性、例句长度、释义长度和义项数报告覆盖。",
            }
        )
        for col in ["词性", "书名", "例句长度分箱", "释义长度分箱", "义项数分箱"]:
            distributions.extend(count_rows(task, semantic, col))

    return summaries, distributions


def build_accuracy_ci() -> list[dict[str, Any]]:
    if not INTEGRATED_METRICS.exists():
        return []
    df = pd.read_excel(INTEGRATED_METRICS, sheet_name="Accuracy整合")
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        n = float(row.get("样本数量", 200) or 200)
        acc = float(row["准确率（%）"]) / 100.0
        correct = round(acc * n)
        lo, hi = wilson_ci(correct, n)
        rows.append(
            {
                "模型名": row["模型名"],
                "任务名": row["任务名"],
                "评测模式": row["评测模式"],
                "样本数量": int(n),
                "准确率(%)": acc * 100,
                "正确数_由准确率反推": correct,
                "Wilson95CI下限(%)": lo * 100,
                "Wilson95CI上限(%)": hi * 100,
                "数据来源": row.get("数据来源", ""),
                "说明": "由既有准确率和样本数计算置信区间，未重算模型输出。",
            }
        )
    return rows


def build_semantic_uncertainty_notes() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if INTEGRATED_METRICS.exists():
        rouge = pd.read_excel(INTEGRATED_METRICS, sheet_name="ROUGE整合")
        sim = pd.read_excel(INTEGRATED_METRICS, sheet_name="语义相似度整合")
        for _, row in rouge.iterrows():
            rows.append(
                {
                    "模型名": row["模型名"],
                    "任务名": row["任务名"],
                    "评测模式": row["评测模式"],
                    "指标": "ROUGE-Avg(%)",
                    "均值": row["ROUGE-Avg(%)"],
                    "样本数量": row.get("样本数量", 200),
                    "95%CI状态": "聚合均值保留；原始逐条ROUGE未在该汇总表中保存，不能仅由均值反推出bootstrap CI。",
                    "建议论文写法": "报告均值，并说明补充统计对准确率给出Wilson CI；生成类任务若使用逐条结果文件，可按同一脚本扩展bootstrap CI。",
                }
            )
        for _, row in sim.iterrows():
            rows.append(
                {
                    "模型名": row["模型名"],
                    "任务名": row["任务名"],
                    "评测模式": row["评测模式"],
                    "指标": "平均语义相似度(%)",
                    "均值": row["平均语义相似度(%)"],
                    "样本数量": row.get("样本数量", 200),
                    "95%CI状态": "聚合均值保留；原始逐条SBERT分数未在该汇总表中保存，不能仅由均值反推出bootstrap CI。",
                    "建议论文写法": "将不确定性说明与人工专家评测相关性结果并列报告，避免把单一自动指标解释为绝对质量。",
                }
            )

    if SUPPLEMENTARY_BASELINES.exists():
        base = pd.read_csv(SUPPLEMENTARY_BASELINES)
        for _, row in base.iterrows():
            for metric in ["rouge_avg_percent", "sbert_percent"]:
                if pd.notna(row.get(metric)):
                    rows.append(
                        {
                            "模型名": row["id"],
                            "任务名": row["task"],
                            "评测模式": "supplementary",
                            "指标": metric,
                            "均值": row[metric],
                            "样本数量": row.get("items", 200),
                            "95%CI状态": "补充基线汇总均值已复用；若需逐条CI，可从对应per_item派生文件重建。",
                            "建议论文写法": "作为基线补充结果报告，不改变原论文模型评测结果。",
                        }
                    )
    return rows


def build_audit_summaries() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    audit_rows: list[dict[str, Any]] = []
    contamination_rows: list[dict[str, Any]] = []
    if CROSS_TASK_AUDIT.exists():
        same = pd.read_excel(CROSS_TASK_AUDIT, sheet_name="同任务关键审计")
        for _, row in same.iterrows():
            audit_rows.append(
                {
                    "审计类型": "同任务训练-测试重叠",
                    "对象": row["任务"],
                    "测试唯一字词数": row["测试唯一字词数"],
                    "训练唯一字词数": row["训练唯一字词数"],
                    "重叠唯一字词数": row["同任务重叠唯一字词数"],
                    "关键结论": row["审计判断"],
                }
            )
        for sheet in ["拼音反切汇总", "词性语境汇总", "释义语境汇总"]:
            detail = pd.read_excel(CROSS_TASK_AUDIT, sheet_name=sheet)
            for _, row in detail.iterrows():
                audit_rows.append(
                    {
                        "审计类型": sheet,
                        "对象": row["审计组合"],
                        "测试唯一字词数": row["测试唯一字词数"],
                        "训练唯一字词数": row["训练唯一字词数"],
                        "重叠唯一字词数": row["重叠唯一字词数"],
                        "关键结论": row["说明"],
                    }
                )

    sample_book = CONTAMINATION_DIR / "contamination_samples_final_校对.xlsx"
    if sample_book.exists():
        summary = pd.read_excel(sample_book, sheet_name="Summary")
        for _, row in summary.iterrows():
            contamination_rows.append(
                {
                    "来源": "污染测试样本构造清单",
                    "项目": row.get("项目", ""),
                    "结果": row.get("结果", ""),
                }
            )
    results_dir = CONTAMINATION_DIR / "results"
    if results_dir.exists():
        for path in sorted(results_dir.glob("*_contamination_results.xlsx")):
            try:
                df = pd.read_excel(path)
                rows = len(df)
                prompt_types = ",".join(map(str, sorted(df.get("prompt_type", pd.Series(dtype=str)).dropna().unique())))
            except Exception as exc:  # pragma: no cover - only for damaged files
                rows = ""
                prompt_types = f"读取失败：{exc}"
            contamination_rows.append(
                {
                    "来源": "模型污染测试结果",
                    "项目": path.stem,
                    "结果": f"rows={rows}; prompt_types={prompt_types}",
                }
            )
    return audit_rows, contamination_rows


def build_response_text(summary: dict[str, Any]) -> str:
    return f"""# 样本量、抽样策略与置信区间补充说明

## 对审稿意见的处理方式

本次不改变既有评测样本，也不重新运行模型。补充工作只做统计说明和不确定性报告：第一，说明每项任务固定使用200条测试样本，并列出样本覆盖的词头、词性、书名、语境长度、释义长度和义项数分布；第二，引用已完成的数据审计和模型污染测试，说明测试集与同任务训练集无字词重叠，释义生成与语境理解之间也未发现字词层面的训练-测试重叠；第三，对准确率类任务基于既有准确率和样本数补充Wilson 95%置信区间。这样回应的是“样本量较小但统计透明度不足”的问题，而不是更换实验口径。

## 为什么这样做是合理的

1. 重新扩大到400-500条会改变评测口径，并要求18个模型及新增模型全部重新推理，成本高且会破坏与原论文结果的可比性。
2. 当前200条不是随意抽取后直接使用，而是已经经过数据来源追溯、训练-测试重叠审计和模型污染测试；因此更合理的修改是把这些证据写进论文，并补充分布和置信区间。
3. 对准确率任务，Wilson区间适合n=200的二项比例估计，比简单正态近似更稳健。
4. 对ROUGE和Sentence-BERT，已有正式表格保存的是聚合均值；不能只凭均值反推出bootstrap置信区间。论文中可报告均值、样本数，并把人工专家评测的一致性/相关性作为生成类任务的质量验证；如需要逐条bootstrap，应从逐条结果文件重建逐条指标后再做。

## 可放入论文的方法表述

To address concerns regarding the relatively small evaluation set, we retained the original fixed test sets of 200 instances per task to preserve comparability across all evaluated models. We added a post-hoc coverage audit of the evaluation samples, reporting distributions over headwords, part-of-speech categories, source classics, contextual length, definition length, and number of senses where applicable. In addition, we report Wilson 95% confidence intervals for accuracy-based tasks. The train-test audit showed no same-task headword overlap between test and training data for all six tasks, and no headword-level overlap between the definition-generation and contextual-sense-disambiguation test sets and the corresponding training sets. Therefore, the added analysis improves transparency about sampling coverage and statistical uncertainty without changing the original evaluation protocol.

## 可放入回复信的表述

We thank the reviewer for pointing out that the sample size and sampling strategy should be better justified. We did not expand the test set after revision because doing so would require re-running all models and would make the revised results less directly comparable with the original experiments. Instead, we added a dedicated sampling and uncertainty analysis based on the existing fixed test sets. Specifically, we report the distribution of the 200 evaluation instances for each task, summarize the train-test overlap and contamination audits, and provide Wilson 95% confidence intervals for accuracy-based metrics. We also clarify that conclusions are made within the scope of the fixed evaluation samples.

## 当前统计产物

- 样本任务摘要：{summary["sample_summary_rows"]}行
- 样本分布明细：{summary["distribution_rows"]}行
- 准确率Wilson区间：{summary["accuracy_ci_rows"]}行
- 自动指标不确定性说明：{summary["semantic_note_rows"]}行
- 审计摘要：{summary["audit_rows"]}行
- 污染测试摘要：{summary["contamination_rows"]}行
"""


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sample_summary, distribution = build_sample_summaries()
    accuracy_ci = build_accuracy_ci()
    semantic_notes = build_semantic_uncertainty_notes()
    audit_summary, contamination_summary = build_audit_summaries()

    write_csv(OUT_DIR / "样本任务摘要.csv", sample_summary)
    write_csv(OUT_DIR / "样本分布明细.csv", distribution)
    write_csv(OUT_DIR / "准确率_Wilson95CI.csv", accuracy_ci)
    write_csv(OUT_DIR / "生成类指标不确定性说明.csv", semantic_notes)
    write_csv(OUT_DIR / "数据审计摘要.csv", audit_summary)
    write_csv(OUT_DIR / "模型污染测试摘要.csv", contamination_summary)

    summary = {
        "output_dir": str(OUT_DIR),
        "source_root": str(ROOT),
        "sample_summary_rows": len(sample_summary),
        "distribution_rows": len(distribution),
        "accuracy_ci_rows": len(accuracy_ci),
        "semantic_note_rows": len(semantic_notes),
        "audit_rows": len(audit_summary),
        "contamination_rows": len(contamination_summary),
        "methods": {
            "accuracy_ci": "Wilson score interval, 95%, computed from existing aggregate accuracy and n.",
            "generation_metrics": "Existing aggregate ROUGE/SBERT means are preserved; no inverse bootstrap from aggregate-only means.",
            "sampling_audit": "Post-hoc coverage audit over the fixed 200-item evaluation sets.",
        },
        "source_files_read_only": [
            str(path)
            for path in [
                *TASK_FILES.values(),
                INTEGRATED_METRICS,
                SUPPLEMENTARY_BASELINES,
                CROSS_TASK_AUDIT,
                CONTAMINATION_DIR,
            ]
        ],
    }
    (OUT_DIR / "样本量与置信区间统计_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "论文与回复信可用文字.md").write_text(
        build_response_text(summary),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
