#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Read-only OCR/data-processing quality-control statistics.

The script summarizes existing process files without changing source data.  It
does not invent page-level OCR accuracy; instead, unsupported metrics are marked
as unavailable from the archived files.
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
PROCESS_ROOT = ROOT / "原始数据处理结果及过程"
OUT_DIR = Path(__file__).resolve().parent / "结果文件"

VOLUMES = ["论语卷", "孟子卷", "孝经卷", "毛诗序"]

OCR_TXT_DIR = PROCESS_ROOT / "十三经辞典_OCR_txt"
RAW_CLEAN_DIR = PROCESS_ROOT / "十三经辞典_clean_excel"
PROOFREAD_DIR = PROCESS_ROOT / "十三经辞典_clean_excel" / "校对版" / "校对完整数据"
MERGED_DIR = PROCESS_ROOT / "十三经辞典_clean_excel" / "校对版" / "合并数据"
EVAL_DIR = ROOT / "eval_data" / "excel"

OCR_ERROR_LOG = PROCESS_ROOT / "OCR错误情况整理.xlsx"
OLD_STATS = PROCESS_ROOT / "data_process_statistics.xlsx"


def clean(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def nonempty(series: pd.Series) -> int:
    return int(series.map(lambda x: bool(clean(x))).sum())


def pct(num: float, den: float) -> float:
    return num / den if den else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def cjk_count(text: str) -> int:
    return len(re.findall(r"[\u3400-\u9fff\uf900-\ufaff]", text))


def txt_stats() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(OCR_TXT_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        lines = [line for line in text.splitlines() if line.strip()]
        rows.append(
            {
                "阶段": "Qwen2-VL OCR输出文本",
                "文件": str(path.relative_to(ROOT)),
                "卷册": path.stem,
                "总字符数": len(text),
                "汉字数": cjk_count(text),
                "非空行数": len(lines),
                "说明": "OCR文本规模统计；不是逐页OCR准确率。",
            }
        )
    return rows


def workbook_field_stats() -> list[dict[str, Any]]:
    rows = []
    fields = ["字词", "拼音", "反切音", "韵", "释义", "词性", "例句", "简体字", "部首"]
    for volume in VOLUMES:
        candidates = [
            ("程序清洗后", RAW_CLEAN_DIR / f"{volume}_clean.xlsx"),
            ("人工校对完整数据", PROOFREAD_DIR / f"{volume}.xlsx"),
        ]
        for stage, path in candidates:
            if not path.exists():
                continue
            df = pd.read_excel(path)
            for field in fields:
                if field in df.columns:
                    n = nonempty(df[field])
                    rows.append(
                        {
                            "阶段": stage,
                            "卷册": volume,
                            "字段": field,
                            "总行数": len(df),
                            "非空行数": n,
                            "缺失行数": len(df) - n,
                            "非空率": pct(n, len(df)),
                            "来源文件": str(path.relative_to(ROOT)),
                        }
                    )
            rows.append(
                {
                    "阶段": stage,
                    "卷册": volume,
                    "字段": "__整体__",
                    "总行数": len(df),
                    "非空行数": "",
                    "缺失行数": "",
                    "非空率": "",
                    "来源文件": str(path.relative_to(ROOT)),
                    "唯一字词数": df["字词"].dropna().astype(str).str.strip().replace("", pd.NA).dropna().nunique()
                    if "字词" in df.columns
                    else "",
                    "单字行数": int(df["字词"].map(lambda x: len(clean(x)) == 1).sum()) if "字词" in df.columns else "",
                    "多字词行数": int(df["字词"].map(lambda x: len(clean(x)) > 1).sum()) if "字词" in df.columns else "",
                }
            )
    return rows


def before_after_summary(field_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(field_rows)
    rows = []
    for volume in VOLUMES:
        raw = df[(df["卷册"] == volume) & (df["阶段"] == "程序清洗后") & (df["字段"] == "__整体__")]
        proof = df[(df["卷册"] == volume) & (df["阶段"] == "人工校对完整数据") & (df["字段"] == "__整体__")]
        if raw.empty or proof.empty:
            continue
        raw_row = raw.iloc[0]
        proof_row = proof.iloc[0]
        rows.append(
            {
                "卷册": volume,
                "清洗后行数": int(raw_row["总行数"]),
                "校对后行数": int(proof_row["总行数"]),
                "行数变化": int(proof_row["总行数"]) - int(raw_row["总行数"]),
                "清洗后唯一字词数": raw_row.get("唯一字词数", ""),
                "校对后唯一字词数": proof_row.get("唯一字词数", ""),
                "说明": "行数变化反映清洗、拆分、补字段、校对后的结构化数据变化，不等同于OCR错误数。",
            }
        )
    return rows


def merged_stats() -> list[dict[str, Any]]:
    files = [
        "merge_拼音&反切.xlsx",
        "merge_字词&反切.xlsx",
        "merged_部首.xlsx",
        "merged_词性&例句.xlsx",
        "merged_词性&例句&释义.xlsx",
        "all_字词及反切音.xlsx",
        "all_字词及释义.xlsx",
    ]
    rows = []
    for name in files:
        path = MERGED_DIR / name
        if not path.exists():
            continue
        df = pd.read_excel(path)
        fields = [c for c in ["字词", "word", "拼音", "反切音", "guangyun", "部首", "词性", "释义", "例句", "所有含义"] if c in df.columns]
        row = {
            "文件": str(path.relative_to(ROOT)),
            "行数": len(df),
            "列数": len(df.columns),
            "列名": ", ".join(map(str, df.columns)),
        }
        for field in fields:
            row[f"{field}_非空"] = nonempty(df[field])
            row[f"{field}_非空率"] = pct(nonempty(df[field]), len(df))
        rows.append(row)
    return rows


def eval_qc_stats() -> list[dict[str, Any]]:
    specs = [
        ("部首识别", EVAL_DIR / "部首_eval.xlsx", ["字词", "部首"], []),
        ("拼音注音", EVAL_DIR / "pinyin&fanqie_eval_labeled.xlsx", ["字词", "拼音"], []),
        ("反切注音", EVAL_DIR / "pinyin&fanqie_eval_labeled.xlsx", ["字词", "反切音"], []),
        ("词性识别", EVAL_DIR / "词性_eval.xlsx", ["字词", "词性", "释义", "例句", "书名", "章节"], ["包含例句的段落"]),
        ("释义生成", EVAL_DIR / "释义&语境_eval.xlsx", ["字词", "所有含义"], []),
        ("语境理解", EVAL_DIR / "释义&语境_eval.xlsx", ["字词", "词性", "释义", "例句", "书名", "章节"], ["包含例句的段落"]),
    ]
    rows = []
    for task, path, required_fields, optional_fields in specs:
        df = pd.read_excel(path)
        missing_fields = [f for f in required_fields if f not in df.columns]
        field_status = []
        total_missing = 0
        for field in required_fields:
            if field not in df.columns:
                field_status.append(f"{field}:列缺失")
                total_missing += len(df)
            else:
                m = len(df) - nonempty(df[field])
                total_missing += m
                field_status.append(f"{field}:缺失{m}")
        optional_status = []
        for field in optional_fields:
            if field not in df.columns:
                optional_status.append(f"{field}:列缺失")
            else:
                optional_status.append(f"{field}:缺失{len(df) - nonempty(df[field])}")
        rows.append(
            {
                "任务": task,
                "测试样本数": len(df),
                "唯一字词数": df["字词"].dropna().astype(str).str.strip().replace("", pd.NA).dropna().nunique()
                if "字词" in df.columns
                else "",
                "必需字段": ", ".join(required_fields),
                "缺失字段": ", ".join(missing_fields) if missing_fields else "无",
                "必需字段缺失单元格数": total_missing,
                "必需字段完整性": "通过" if total_missing == 0 and not missing_fields else "需复核",
                "来源文件": str(path.relative_to(ROOT)),
                "字段明细": "; ".join(field_status),
                "可选字段明细": "; ".join(optional_status),
            }
        )
    return rows


def ocr_error_examples() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not OCR_ERROR_LOG.exists():
        return [], []
    df = pd.read_excel(OCR_ERROR_LOG)
    rows = []
    for i, row in df.iterrows():
        rows.append(
            {
                "序号": i + 1,
                "错误类型": row.get("错误类型", ""),
                "原文本内容": row.get("原文本内容", ""),
                "OCR识别文本": row.get("OCR识别文本", ""),
                "是否可能影响下游": impact_note(clean(row.get("错误类型", ""))),
            }
        )
    counts = [
        {"错误类型": key, "记录条数": val, "说明": "来自既有错误案例整理表；不是全量错误频数。"}
        for key, val in Counter(clean(x) for x in df["错误类型"]).items()
    ]
    return rows, counts


def impact_note(error_type: str) -> str:
    if "字符" in error_type or "音" in error_type:
        return "高：可能影响字头、部首、拼音或反切标签。"
    if "遗漏" in error_type or "错位" in error_type:
        return "高：可能影响释义、例句与词条对应关系。"
    if "格式" in error_type or "重复" in error_type:
        return "中：主要影响条目切分和字段抽取。"
    return "需人工判断。"


def workflow_steps() -> list[dict[str, Any]]:
    return [
        {
            "步骤": 1,
            "阶段": "PDF/扫描页转图像",
            "代码或文件": "pdfToimage.py",
            "处理内容": "将扫描页转为图像，供视觉语言模型OCR。",
            "质量控制点": "保留页图像到文本的可追溯链路；当前归档未保存逐页CER统计。",
        },
        {
            "步骤": 2,
            "阶段": "OCR识别",
            "代码或文件": "Qwen2_VL_OCR.py",
            "处理内容": "使用Qwen2-VL-72B-Instruct-GPTQ-Int4识别正文，提示中要求忽略页眉页脚、按左右栏顺序输出、保持繁体。",
            "质量控制点": "减少版面顺序错误；后续人工校对与错误案例整理用于修正OCR问题。",
        },
        {
            "步骤": 3,
            "阶段": "OCR文本转结构化Excel",
            "代码或文件": "txtToExcel.py",
            "处理内容": "用正则从txt提取字词、拼音和释义初始字段。",
            "质量控制点": "将非结构化OCR文本转为可校对表格。",
        },
        {
            "步骤": 4,
            "阶段": "词典条目清洗与义项拆分",
            "代码或文件": "CiDianClean.py, ssjDaCiDianClean.py, data_clean_V2.py",
            "处理内容": "拆分义项，抽取词性、释义、例句、反切、韵部，清理编号、括号、异常字符和字段错位。",
            "质量控制点": "面向反切、词性、释义、例句等下游字段做结构化校正。",
        },
        {
            "步骤": 5,
            "阶段": "人工校对与补充字段",
            "代码或文件": "校对版/校对完整数据/*.xlsx, 校对版/部首数据/*.xlsx",
            "处理内容": "形成四册校对完整数据，并补充简体字、部首等字段。",
            "质量控制点": "最终进入训练和评测的数据来自校对版，而非直接OCR文本。",
        },
        {
            "步骤": 6,
            "阶段": "合并任务资源",
            "代码或文件": "data_extract.py, bushou_data_extract.py, cixing_extract.py, shiyi&yujing_eval_extract.py",
            "处理内容": "生成部首、拼音/反切、词性/例句、释义/语境等任务资源。",
            "质量控制点": "按任务字段要求生成独立数据表。",
        },
        {
            "步骤": 7,
            "阶段": "评测集抽取与训练集过滤",
            "代码或文件": "eval_data_extract/*.xlsx, 微调数据/*过滤测试数据.xlsx, cross_task_overlap_audit.py",
            "处理内容": "抽取固定200条评测样本，并从微调训练数据中过滤测试词头，进行跨任务重叠审计。",
            "质量控制点": "测试集必需字段完整，训练-测试重叠已审计。",
        },
    ]


def unavailable_metrics() -> list[dict[str, Any]]:
    return [
        {
            "指标": "Character Error Rate (CER)",
            "现有文件能否支持": "否",
            "原因": "归档中没有逐页人工转录金标准或逐字符OCR-人工对照表。",
            "建议写法": "不报告具体CER；说明已进行人工校对、字段完整性检查和测试集100%字段核验，并将逐页CER作为未来数据发布的补充质量审计。",
        },
        {
            "指标": "页面级OCR准确率",
            "现有文件能否支持": "否",
            "原因": "没有按页面抽样记录的人工校对字符数、错误字符数和页码。",
            "建议写法": "报告OCR错误类型案例和校对流程，不声称页面级准确率。",
        },
        {
            "指标": "字段完整率",
            "现有文件能否支持": "是",
            "原因": "校对版Excel、合并数据和评测集均保留结构化字段。",
            "建议写法": "报告校对后字段非空率、最终评测集必需字段完整性。",
        },
        {
            "指标": "评测集100%人工核验状态",
            "现有文件能否支持": "部分支持",
            "原因": "评测集必需字段可做100%完整性与来源一致性统计；若论文要写“人工核验”，仍需作者确认实际校对者和流程。",
            "建议写法": "写为“最终评测集在进入实验前逐项检查必需字段完整性，并基于校对版数据生成”。",
        },
    ]


def build_text(summary: dict[str, Any]) -> str:
    return f"""# OCR准确率与语料质量控制补充说明

## 处理策略

现有归档文件不包含逐页人工转录金标准，因此不能补报严格意义上的 Character Error Rate 或页面级OCR准确率。为避免过度声明，本次补充采用可复核的质量控制证据：梳理OCR到评测集的数据处理流程，统计OCR文本规模、清洗/校对后结构化字段完整率、最终合并数据字段覆盖、六项评测集必需字段完整性，并列出已记录的OCR错误类型案例。

## 可回应审稿人的核心结论

1. 最终实验没有直接使用OCR原始文本，而是使用程序清洗和人工校对后的结构化数据。
2. 已记录的OCR问题包括文本重复、字符混淆、文本遗漏、格式识别不准确、文本识别错位和音标识别错误；这些错误类型均与Reviewer指出的风险相关。
3. 六项任务的200条评测样本均可进行必需字段完整性核验；本次统计表中给出了每个任务的必需字段缺失数。
4. 对反切、拼音、部首等单字符敏感任务，应在论文中强调最终标签来自校对版结构化字段，而不是直接由OCR文本自动生成。
5. 由于缺少逐页人工金标准，不建议在论文中声称OCR字符准确率；更稳妥的说法是“进行了人工校对和字段级质量控制，并补充报告字段完整性和错误类型案例”。

## 可放入论文的方法表述

The OCR outputs were not used directly as evaluation labels. Scanned dictionary pages were first converted into images and recognized with Qwen2-VL using prompts that required preservation of traditional Chinese characters, exclusion of headers and footers, and left-to-right column ordering. The OCR text was then converted into structured spreadsheets, where headword, pinyin, fanqie, rhyme, definition, part-of-speech, and example fields were extracted with rule-based scripts. The resulting spreadsheets were manually proofread and further normalized before task-specific datasets were generated. Because the archived processing files do not contain page-level gold transcriptions, we do not report a page-level character error rate. Instead, we report field-level completeness statistics, documented OCR error categories, and a 100% required-field check for all evaluation items.

## 可放入回复信的表述

We agree that OCR errors may affect downstream tasks, especially radical recognition and fanqie annotation. In the revision, we clarified that the final evaluation data were not directly taken from raw OCR outputs. They were produced through OCR, rule-based structuring, manual proofreading, field normalization, and task-specific extraction. We added a quality-control summary that reports the size of OCR text files, field completeness after proofreading, documented OCR error categories, and a required-field check for every evaluation item. We also explicitly state that page-level CER cannot be reconstructed from the archived files because no page-level gold transcription table was kept; therefore, we avoid making unsupported OCR-accuracy claims.

## 作者需要补充确认的信息

以下内容不能从代码自动恢复，需要你根据实际校对过程补一句说明：

- 校对者人数；
- 校对者专业背景；
- 是单人校对还是双人独立校对；
- 分歧如何处理；
- 是否有第三位专家仲裁；
- 哪些字段是完全人工核验。

如果实际情况是你本人逐项校对，可以写成“由作者依据原书扫描图像与结构化表格逐项校对；对字头、拼音、反切、释义、例句和部首字段进行重点核验”。如果还有导师或同门参与，则把人数和分工写实即可。

## 当前统计产物

- OCR文本规模：{summary["txt_rows"]}行
- 清洗/校对字段完整率：{summary["field_rows"]}行
- 清洗前后规模变化：{summary["before_after_rows"]}行
- 合并数据统计：{summary["merged_rows"]}行
- 评测集字段核验：{summary["eval_rows"]}行
- OCR错误案例：{summary["error_example_rows"]}行
"""


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    txt = txt_stats()
    field = workbook_field_stats()
    before_after = before_after_summary(field)
    merged = merged_stats()
    eval_rows = eval_qc_stats()
    error_examples, error_counts = ocr_error_examples()
    workflow = workflow_steps()
    unavailable = unavailable_metrics()

    write_csv(OUT_DIR / "01_OCR文本规模.csv", txt)
    write_csv(OUT_DIR / "02_清洗校对字段完整率.csv", field)
    write_csv(OUT_DIR / "03_清洗前后规模变化.csv", before_after)
    write_csv(OUT_DIR / "04_最终合并数据统计.csv", merged)
    write_csv(OUT_DIR / "05_评测集必需字段核验.csv", eval_rows)
    write_csv(OUT_DIR / "06_OCR错误案例.csv", error_examples)
    write_csv(OUT_DIR / "07_OCR错误类型计数.csv", error_counts)
    write_csv(OUT_DIR / "08_数据处理流程梳理.csv", workflow)
    write_csv(OUT_DIR / "09_不可由现有归档支持的指标.csv", unavailable)

    summary = {
        "source_root": str(PROCESS_ROOT),
        "output_dir": str(OUT_DIR),
        "txt_rows": len(txt),
        "field_rows": len(field),
        "before_after_rows": len(before_after),
        "merged_rows": len(merged),
        "eval_rows": len(eval_rows),
        "error_example_rows": len(error_examples),
        "error_type_rows": len(error_counts),
        "workflow_rows": len(workflow),
        "unavailable_metric_rows": len(unavailable),
        "main_caveat": "No page-level gold transcription table is archived; CER cannot be reconstructed without new manual annotation.",
        "read_only_sources": [
            str(OCR_TXT_DIR),
            str(RAW_CLEAN_DIR),
            str(PROOFREAD_DIR),
            str(MERGED_DIR),
            str(EVAL_DIR),
            str(OCR_ERROR_LOG),
            str(OLD_STATS),
        ],
    }
    (OUT_DIR / "OCR质量控制统计_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "论文与回复信可用文字.md").write_text(build_text(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
