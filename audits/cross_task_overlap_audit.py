# -*- coding: utf-8 -*-
"""
六任务训练集—测试集跨任务字词与内容重叠审计

用途
----
1. 统计六项测试任务 × 六项微调任务的跨任务字词重叠矩阵；
2. 分别输出：
   - 重叠唯一字词数；
   - 重叠字词占测试集唯一字词比例；
   - 重叠字词占训练集唯一字词比例；
   - Jaccard重叠系数；
3. 对以下高相关任务进行内容级重叠审计：
   - 拼音测试—反切训练，以及反切测试—拼音训练；
   - 词性测试—语境训练，以及语境测试—词性训练；
   - 释义测试—语境训练，以及语境测试—释义训练；
4. 仅生成统计结果和审计明细，不修改任何原始Excel文件。

默认目录
--------
微调数据：./微调数据
测试数据：./eval_data_extract
输出目录：./audit_output_cross_task

运行
----
python cross_task_overlap_audit.py

也可以指定项目根目录和输出目录：
python cross_task_overlap_audit.py --project-root . --output-dir ./audit_output_cross_task

依赖
----
pip install pandas openpyxl
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ============================================================
# 一、任务和文件配置
# ============================================================

TASK_ORDER = [
    "拼音注音",
    "反切注音",
    "部首识别",
    "词性识别",
    "释义生成",
    "语境理解",
]


@dataclass(frozen=True)
class DatasetSpec:
    """描述一个任务数据集的文件、字段和角色。"""

    task: str
    split: str                 # "train" 或 "test"
    relative_path: str
    word_col: str
    label_col: str
    example_col: Optional[str] = None


DATASET_SPECS: Sequence[DatasetSpec] = [
    # -------------------- 测试数据 --------------------
    DatasetSpec(
        task="释义生成",
        split="test",
        relative_path="eval_data_extract/释义&语境_eval.xlsx",
        word_col="字词",
        label_col="所有含义",
    ),
    DatasetSpec(
        task="语境理解",
        split="test",
        relative_path="eval_data_extract/释义&语境_eval.xlsx",
        word_col="字词",
        label_col="释义",
        example_col="例句",
    ),
    DatasetSpec(
        task="拼音注音",
        split="test",
        relative_path="eval_data_extract/pinyin&fanqie_eval_labeled.xlsx",
        word_col="字词",
        label_col="拼音",
    ),
    DatasetSpec(
        task="反切注音",
        split="test",
        relative_path="eval_data_extract/pinyin&fanqie_eval_labeled.xlsx",
        word_col="字词",
        label_col="反切音",
    ),
    DatasetSpec(
        task="部首识别",
        split="test",
        relative_path="eval_data_extract/部首_eval.xlsx",
        word_col="字词",
        label_col="部首",
    ),
    DatasetSpec(
        task="词性识别",
        split="test",
        relative_path="eval_data_extract/词性_eval.xlsx",
        word_col="字词",
        label_col="词性",
        example_col="例句",
    ),

    # -------------------- 微调数据 --------------------
    DatasetSpec(
        task="释义生成",
        split="train",
        relative_path="微调数据/字词_释义_过滤测试数据.xlsx",
        word_col="字词",
        label_col="释义",
    ),
    DatasetSpec(
        task="语境理解",
        split="train",
        relative_path="微调数据/字词_例句_过滤测试数据.xlsx",
        word_col="字词",
        label_col="释义",
        example_col="例句_校准",
    ),
    DatasetSpec(
        task="反切注音",
        split="train",
        relative_path="微调数据/字词_反切音_过滤测试数据.xlsx",
        word_col="字词",
        label_col="反切音",
    ),
    DatasetSpec(
        task="拼音注音",
        split="train",
        relative_path="微调数据/字词_拼音及反切_过滤测试数据.xlsx",
        word_col="字词",
        label_col="拼音",
    ),
    DatasetSpec(
        task="部首识别",
        split="train",
        relative_path="微调数据/字词_部首_过滤测试数据.xlsx",
        word_col="字词",
        label_col="部首",
    ),
    DatasetSpec(
        task="词性识别",
        split="train",
        relative_path="微调数据/字词_词性&例句_过滤测试数据.xlsx",
        word_col="字词",
        label_col="词性",
        example_col="例句_校准",
    ),
]


# ============================================================
# 二、文本规范化
# ============================================================

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_TRANSLATION = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "：": ":",
        "；": ";",
        "，": ",",
        "。": ".",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "　": " ",
    }
)


def is_missing(value: object) -> bool:
    """判断Excel值是否为空。"""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() == ""


def normalize_basic(value: object) -> str:
    """
    基础规范化：
    - Unicode NFKC；
    - 去除首尾空格；
    - 连续空白压缩为一个空格。

    注意：不自动进行繁简转换，不改变异体字。
    """
    if is_missing(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def normalize_compact(value: object) -> str:
    """
    紧凑规范化，主要用于例句和释义比较：
    - 执行基础规范化；
    - 统一部分中英文标点；
    - 删除所有空白。

    保留文字本身，不进行繁简转换。
    """
    text = normalize_basic(value).translate(_PUNCT_TRANSLATION)
    return _WHITESPACE_RE.sub("", text)


# ============================================================
# 三、加载与检查数据
# ============================================================

def required_columns(spec: DatasetSpec) -> List[str]:
    cols = [spec.word_col, spec.label_col]
    if spec.example_col:
        cols.append(spec.example_col)
    return cols


def load_dataset(project_root: Path, spec: DatasetSpec) -> pd.DataFrame:
    """读取一个Excel数据集并映射为统一逻辑字段。"""
    path = project_root / spec.relative_path
    if not path.exists():
        raise FileNotFoundError(f"未找到文件：{path}")

    try:
        raw = pd.read_excel(path, engine="openpyxl")
    except Exception as exc:
        raise RuntimeError(f"读取Excel失败：{path}\n原始错误：{exc}") from exc

    missing_cols = [col for col in required_columns(spec) if col not in raw.columns]
    if missing_cols:
        raise ValueError(
            f"文件 {path} 缺少字段：{missing_cols}\n"
            f"现有字段：{list(raw.columns)}"
        )

    selected_cols = required_columns(spec)
    df = raw[selected_cols].copy()

    rename_map = {
        spec.word_col: "字词_原始",
        spec.label_col: "标签_原始",
    }
    if spec.example_col:
        rename_map[spec.example_col] = "例句_原始"

    df = df.rename(columns=rename_map)
    if "例句_原始" not in df.columns:
        df["例句_原始"] = ""

    df.insert(0, "原始行号", np.arange(2, len(df) + 2))
    df.insert(0, "数据划分", spec.split)
    df.insert(0, "任务", spec.task)
    df.insert(0, "来源文件", spec.relative_path)

    df["字词"] = df["字词_原始"].map(normalize_basic)
    df["标签"] = df["标签_原始"].map(normalize_basic)
    df["例句"] = df["例句_原始"].map(normalize_basic)

    df["字词_key"] = df["字词"].map(normalize_compact)
    df["标签_key"] = df["标签"].map(normalize_compact)
    df["例句_key"] = df["例句"].map(normalize_compact)

    df["字词为空"] = df["字词_key"].eq("")
    df["标签为空"] = df["标签_key"].eq("")
    df["例句为空"] = (
        df["例句_key"].eq("") if spec.example_col else False
    )

    # 字词为空的记录不参与重叠计算，但保留在质量报告中。
    return df


def load_all_datasets(
    project_root: Path,
) -> Dict[Tuple[str, str], pd.DataFrame]:
    datasets: Dict[Tuple[str, str], pd.DataFrame] = {}
    errors: List[str] = []

    for spec in DATASET_SPECS:
        key = (spec.split, spec.task)
        try:
            datasets[key] = load_dataset(project_root, spec)
            print(
                f"[OK] {spec.split:5s} | {spec.task:6s} | "
                f"{spec.relative_path} | {len(datasets[key])} 行"
            )
        except Exception as exc:
            errors.append(str(exc))

    if errors:
        message = "\n\n".join(f"{i + 1}. {err}" for i, err in enumerate(errors))
        raise RuntimeError(f"数据加载失败，共发现 {len(errors)} 个问题：\n\n{message}")

    return datasets


# ============================================================
# 四、基础数据集统计
# ============================================================

def dataset_summary(
    datasets: Dict[Tuple[str, str], pd.DataFrame]
) -> pd.DataFrame:
    rows: List[dict] = []

    for split in ["train", "test"]:
        for task in TASK_ORDER:
            df = datasets[(split, task)]
            valid_words = df.loc[~df["字词为空"], "字词_key"]
            valid_labels = df.loc[~df["标签为空"], "标签_key"]

            rows.append(
                {
                    "数据划分": split,
                    "任务": task,
                    "来源文件": df["来源文件"].iloc[0],
                    "总行数": len(df),
                    "唯一字词数": valid_words.nunique(),
                    "唯一标签数": valid_labels.nunique(),
                    "字词空值行数": int(df["字词为空"].sum()),
                    "标签空值行数": int(df["标签为空"].sum()),
                    "例句空值行数": int(df["例句为空"].sum()),
                    "重复字词涉及行数": int(
                        valid_words.duplicated(keep=False).sum()
                    ),
                    "字词标签完全重复涉及行数": int(
                        df.loc[
                            ~(df["字词为空"] | df["标签为空"]),
                            ["字词_key", "标签_key"],
                        ].duplicated(keep=False).sum()
                    ),
                }
            )

    return pd.DataFrame(rows)


def word_set(df: pd.DataFrame) -> set[str]:
    return set(df.loc[~df["字词为空"], "字词_key"])


# ============================================================
# 五、六乘六跨任务字词重叠矩阵
# ============================================================

def build_headword_overlap_matrices(
    datasets: Dict[Tuple[str, str], pd.DataFrame]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    count_matrix = pd.DataFrame(index=TASK_ORDER, columns=TASK_ORDER, dtype=int)
    test_rate_matrix = pd.DataFrame(index=TASK_ORDER, columns=TASK_ORDER, dtype=float)
    train_rate_matrix = pd.DataFrame(index=TASK_ORDER, columns=TASK_ORDER, dtype=float)
    jaccard_matrix = pd.DataFrame(index=TASK_ORDER, columns=TASK_ORDER, dtype=float)

    detail_rows: List[dict] = []

    for test_task in TASK_ORDER:
        test_df = datasets[("test", test_task)]
        test_words = word_set(test_df)

        for train_task in TASK_ORDER:
            train_df = datasets[("train", train_task)]
            train_words = word_set(train_df)

            overlap = sorted(test_words & train_words)
            union = test_words | train_words

            count_matrix.loc[test_task, train_task] = len(overlap)
            test_rate_matrix.loc[test_task, train_task] = (
                len(overlap) / len(test_words) if test_words else np.nan
            )
            train_rate_matrix.loc[test_task, train_task] = (
                len(overlap) / len(train_words) if train_words else np.nan
            )
            jaccard_matrix.loc[test_task, train_task] = (
                len(overlap) / len(union) if union else np.nan
            )

            # 将同一字词在测试和训练中的标签汇总，避免明细表出现笛卡尔爆炸。
            if overlap:
                test_group = (
                    test_df[test_df["字词_key"].isin(overlap)]
                    .groupby("字词_key", dropna=False)
                    .agg(
                        测试字词=("字词", lambda x: " || ".join(sorted(set(x)))),
                        测试标签=("标签", lambda x: " || ".join(sorted(set(x)))),
                        测试例句=("例句", lambda x: " || ".join(sorted(v for v in set(x) if v))),
                        测试行号=("原始行号", lambda x: ", ".join(map(str, x))),
                    )
                    .reset_index()
                )
                train_group = (
                    train_df[train_df["字词_key"].isin(overlap)]
                    .groupby("字词_key", dropna=False)
                    .agg(
                        训练字词=("字词", lambda x: " || ".join(sorted(set(x)))),
                        训练标签=("标签", lambda x: " || ".join(sorted(set(x)))),
                        训练例句=("例句", lambda x: " || ".join(sorted(v for v in set(x) if v))),
                        训练行号=("原始行号", lambda x: ", ".join(map(str, x))),
                    )
                    .reset_index()
                )

                pair_detail = test_group.merge(
                    train_group,
                    on="字词_key",
                    how="inner",
                    validate="one_to_one",
                )

                for _, row in pair_detail.iterrows():
                    detail_rows.append(
                        {
                            "测试任务": test_task,
                            "训练任务": train_task,
                            "是否同任务": test_task == train_task,
                            "字词_key": row["字词_key"],
                            "测试字词": row["测试字词"],
                            "训练字词": row["训练字词"],
                            "测试标签": row["测试标签"],
                            "训练标签": row["训练标签"],
                            "测试例句": row["测试例句"],
                            "训练例句": row["训练例句"],
                            "测试Excel行号": row["测试行号"],
                            "训练Excel行号": row["训练行号"],
                        }
                    )

    count_matrix.index.name = "测试任务\\训练任务"
    test_rate_matrix.index.name = "测试任务\\训练任务"
    train_rate_matrix.index.name = "测试任务\\训练任务"
    jaccard_matrix.index.name = "测试任务\\训练任务"

    detail_df = pd.DataFrame(detail_rows)
    return (
        count_matrix.astype("Int64"),
        test_rate_matrix,
        train_rate_matrix,
        jaccard_matrix,
        detail_df,
    )


# ============================================================
# 六、内容级重叠：通用辅助函数
# ============================================================

def aggregate_values(values: Iterable[object]) -> str:
    normalized = [normalize_basic(v) for v in values if not is_missing(v)]
    return " || ".join(sorted(set(v for v in normalized if v)))


def pair_by_word(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    """按规范化字词连接测试和训练记录，保留行级组合。"""
    left = test_df.loc[
        ~test_df["字词为空"],
        [
            "来源文件",
            "原始行号",
            "字词",
            "字词_key",
            "标签",
            "标签_key",
            "例句",
            "例句_key",
        ],
    ].copy()
    left = left.rename(
        columns={
            "来源文件": "测试来源文件",
            "原始行号": "测试行号",
            "字词": "测试字词",
            "标签": "测试标签",
            "标签_key": "测试标签_key",
            "例句": "测试例句",
            "例句_key": "测试例句_key",
        }
    )

    right = train_df.loc[
        ~train_df["字词为空"],
        [
            "来源文件",
            "原始行号",
            "字词",
            "字词_key",
            "标签",
            "标签_key",
            "例句",
            "例句_key",
        ],
    ].copy()
    right = right.rename(
        columns={
            "来源文件": "训练来源文件",
            "原始行号": "训练行号",
            "字词": "训练字词",
            "标签": "训练标签",
            "标签_key": "训练标签_key",
            "例句": "训练例句",
            "例句_key": "训练例句_key",
        }
    )

    return left.merge(right, on="字词_key", how="inner")


def overlap_summary_row(
    pair_name: str,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    paired: pd.DataFrame,
    exact_example_mask: Optional[pd.Series] = None,
    exact_label_mask: Optional[pd.Series] = None,
    label_containment_mask: Optional[pd.Series] = None,
    note: str = "",
) -> dict:
    test_words = word_set(test_df)
    train_words = word_set(train_df)
    overlap_words = test_words & train_words

    row = {
        "审计组合": pair_name,
        "测试总行数": len(test_df),
        "训练总行数": len(train_df),
        "测试唯一字词数": len(test_words),
        "训练唯一字词数": len(train_words),
        "重叠唯一字词数": len(overlap_words),
        "重叠字词占测试集比例": (
            len(overlap_words) / len(test_words) if test_words else np.nan
        ),
        "重叠字词占训练集比例": (
            len(overlap_words) / len(train_words) if train_words else np.nan
        ),
        "同字行级组合数": len(paired),
        "相同字词+相同例句组合数": (
            int(exact_example_mask.sum()) if exact_example_mask is not None else np.nan
        ),
        "相同字词+标签完全相同组合数": (
            int(exact_label_mask.sum()) if exact_label_mask is not None else np.nan
        ),
        "相同字词+标签包含关系组合数": (
            int(label_containment_mask.sum())
            if label_containment_mask is not None
            else np.nan
        ),
        "说明": note,
    }
    return row


# ============================================================
# 七、拼音—反切内容级统计
# ============================================================

def audit_pinyin_fanqie(
    datasets: Dict[Tuple[str, str], pd.DataFrame]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    拼音和反切标签不同，因此不将同字自动视为直接标签泄露。
    输出两个方向的同字重叠及标签明细：
    1. 拼音测试 → 反切训练
    2. 反切测试 → 拼音训练
    """
    combinations = [
        (
            "拼音测试→反切训练",
            datasets[("test", "拼音注音")],
            datasets[("train", "反切注音")],
        ),
        (
            "反切测试→拼音训练",
            datasets[("test", "反切注音")],
            datasets[("train", "拼音注音")],
        ),
    ]

    summary_rows: List[dict] = []
    detail_parts: List[pd.DataFrame] = []

    for pair_name, test_df, train_df in combinations:
        paired = pair_by_word(test_df, train_df)
        paired.insert(0, "审计组合", pair_name)
        paired["风险解释"] = (
            "同一字词在相关语音任务中出现，属于跨任务语音监督；"
            "测试任务标签并未直接出现在该训练记录中。"
        )

        summary_rows.append(
            overlap_summary_row(
                pair_name=pair_name,
                test_df=test_df,
                train_df=train_df,
                paired=paired,
                note=(
                    "仅统计同字语音信息重叠。拼音和反切属于相关但不同标签，"
                    "该统计用于描述跨任务知识迁移，不自动判定为数据泄露。"
                ),
            )
        )

        detail_parts.append(paired)

    return pd.DataFrame(summary_rows), pd.concat(detail_parts, ignore_index=True)


# ============================================================
# 八、词性—语境内容级统计
# ============================================================

def audit_pos_context(
    datasets: Dict[Tuple[str, str], pd.DataFrame]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    combinations = [
        (
            "词性测试→语境训练",
            datasets[("test", "词性识别")],
            datasets[("train", "语境理解")],
        ),
        (
            "语境测试→词性训练",
            datasets[("test", "语境理解")],
            datasets[("train", "词性识别")],
        ),
    ]

    summary_rows: List[dict] = []
    detail_parts: List[pd.DataFrame] = []

    for pair_name, test_df, train_df in combinations:
        paired = pair_by_word(test_df, train_df)

        exact_example = (
            paired["测试例句_key"].ne("")
            & paired["训练例句_key"].ne("")
            & paired["测试例句_key"].eq(paired["训练例句_key"])
        )

        paired.insert(0, "审计组合", pair_name)
        paired["字词与例句完全相同"] = exact_example
        paired["风险等级"] = np.where(
            exact_example,
            "较高相关内容重叠",
            "仅字词重叠或例句不同",
        )
        paired["风险解释"] = np.where(
            exact_example,
            "同一字词和同一例句同时出现，可能提供较强的跨任务语境监督。",
            "同一字词出现在不同例句中，通常属于多任务共享字词。",
        )

        summary_rows.append(
            overlap_summary_row(
                pair_name=pair_name,
                test_df=test_df,
                train_df=train_df,
                paired=paired,
                exact_example_mask=exact_example,
                note=(
                    "词性与语境任务重点检查“字词+例句”是否完全相同。"
                    "只有同字同例句才记为较强内容重叠；同字异例句不自动视为泄露。"
                ),
            )
        )

        detail_parts.append(paired)

    return pd.DataFrame(summary_rows), pd.concat(detail_parts, ignore_index=True)


# ============================================================
# 九、释义—语境内容级统计
# ============================================================

def label_containment(
    left: pd.Series,
    right: pd.Series,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    返回：
    1. 完全相同；
    2. 左标签包含右标签；
    3. 右标签包含左标签。

    这是文本包含关系启发式，不等同于语义等价。

    说明：
    旧版代码在没有可比较记录时，会尝试把空列表写入bool列，
    在部分较新的pandas版本中触发：
    Invalid value '[]' for dtype 'bool'
    这里改为直接按行生成布尔Series，因此空DataFrame也可安全处理。
    """
    left_text = left.fillna("").astype(str)
    right_text = right.fillna("").astype(str)

    exact = (
        left_text.ne("")
        & right_text.ne("")
        & left_text.eq(right_text)
    ).astype(bool)

    left_contains_right = pd.Series(
        [
            bool(l and r and r in l)
            for l, r in zip(left_text.tolist(), right_text.tolist())
        ],
        index=left.index,
        dtype=bool,
    )

    right_contains_left = pd.Series(
        [
            bool(l and r and l in r)
            for l, r in zip(left_text.tolist(), right_text.tolist())
        ],
        index=left.index,
        dtype=bool,
    )

    return exact, left_contains_right, right_contains_left


def audit_definition_context(
    datasets: Dict[Tuple[str, str], pd.DataFrame]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    combinations = [
        (
            "释义测试→语境训练",
            datasets[("test", "释义生成")],
            datasets[("train", "语境理解")],
            "测试标签为“所有含义”，训练标签为具体语境义。",
        ),
        (
            "语境测试→释义训练",
            datasets[("test", "语境理解")],
            datasets[("train", "释义生成")],
            "测试标签为具体语境义，训练标签为释义生成标签。",
        ),
    ]

    summary_rows: List[dict] = []
    detail_parts: List[pd.DataFrame] = []

    for pair_name, test_df, train_df, direction_note in combinations:
        paired = pair_by_word(test_df, train_df)

        exact, test_contains_train, train_contains_test = label_containment(
            paired["测试标签_key"],
            paired["训练标签_key"],
        )
        any_containment = test_contains_train | train_contains_test

        paired.insert(0, "审计组合", pair_name)
        paired["标签完全相同"] = exact
        paired["测试标签包含训练标签"] = test_contains_train
        paired["训练标签包含测试标签"] = train_contains_test
        paired["存在标签文本包含关系"] = any_containment
        paired["风险等级"] = np.select(
            [
                exact,
                any_containment,
            ],
            [
                "标签完全重叠",
                "标签部分文本重叠",
            ],
            default="仅字词重叠",
        )
        paired["风险解释"] = np.select(
            [
                exact,
                any_containment,
            ],
            [
                "同一字词的标签文本完全一致，属于较强的跨任务标签内容重叠。",
                "同一字词的一个标签文本包含另一个标签，可能反映义项知识迁移。",
            ],
            default="同一字词出现，但当前字符串规则未发现标签内容直接复用。",
        )

        summary_rows.append(
            overlap_summary_row(
                pair_name=pair_name,
                test_df=test_df,
                train_df=train_df,
                paired=paired,
                exact_label_mask=exact,
                label_containment_mask=any_containment,
                note=(
                    f"{direction_note}"
                    "标签包含关系采用规范化字符串包含规则，仅作为审计提示，"
                    "不等同于人工确认的语义等价。"
                ),
            )
        )

        detail_parts.append(paired)

    return pd.DataFrame(summary_rows), pd.concat(detail_parts, ignore_index=True)


# ============================================================
# 十、同任务关键审计摘要
# ============================================================

def same_task_audit(
    datasets: Dict[Tuple[str, str], pd.DataFrame]
) -> pd.DataFrame:
    """
    汇总每个任务对应训练集—测试集的：
    - 字词重叠；
    - 字词+标签完全重叠；
    - 完整内容重叠。

    这里的“完整内容”按任务已有字段定义：
    - 无例句任务：字词+标签；
    - 有例句任务：字词+例句+标签。
    """
    rows: List[dict] = []

    for task in TASK_ORDER:
        test_df = datasets[("test", task)]
        train_df = datasets[("train", task)]
        paired = pair_by_word(test_df, train_df)

        label_exact = (
            paired["测试标签_key"].ne("")
            & paired["训练标签_key"].ne("")
            & paired["测试标签_key"].eq(paired["训练标签_key"])
        )

        has_examples = (
            test_df["例句_key"].ne("").any()
            or train_df["例句_key"].ne("").any()
        )
        if has_examples:
            example_exact = (
                paired["测试例句_key"].ne("")
                & paired["训练例句_key"].ne("")
                & paired["测试例句_key"].eq(paired["训练例句_key"])
            )
            full_exact = label_exact & example_exact
        else:
            example_exact = pd.Series(False, index=paired.index)
            full_exact = label_exact

        rows.append(
            {
                "任务": task,
                "测试唯一字词数": len(word_set(test_df)),
                "训练唯一字词数": len(word_set(train_df)),
                "同任务重叠唯一字词数": len(word_set(test_df) & word_set(train_df)),
                "同任务字词+标签完全相同组合数": int(label_exact.sum()),
                "同任务字词+例句完全相同组合数": int(example_exact.sum()),
                "同任务完整内容完全相同组合数": int(full_exact.sum()),
                "审计判断": (
                    "通过：未发现同任务字词重叠"
                    if len(word_set(test_df) & word_set(train_df)) == 0
                    else "需核查：发现同任务字词重叠"
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 十一、输出
# ============================================================

def autosize_worksheet(writer: pd.ExcelWriter, sheet_name: str, df: pd.DataFrame) -> None:
    """设置审计结果工作表的基本可读性。"""
    from copy import copy

    worksheet = writer.sheets[sheet_name]
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions

    # 直接按已经写入工作表的实际列估算宽度，避免索引列错位。
    for column_cells in worksheet.iter_cols(
        min_row=1, max_row=min(worksheet.max_row, 2001)
    ):
        max_len = max(
            len(str(cell.value)) if cell.value is not None else 0
            for cell in column_cells
        )
        width = min(max(max_len + 2, 10), 45)
        worksheet.column_dimensions[column_cells[0].column_letter].width = width

    for cell in worksheet[1]:
        font = copy(cell.font)
        font.bold = True
        cell.font = font

        alignment = copy(cell.alignment)
        alignment.horizontal = "center"
        alignment.vertical = "center"
        cell.alignment = alignment

    for row in worksheet.iter_rows(min_row=2):
        for cell in row:
            alignment = copy(cell.alignment)
            alignment.vertical = "top"
            alignment.wrap_text = True
            cell.alignment = alignment


def write_csv(df: pd.DataFrame, path: Path, index: bool = True) -> None:
    df.to_csv(path, index=index, encoding="utf-8-sig")


def export_results(
    output_dir: Path,
    summary_df: pd.DataFrame,
    count_matrix: pd.DataFrame,
    test_rate_matrix: pd.DataFrame,
    train_rate_matrix: pd.DataFrame,
    jaccard_matrix: pd.DataFrame,
    headword_detail: pd.DataFrame,
    same_task_df: pd.DataFrame,
    pinyin_fanqie_summary: pd.DataFrame,
    pinyin_fanqie_detail: pd.DataFrame,
    pos_context_summary: pd.DataFrame,
    pos_context_detail: pd.DataFrame,
    definition_context_summary: pd.DataFrame,
    definition_context_detail: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 百分比显示版本，底层数值仍保留为0-1。
    test_rate_display = test_rate_matrix.copy()
    train_rate_display = train_rate_matrix.copy()
    jaccard_display = jaccard_matrix.copy()

    xlsx_path = output_dir / "跨任务字词与内容重叠审计.xlsx"

    sheets: List[Tuple[str, pd.DataFrame, bool]] = [
        ("数据集汇总", summary_df, False),
        ("同任务关键审计", same_task_df, False),
        ("字词重叠数矩阵", count_matrix, True),
        ("占测试集比例矩阵", test_rate_display, True),
        ("占训练集比例矩阵", train_rate_display, True),
        ("Jaccard矩阵", jaccard_display, True),
        ("全部字词重叠明细", headword_detail, False),
        ("拼音反切汇总", pinyin_fanqie_summary, False),
        ("拼音反切明细", pinyin_fanqie_detail, False),
        ("词性语境汇总", pos_context_summary, False),
        ("词性语境明细", pos_context_detail, False),
        ("释义语境汇总", definition_context_summary, False),
        ("释义语境明细", definition_context_detail, False),
    ]

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for sheet_name, df, use_index in sheets:
            safe_df = df.copy()
            safe_df.to_excel(writer, sheet_name=sheet_name, index=use_index)
            autosize_worksheet(writer, sheet_name, safe_df)

        # 设置比例工作表为百分比格式。
        for sheet_name in ["占测试集比例矩阵", "占训练集比例矩阵", "Jaccard矩阵"]:
            ws = writer.sheets[sheet_name]
            for row in ws.iter_rows(min_row=2, min_col=2):
                for cell in row:
                    cell.number_format = "0.00%"

        # 内容级汇总中的比例列。
        for sheet_name in ["拼音反切汇总", "词性语境汇总", "释义语境汇总"]:
            ws = writer.sheets[sheet_name]
            headers = {cell.value: cell.column for cell in ws[1]}
            for header in ["重叠字词占测试集比例", "重叠字词占训练集比例"]:
                col_idx = headers.get(header)
                if col_idx:
                    for row in range(2, ws.max_row + 1):
                        ws.cell(row=row, column=col_idx).number_format = "0.00%"

    # 同时输出便于投稿附录或进一步处理的CSV。
    write_csv(count_matrix, output_dir / "01_跨任务字词重叠数矩阵.csv")
    write_csv(test_rate_matrix, output_dir / "02_跨任务字词占测试集比例矩阵.csv")
    write_csv(train_rate_matrix, output_dir / "03_跨任务字词占训练集比例矩阵.csv")
    write_csv(jaccard_matrix, output_dir / "04_跨任务字词Jaccard矩阵.csv")
    write_csv(same_task_df, output_dir / "05_同任务关键审计.csv", index=False)
    write_csv(pinyin_fanqie_summary, output_dir / "06_拼音反切内容重叠汇总.csv", index=False)
    write_csv(pos_context_summary, output_dir / "07_词性语境内容重叠汇总.csv", index=False)
    write_csv(definition_context_summary, output_dir / "08_释义语境内容重叠汇总.csv", index=False)

    print(f"\n审计完成。主结果文件：{xlsx_path}")
    print(f"CSV结果目录：{output_dir}")


# ============================================================
# 十二、主程序
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="统计六任务训练集与测试集的跨任务字词及内容级重叠。"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="项目根目录，默认当前目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./audit_output_cross_task"),
        help="输出目录；相对路径将基于项目根目录解析。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir = output_dir.resolve()

    print("=" * 72)
    print("六任务训练集—测试集跨任务重叠审计")
    print(f"项目根目录：{project_root}")
    print(f"结果输出目录：{output_dir}")
    print("=" * 72)

    try:
        datasets = load_all_datasets(project_root)

        summary_df = dataset_summary(datasets)
        (
            count_matrix,
            test_rate_matrix,
            train_rate_matrix,
            jaccard_matrix,
            headword_detail,
        ) = build_headword_overlap_matrices(datasets)

        same_task_df = same_task_audit(datasets)

        pinyin_fanqie_summary, pinyin_fanqie_detail = audit_pinyin_fanqie(datasets)
        pos_context_summary, pos_context_detail = audit_pos_context(datasets)
        definition_context_summary, definition_context_detail = (
            audit_definition_context(datasets)
        )

        export_results(
            output_dir=output_dir,
            summary_df=summary_df,
            count_matrix=count_matrix,
            test_rate_matrix=test_rate_matrix,
            train_rate_matrix=train_rate_matrix,
            jaccard_matrix=jaccard_matrix,
            headword_detail=headword_detail,
            same_task_df=same_task_df,
            pinyin_fanqie_summary=pinyin_fanqie_summary,
            pinyin_fanqie_detail=pinyin_fanqie_detail,
            pos_context_summary=pos_context_summary,
            pos_context_detail=pos_context_detail,
            definition_context_summary=definition_context_summary,
            definition_context_detail=definition_context_detail,
        )

        print("\n同任务关键审计摘要：")
        print(
            same_task_df[
                [
                    "任务",
                    "同任务重叠唯一字词数",
                    "同任务字词+标签完全相同组合数",
                    "同任务完整内容完全相同组合数",
                    "审计判断",
                ]
            ].to_string(index=False)
        )

        return 0

    except Exception as exc:
        print("\n[ERROR] 审计未完成：")
        print(exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())