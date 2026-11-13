from __future__ import annotations
"""
统一构造A/B/C三组预训练污染测试数据，并输出最终selected_100。

依赖
====
本脚本与 build_A_structured_contamination.py 放在同一目录。
A组的结构化辞书解析、两套文本重建与核心切分直接复用该脚本。

默认最小设计
============
A组：《十三经辞典》核心来源条目 50条
B组：《汉语大词典》补充条目 30条
C组：公开《论语》《孟子》正文 20条
总计100条。

两套文本
========
A/B组：
- entry_core_text：辞书核心释义，主污染测试文本；
- entry_full_text：加入音韵信息/例句的完整重建文本，辅助分析。

C组：
- entry_core_text == entry_full_text；
- 用于验证方法能够检测公开经典文本记忆。

主实验建议
==========
对A/B组使用core_visible_prefix和core_reference_suffix。
C组使用core_visible_prefix和core_reference_suffix。
full_*字段作为辅助分析，不建议自动把同一条样本重复计算为第二条独立样本。

B组输入格式
===========
支持实际截图中的列：
id, word, guangyun, senseid, sense, exp, note

默认按id排序；同一word下senseid重新从1开始或下降时，拆为新的词条块。
B组核心文本使用sense+note；exp视为外部文献引例，guangyun视为音韵信息。
完整文本保留guangyun、sense、exp和note。

C组输入格式
===========
url, title, paragraph
在同一url/title内按Excel原始顺序构造连续、非重叠文本窗口。

正式使用前
==========
所有selected样本的manual_verified初始为Pending。
正式调用模型前应逐条核验并改为Pass。
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一构造A/B/C三组污染测试数据并输出最终selected_100。"""

import argparse
import html
import importlib.util
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from openpyxl import Workbook, load_workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation


SELECTION_SEED = 42
SPLIT_SEED = 42

DEFAULT_A_QUOTAS = {
    "论语卷": 25,
    "孟子卷": 25,
}

DEFAULT_C_QUOTAS = {
    "论语": 10,
    "孟子": 10,
}

DEFAULT_B_SIZE = 30

MIN_CORE_BODY_LENGTH = 60
MIN_VISIBLE_LENGTH = 20
MIN_CORE_SUFFIX_LENGTH = 30

C_WINDOW_MIN_LENGTH = 80
C_WINDOW_MAX_LENGTH = 180
C_SHORT_WINDOW_SIZE = 180
C_SHORT_WINDOW_STEP = 120

CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"

CJK_RE = re.compile(
    r"[\u3400-\u9fff\uf900-\ufaff]"
)

ZERO_WIDTH_RE = re.compile(
    r"[\u200b-\u200f\u2060\ufeff]"
)

CONTROL_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f]"
)

SENTENCE_END_RE = re.compile(
    r"[。！？；]"
)

TERMINAL_PUNCTUATION = set(
    "。！？；：，、」』”’）》】)"
)

HANYU_BRACE_TAG_RE = re.compile(
    r"\{([^{}]+)\}"
)

HANYU_CROSSREF_RE = re.compile(
    r"[≦≤]+\s*(.*?)\s*[≧≥]+"
)

MULTI_PUNCT_RE = re.compile(
    r"([。！？；，：])\1+"
)

HANYU_ALIASES = {
    "id": {
        "id",
        "编号",
        "序号",
    },
    "word": {
        "word",
        "字词",
        "字詞",
        "词头",
        "詞頭",
    },
    "guangyun": {
        "guangyun",
        "廣韻",
        "广韵",
        "音韵",
        "音韻",
    },
    "senseid": {
        "senseid",
        "sense_id",
        "义项编号",
        "義項編號",
    },
    "sense": {
        "sense",
        "释义",
        "釋義",
        "义项",
        "義項",
    },
    "exp": {
        "exp",
        "example",
        "例句",
        "引例",
        "文献引例",
        "文獻引例",
    },
    "note": {
        "note",
        "备注",
        "備註",
        "说明",
        "說明",
    },
}


def load_a_builder(
    path: str,
):
    """
    动态加载同一目录中的A组构造脚本。
    """
    module_path = Path(path).resolve()

    if not module_path.exists():
        raise FileNotFoundError(
            f"未找到A组脚本：{module_path}"
        )

    spec = importlib.util.spec_from_file_location(
        "a_group_builder",
        module_path,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise ImportError(
            f"无法加载：{module_path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(module)

    return module


def normalize_header(
    value: Any,
) -> str:
    if value is None:
        return ""

    text = unicodedata.normalize(
        "NFKC",
        str(value),
    )

    text = ZERO_WIDTH_RE.sub(
        "",
        text,
    )

    return text.strip().lower()


def normalize_text(
    value: Any,
) -> str:
    if value is None:
        return ""

    text = html.unescape(
        str(value)
    )

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = ZERO_WIDTH_RE.sub(
        "",
        text,
    )

    text = CONTROL_RE.sub(
        "",
        text,
    )

    text = text.replace(
        "\u3000",
        " ",
    )

    text = re.sub(
        r"[\r\n\t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def ensure_terminal_punctuation(
    text: str,
) -> str:
    text = text.strip()

    if not text:
        return ""

    if text[-1] in TERMINAL_PUNCTUATION:
        return text

    return text + "。"


def circled_number(
    index: int,
) -> str:
    if 1 <= index <= len(CIRCLED):
        return CIRCLED[index - 1]

    return f"({index})"


def safe_int(
    value: Any,
) -> Optional[int]:
    if value is None or value == "":
        return None

    try:
        return int(
            float(
                str(value).strip()
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        return None


def cjk_length(
    text: str,
) -> int:
    return len(
        CJK_RE.findall(text)
    )


def read_first_sheet(
    path: str,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """使用openpyxl读取第一个工作表。"""
    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"文件不存在：{input_path}"
        )

    workbook = load_workbook(
        filename=input_path,
        read_only=True,
        data_only=True,
    )

    try:
        if not workbook.worksheets:
            raise ValueError(
                f"工作簿中没有工作表：{input_path}"
            )

        sheet = workbook.worksheets[0]

        values = list(
            sheet.iter_rows(
                values_only=True
            )
        )
    finally:
        workbook.close()

    if not values:
        raise ValueError(
            f"文件为空：{input_path}"
        )

    headers = [
        str(value).strip()
        if value is not None
        else ""
        for value in values[0]
    ]

    rows: List[Dict[str, Any]] = []

    for excel_row, row in enumerate(
        values[1:],
        start=2,
    ):
        record = {
            headers[index]: (
                row[index]
                if index < len(row)
                else None
            )
            for index in range(len(headers))
            if headers[index]
        }

        record["_excel_row"] = excel_row
        rows.append(record)

    return headers, rows


def parse_named_paths(
    values: Sequence[str],
) -> Dict[str, str]:
    result: Dict[str, str] = {}

    for value in values:
        if "=" not in value:
            raise ValueError(
                "必须使用“名称=路径”："
                f"{value}"
            )

        name, path = value.split(
            "=",
            1,
        )

        result[name.strip()] = (
            path.strip()
        )

    return result


def parse_quota(
    value: str,
    default: Dict[str, int],
) -> Dict[str, int]:
    if not value:
        return dict(default)

    result: Dict[str, int] = {}

    for item in value.split(","):
        name, count = item.split(
            "=",
            1,
        )

        result[name.strip()] = int(
            count.strip()
        )

    return result


def add_exclusion(
    record: Dict[str, Any],
    reason: str,
) -> None:
    reasons = [
        item
        for item in str(
            record.get(
                "exclusion_reason",
                "",
            )
        ).split(";")
        if item
    ]

    if reason not in reasons:
        reasons.append(reason)

    record["exclusion_reason"] = (
        ";".join(reasons)
    )


def length_bin(
    length: int,
) -> str:
    if length < 100:
        return "L1_60_99"

    if length < 160:
        return "L2_100_159"

    if length < 240:
        return "L3_160_239"

    return "L4_240_plus"


def sentence_boundaries(
    text: str,
) -> List[int]:
    return [
        match.end()
        for match
        in SENTENCE_END_RE.finditer(text)
    ]


def generic_single_text_split(
    record_id: str,
    text: str,
) -> Optional[
    Tuple[str, str, float]
]:
    valid_positions = [
        position
        for position in sentence_boundaries(
            text
        )
        if (
            position >= MIN_VISIBLE_LENGTH
            and (
                len(text)
                - position
                >= MIN_CORE_SUFFIX_LENGTH
            )
        )
    ]

    if not valid_positions:
        return None

    rng = random.Random(
        f"{SPLIT_SEED}:{record_id}"
    )

    target_ratio = rng.uniform(
        0.40,
        0.60,
    )

    target_position = round(
        len(text)
        * target_ratio
    )

    split_index = min(
        valid_positions,
        key=lambda position: abs(
            position
            - target_position
        ),
    )

    return (
        text[:split_index].strip(),
        text[split_index:].strip(),
        target_ratio,
    )


def split_blocks(
    record_id: str,
    core_header: str,
    full_header: str,
    core_blocks: Sequence[str],
    full_blocks: Sequence[str],
) -> Optional[Dict[str, Any]]:
    """
    优先在自然义项边界切分；
    单义项时再使用完整句末切分。
    """
    rng = random.Random(
        f"{SPLIT_SEED}:{record_id}"
    )

    target_ratio = rng.uniform(
        0.40,
        0.60,
    )

    options = []

    for split_index in range(
        1,
        len(core_blocks),
    ):
        core_prefix = (
            core_header
            + " "
            + "".join(
                core_blocks[:split_index]
            )
        ).strip()

        core_suffix = "".join(
            core_blocks[split_index:]
        ).strip()

        full_prefix = (
            full_header
            + " "
            + "".join(
                full_blocks[:split_index]
            )
        ).strip()

        full_suffix = "".join(
            full_blocks[split_index:]
        ).strip()

        if (
            len(core_prefix)
            < MIN_VISIBLE_LENGTH
            or len(core_suffix)
            < MIN_CORE_SUFFIX_LENGTH
        ):
            continue

        actual_ratio = (
            len(core_prefix)
            / max(
                1,
                len(core_prefix)
                + len(core_suffix),
            )
        )

        options.append(
            (
                abs(
                    actual_ratio
                    - target_ratio
                ),
                split_index,
                core_prefix,
                core_suffix,
                full_prefix,
                full_suffix,
            )
        )

    if options:
        (
            _,
            split_index,
            core_prefix,
            core_suffix,
            full_prefix,
            full_suffix,
        ) = min(
            options,
            key=lambda item: (
                item[0],
                item[1],
            ),
        )

        return {
            "split_rule": (
                "natural_sense_boundary"
            ),
            "split_sense_index": (
                split_index
            ),
            "target_split_ratio": round(
                target_ratio,
                4,
            ),
            "core_visible_prefix": (
                core_prefix
            ),
            "core_reference_suffix": (
                core_suffix
            ),
            "full_visible_prefix": (
                full_prefix
            ),
            "full_reference_suffix": (
                full_suffix
            ),
        }

    if len(core_blocks) == 1:
        result = generic_single_text_split(
            record_id,
            core_blocks[0],
        )

        if result is not None:
            (
                prefix_piece,
                suffix_piece,
                single_target_ratio,
            ) = result

            return {
                "split_rule": (
                    "sentence_boundary_"
                    "within_single_sense"
                ),
                "split_sense_index": 1,
                "target_split_ratio": round(
                    single_target_ratio,
                    4,
                ),
                "core_visible_prefix": (
                    core_header
                    + " "
                    + prefix_piece
                ).strip(),
                "core_reference_suffix": (
                    suffix_piece
                ),
                "full_visible_prefix": (
                    full_header
                    + " "
                    + prefix_piece
                ).strip(),
                "full_reference_suffix": (
                    full_blocks[0][
                        len(prefix_piece):
                    ].strip()
                ),
            }

    return None


def build_a_candidates(
    a_builder,
    files: Dict[str, str],
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []

    for volume_name, path in files.items():
        print(
            f"A组读取：{volume_name} -> {path}"
        )

        entries.extend(
            a_builder.parse_structured_entries(
                volume_name,
                path,
            )
        )

    candidates: List[Dict[str, Any]] = []

    for index, entry in enumerate(
        entries,
        start=1,
    ):
        record = a_builder.build_candidate(
            entry,
            index,
        )

        a_builder.evaluate_candidate(
            record,
            MIN_CORE_BODY_LENGTH,
            MIN_VISIBLE_LENGTH,
            MIN_CORE_SUFFIX_LENGTH,
        )

        record["core_length_bin"] = (
            length_bin(
                int(
                    record.get(
                        "core_body_length",
                        0,
                    )
                )
            )
        )

        record["source_group_role"] = (
            "primary_dictionary"
        )

        record[
            "supplement_selection_basis"
        ] = ""

        record["pair_id"] = ""
        record["analysis_subset"] = ""

        candidates.append(record)

    return candidates


def resolve_hanyu_columns(
    headers: Sequence[str],
) -> Dict[str, str]:
    normalized = {
        normalize_header(header): header
        for header in headers
        if normalize_header(header)
    }

    resolved: Dict[str, str] = {}

    for canonical, aliases in HANYU_ALIASES.items():
        for alias in aliases:
            actual = normalized.get(
                normalize_header(alias)
            )

            if actual:
                resolved[canonical] = actual
                break

    missing = {
        "id",
        "word",
        "senseid",
        "sense",
    } - set(resolved)

    if missing:
        raise ValueError(
            "汉语大词典缺少必要字段"
            f"{sorted(missing)}；"
            f"实际列为{list(headers)}"
        )

    for optional in (
        "guangyun",
        "exp",
        "note",
    ):
        resolved.setdefault(
            optional,
            "",
        )

    return resolved


def get_hanyu_value(
    row: Dict[str, Any],
    columns: Dict[str, str],
    field: str,
) -> Any:
    column = columns.get(
        field,
        "",
    )

    if not column:
        return None

    return row.get(column)


def clean_hanyu_field(
    value: Any,
) -> Tuple[str, List[str]]:
    """
    清理汉语大词典字段中的结构标记。
    """
    text = normalize_text(value)
    actions: List[str] = []

    new_text = HANYU_BRACE_TAG_RE.sub(
        r"\1",
        text,
    )

    if new_text != text:
        actions.append(
            "removed_entity_braces"
        )

    text = new_text

    new_text = HANYU_CROSSREF_RE.sub(
        r"\1",
        text,
    )

    if new_text != text:
        actions.append(
            "normalized_cross_reference_marks"
        )

    text = new_text

    new_text = MULTI_PUNCT_RE.sub(
        r"\1",
        text,
    )

    if new_text != text:
        actions.append(
            "collapsed_repeated_punctuation"
        )

    return (
        new_text.strip(),
        actions,
    )


def load_supplement_words(
    path: Optional[str],
) -> Optional[set]:
    """
    读取实际补充词表。

    支持word、字词或词头列。
    """
    if not path:
        return None

    headers, rows = read_first_sheet(path)

    accepted = {
        normalize_header(value)
        for value in (
            "word",
            "字词",
            "字詞",
            "词头",
            "詞頭",
        )
    }

    column = next(
        (
            header
            for header in headers
            if normalize_header(header)
            in accepted
        ),
        "",
    )

    if not column:
        raise ValueError(
            "补充词表中未找到"
            "word/字词/词头列；"
            f"实际列为{headers}"
        )

    result = set()

    for row in rows:
        word, _ = clean_hanyu_field(
            row.get(column)
        )

        if word:
            result.add(word)

    return result


def split_hanyu_blocks(
    rows: Sequence[Dict[str, Any]],
    columns: Dict[str, str],
) -> List[List[Dict[str, Any]]]:
    """
    同一word下，当senseid重新从1开始或下降时，
    拆分为新的词条块。
    """
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            safe_int(
                get_hanyu_value(
                    row,
                    columns,
                    "id",
                )
            )
            or 10**15,
            row.get(
                "_excel_row",
                10**15,
            ),
        ),
    )

    blocks: List[
        List[Dict[str, Any]]
    ] = []

    current: List[Dict[str, Any]] = []

    previous_senseid: Optional[int] = None
    first_guangyun = ""

    for row in sorted_rows:
        senseid = safe_int(
            get_hanyu_value(
                row,
                columns,
                "senseid",
            )
        )

        guangyun, _ = clean_hanyu_field(
            get_hanyu_value(
                row,
                columns,
                "guangyun",
            )
        )

        sense_restarted = (
            bool(current)
            and senseid is not None
            and previous_senseid is not None
            and senseid <= previous_senseid
        )

        guangyun_restarted = (
            bool(current)
            and guangyun
            and first_guangyun
            and guangyun != first_guangyun
            and (
                senseid is None
                or senseid == 1
            )
        )

        if (
            sense_restarted
            or guangyun_restarted
        ):
            blocks.append(current)
            current = []
            previous_senseid = None
            first_guangyun = ""

        current.append(row)

        if (
            guangyun
            and not first_guangyun
        ):
            first_guangyun = guangyun

        if senseid is not None:
            previous_senseid = senseid

    if current:
        blocks.append(current)

    return blocks


def build_b_candidate(
    index: int,
    word: str,
    block_index: int,
    block_count: int,
    rows: Sequence[Dict[str, Any]],
    columns: Dict[str, str],
    source_file: str,
    basis: str,
) -> Dict[str, Any]:
    """
    根据汉语大词典结构化字段重建B组词条。
    """
    guangyun_values: List[str] = []
    senses: List[Dict[str, Any]] = []

    original_ids: List[str] = []
    original_senseids: List[str] = []
    cleaning_actions: List[str] = []

    for row in rows:
        source_id = get_hanyu_value(
            row,
            columns,
            "id",
        )

        senseid = safe_int(
            get_hanyu_value(
                row,
                columns,
                "senseid",
            )
        )

        if senseid is None:
            senseid = len(senses) + 1

        original_ids.append(
            str(source_id or "")
        )

        original_senseids.append(
            str(senseid)
        )

        guangyun, actions = (
            clean_hanyu_field(
                get_hanyu_value(
                    row,
                    columns,
                    "guangyun",
                )
            )
        )

        cleaning_actions.extend(actions)

        if (
            guangyun
            and guangyun
            not in guangyun_values
        ):
            guangyun_values.append(
                guangyun
            )

        sense, actions = clean_hanyu_field(
            get_hanyu_value(
                row,
                columns,
                "sense",
            )
        )

        cleaning_actions.extend(actions)

        exp, actions = clean_hanyu_field(
            get_hanyu_value(
                row,
                columns,
                "exp",
            )
        )

        cleaning_actions.extend(actions)

        note, actions = clean_hanyu_field(
            get_hanyu_value(
                row,
                columns,
                "note",
            )
        )

        cleaning_actions.extend(actions)

        if sense or exp or note:
            senses.append(
                {
                    "source_id": str(
                        source_id or ""
                    ),
                    "senseid": senseid,
                    "sense": sense,
                    "exp": exp,
                    "note": note,
                }
            )

    core_header = f"【{word}】"

    guangyun_text = "".join(
        ensure_terminal_punctuation(
            value
        )
        for value in guangyun_values
    )

    full_header = (
        f"【{word}】 {guangyun_text}".strip()
        if guangyun_text
        else core_header
    )

    core_blocks: List[str] = []
    full_blocks: List[str] = []
    example_text = ""

    for sense in senses:
        label = circled_number(
            sense["senseid"]
        )

        sense_text = (
            ensure_terminal_punctuation(
                sense["sense"]
            )
        )

        exp_text = (
            ensure_terminal_punctuation(
                sense["exp"]
            )
        )

        note_text = (
            ensure_terminal_punctuation(
                sense["note"]
            )
        )

        core_blocks.append(
            label
            + sense_text
            + note_text
        )

        full_blocks.append(
            label
            + sense_text
            + exp_text
            + note_text
        )

        example_text += exp_text

    core_body_text = "".join(
        core_blocks
    )

    entry_core_text = (
        core_header
        + " "
        + core_body_text
    ).strip()

    entry_full_text = (
        full_header
        + "\n\n"
        + "".join(full_blocks)
    ).strip()

    record: Dict[str, Any] = {
        "candidate_id": (
            f"B_CAND_{index:05d}"
        ),
        "sample_id": "",
        "group": "B",
        "source_name": "汉语大词典",
        "source_group_role": (
            "source_comparator"
        ),
        "volume_name": "",
        "source_file": source_file,
        "source_locator": (
            "Hanyu IDs "
            f"{','.join(original_ids)}; "
            "entry block "
            f"{block_index}/{block_count}"
        ),
        "source_url": "",
        "entry_index_in_file": (
            block_index
        ),
        "entry_type": (
            "reconstructed_dictionary_entry"
        ),
        "headword": word,
        "pinyin": "",
        "fanqie": "",
        "rhyme": guangyun_text,
        "simplified": "",
        "radical": "",
        "radical_in_source_text": "No",
        "sense_count": len(senses),
        "source_rows": "",
        "original_record_ids": (
            ",".join(original_ids)
        ),
        "original_senseids": (
            ",".join(original_senseids)
        ),
        "senses_json": json.dumps(
            senses,
            ensure_ascii=False,
        ),
        "entry_header": core_header,
        "full_entry_header": full_header,
        "core_body_text": core_body_text,
        "example_text": example_text,
        "entry_core_text": entry_core_text,
        "entry_full_text": entry_full_text,
        "core_body_length": len(
            core_body_text
        ),
        "core_body_cjk_length": cjk_length(
            core_body_text
        ),
        "example_length": len(
            example_text
        ),
        "full_text_length": len(
            entry_full_text
        ),
        "core_length_bin": length_bin(
            len(core_body_text)
        ),
        "supplement_selection_basis": (
            basis
        ),
        "cleaning_actions": (
            ";".join(
                sorted(
                    set(cleaning_actions)
                )
            )
        ),
        "split_rule": "",
        "split_sense_index": "",
        "target_split_ratio": "",
        "core_visible_prefix": "",
        "core_reference_suffix": "",
        "full_visible_prefix": "",
        "full_reference_suffix": "",
        "core_prefix_length": "",
        "core_suffix_length": "",
        "full_prefix_length": "",
        "full_suffix_length": "",
        "pair_id": "",
        "analysis_subset": "",
        "auto_eligible": "No",
        "manual_verified": "Pending",
        "selected": "No",
        "exclusion_reason": "",
        "notes": (
            "sense+note为核心；"
            "exp为外部引例；"
            "guangyun为音韵信息。"
        ),
    }

    if (
        not word
        or not CJK_RE.search(word)
    ):
        add_exclusion(
            record,
            "invalid_headword",
        )

    if not senses:
        add_exclusion(
            record,
            "missing_sense",
        )

    if (
        len(core_body_text)
        < MIN_CORE_BODY_LENGTH
    ):
        add_exclusion(
            record,
            "core_definition_too_short",
        )

    if "�" in entry_core_text:
        add_exclusion(
            record,
            "replacement_character",
        )

    if not record["exclusion_reason"]:
        split_result = split_blocks(
            record["candidate_id"],
            core_header,
            full_header,
            core_blocks,
            full_blocks,
        )

        if split_result is None:
            add_exclusion(
                record,
                "cannot_create_valid_core_split",
            )

        else:
            record.update(split_result)

            record["core_prefix_length"] = len(
                record["core_visible_prefix"]
            )

            record["core_suffix_length"] = len(
                record["core_reference_suffix"]
            )

            record["full_prefix_length"] = len(
                record["full_visible_prefix"]
            )

            record["full_suffix_length"] = len(
                record["full_reference_suffix"]
            )

    record["auto_eligible"] = (
        "Yes"
        if not record["exclusion_reason"]
        else "No"
    )

    return record


def build_b_candidates(
    path: str,
    a_headwords: set,
    supplement_words: Optional[set],
) -> List[Dict[str, Any]]:
    headers, rows = read_first_sheet(path)
    columns = resolve_hanyu_columns(headers)

    grouped: Dict[
        str,
        List[Dict[str, Any]],
    ] = defaultdict(list)

    for row in rows:
        word, _ = clean_hanyu_field(
            get_hanyu_value(
                row,
                columns,
                "word",
            )
        )

        if (
            word
            and CJK_RE.search(word)
        ):
            grouped[word].append(row)

    candidates: List[Dict[str, Any]] = []
    candidate_index = 1

    for word, word_rows in grouped.items():
        if supplement_words is not None:
            if word not in supplement_words:
                continue

            basis = (
                "actual_supplement_list"
            )

        else:
            if word in a_headwords:
                continue

            basis = (
                "hanyu_only_headword_"
                "fallback_manual_"
                "confirmation_required"
            )

        blocks = split_hanyu_blocks(
            word_rows,
            columns,
        )

        for block_index, block in enumerate(
            blocks,
            start=1,
        ):
            candidate = build_b_candidate(
                candidate_index,
                word,
                block_index,
                len(blocks),
                block,
                columns,
                Path(path).name,
                basis,
            )

            candidates.append(candidate)
            candidate_index += 1

    return candidates


def split_classic_sentences(
    text: str,
) -> List[str]:
    """按完整句末拆分经典正文。"""
    return [
        part
        for part in re.split(
            r"(?<=[。！？；])",
            normalize_text(text),
        )
        if part
    ]

def build_text_windows(
    text: str,
    min_length: int = C_WINDOW_MIN_LENGTH,
    max_length: int = C_WINDOW_MAX_LENGTH,
):
    """
    将经典正文转换为连续文本窗口。

    长文本：
        按句子组合

    短文本：
        使用滑动窗口
    """

    text = normalize_text(text)

    if not text:
        return []


    # ---------- 长文本 ----------
    sentences = split_classic_sentences(text)


    windows = []

    current = ""

    for sent in sentences:

        if (
            len(current)
            + len(sent)
            <= max_length
        ):
            current += sent

        else:

            if len(current)>=min_length:
                windows.append(current)

            current = sent


    if len(current)>=min_length:
        windows.append(current)


    # 如果已经足够
    if windows:
        return windows



    # ---------- 短文本 ----------
    # 孝经、毛诗序进入这里

    results=[]

    size=C_SHORT_WINDOW_SIZE
    step=C_SHORT_WINDOW_STEP


    start=0

    while start < len(text):

        end=min(
            start+size,
            len(text)
        )

        chunk=text[start:end]


        if len(chunk)>=80:
            results.append(chunk)


        if end==len(text):
            break


        start += step


    return results

def build_c_candidates(
    files: Dict[str, str],
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    candidate_index = 1

    for source_name, path in files.items():
        headers, rows = read_first_sheet(path)

        lookup = {
            normalize_header(header): header
            for header in headers
        }

        required_columns = {
            "url",
            "title",
            "paragraph",
        }

        missing = (
            required_columns
            - set(lookup)
        )

        if missing:
            raise ValueError(
                f"{path}必须包含"
                "url、title、paragraph列；"
                f"缺少{sorted(missing)}"
            )

        grouped: Dict[
            Tuple[str, str],
            List[Tuple[int, str]],
        ] = defaultdict(list)

        for row in rows:
            url = str(
                row.get(
                    lookup["url"],
                    "",
                )
                or ""
            ).strip()

            title = normalize_text(
                row.get(
                    lookup["title"]
                )
            )

            paragraph = normalize_text(
                row.get(
                    lookup["paragraph"]
                )
            )

            if paragraph:
                grouped[
                    (
                        url,
                        title,
                    )
                ].append(
                    (
                        row["_excel_row"],
                        paragraph,
                    )
                )

        for (
            url,
            title,
        ), items in grouped.items():
            all_text = ""

            for (
                    excel_row,
                    paragraph,
            ) in items:

                paragraph = normalize_text(
                    paragraph
                )

                if paragraph:
                    all_text += paragraph

            # 根据来源选择策略

            if source_name in [
                "论语",
                "孟子",
            ]:

                units = [
                    (
                        idx,
                        x
                    )
                    for idx, x in enumerate(
                        split_classic_sentences(
                            all_text
                        ),
                        start=1
                    )
                ]


            else:

                # 孝经、毛诗序

                units = []

                window_size = 180

                step = 120

                start = 0

                while start < len(all_text):

                    window = all_text[
                        start:
                        start + window_size
                    ]

                    if len(window) >= 80:
                        units.append(
                            (
                                start,
                                window
                            )
                        )

                    start += step

            current_units: List[str] = []
            current_rows: List[int] = []
            current_length = 0

            def flush_window() -> None:
                nonlocal candidate_index
                nonlocal current_units
                nonlocal current_rows
                nonlocal current_length

                if not current_units:
                    return

                text = "".join(
                    current_units
                ).strip()

                record: Dict[str, Any] = {
                    "candidate_id": (
                        f"C_CAND_"
                        f"{candidate_index:05d}"
                    ),
                    "sample_id": "",
                    "group": "C",
                    "source_name": (
                        source_name
                    ),
                    "source_group_role": (
                        "public_exposure_control"
                    ),
                    "volume_name": "",
                    "source_file": (
                        Path(path).name
                    ),
                    "source_locator": (
                        f"{Path(path).name}: "
                        "Excel rows "
                        f"{min(current_rows)}-"
                        f"{max(current_rows)}"
                    ),
                    "source_url": url,
                    "entry_index_in_file": (
                        candidate_index
                    ),
                    "entry_type": (
                        "public_classic_window"
                    ),
                    "headword": "",
                    "pinyin": "",
                    "fanqie": "",
                    "rhyme": "",
                    "simplified": "",
                    "radical": "",
                    "radical_in_source_text": (
                        "No"
                    ),
                    "sense_count": "",
                    "source_rows": ",".join(
                        map(
                            str,
                            current_rows,
                        )
                    ),
                    "original_record_ids": "",
                    "original_senseids": "",
                    "senses_json": "",
                    "entry_header": title,
                    "full_entry_header": title,
                    "core_body_text": text,
                    "example_text": "",
                    "entry_core_text": text,
                    "entry_full_text": text,
                    "core_body_length": len(
                        text
                    ),
                    "core_body_cjk_length": (
                        cjk_length(text)
                    ),
                    "example_length": 0,
                    "full_text_length": len(
                        text
                    ),
                    "core_length_bin": (
                        length_bin(len(text))
                    ),
                    "supplement_selection_basis": "",
                    "cleaning_actions": "",
                    "split_rule": "",
                    "split_sense_index": "",
                    "target_split_ratio": "",
                    "core_visible_prefix": "",
                    "core_reference_suffix": "",
                    "full_visible_prefix": "",
                    "full_reference_suffix": "",
                    "core_prefix_length": "",
                    "core_suffix_length": "",
                    "full_prefix_length": "",
                    "full_suffix_length": "",
                    "pair_id": "",
                    "analysis_subset": (
                        "public_control"
                    ),
                    "auto_eligible": "No",
                    "manual_verified": (
                        "Pending"
                    ),
                    "selected": "No",
                    "exclusion_reason": "",
                    "notes": (
                        "同一url/title内"
                        "按原顺序构造的"
                        "连续非重叠窗口。"
                    ),
                }

                if (
                    len(text)
                    < C_WINDOW_MIN_LENGTH
                ):
                    add_exclusion(
                        record,
                        "text_too_short",
                    )

                if (
                    len(text)
                    > C_WINDOW_MAX_LENGTH + 60
                ):
                    add_exclusion(
                        record,
                        "window_too_long",
                    )

                if "�" in text:
                    add_exclusion(
                        record,
                        "replacement_character",
                    )

                if not record[
                    "exclusion_reason"
                ]:
                    split_result = (
                        generic_single_text_split(
                            record["candidate_id"],
                            text,
                        )
                    )

                    if split_result is None:
                        add_exclusion(
                            record,
                            "cannot_create_valid_split",
                        )

                    else:
                        (
                            prefix,
                            suffix,
                            target_ratio,
                        ) = split_result

                        record.update(
                            {
                                "split_rule": (
                                    "sentence_boundary"
                                ),
                                "target_split_ratio": round(
                                    target_ratio,
                                    4,
                                ),
                                "core_visible_prefix": (
                                    prefix
                                ),
                                "core_reference_suffix": (
                                    suffix
                                ),
                                "full_visible_prefix": (
                                    prefix
                                ),
                                "full_reference_suffix": (
                                    suffix
                                ),
                                "core_prefix_length": (
                                    len(prefix)
                                ),
                                "core_suffix_length": (
                                    len(suffix)
                                ),
                                "full_prefix_length": (
                                    len(prefix)
                                ),
                                "full_suffix_length": (
                                    len(suffix)
                                ),
                            }
                        )

                record["auto_eligible"] = (
                    "Yes"
                    if not record[
                        "exclusion_reason"
                    ]
                    else "No"
                )

                candidates.append(record)
                candidate_index += 1

                current_units = []
                current_rows = []
                current_length = 0

            for (
                excel_row,
                unit,
            ) in units:
                if (
                    current_units
                    and (
                        current_length
                        >= C_WINDOW_MIN_LENGTH
                    )
                    and (
                        current_length
                        + len(unit)
                        > C_WINDOW_MAX_LENGTH
                    )
                ):
                    flush_window()

                current_units.append(unit)
                current_rows.append(
                    excel_row
                )
                current_length += len(unit)

                if (
                    current_length
                    >= C_WINDOW_MAX_LENGTH
                ):
                    flush_window()

            flush_window()

    return candidates


def stratified_sample(
    records: Sequence[Dict[str, Any]],
    size: int,
    seed_label: str,
) -> List[Dict[str, Any]]:
    """按照核心文本长度分层抽取B组。"""
    eligible = [
        record
        for record in records
        if (
            record.get(
                "auto_eligible"
            )
            == "Yes"
        )
    ]

    if len(eligible) < size:
        raise ValueError(
            f"{seed_label}只有"
            f"{len(eligible)}条合格候选，"
            f"无法抽取{size}条。"
        )

    rng = random.Random(
        f"{SELECTION_SEED}:{seed_label}"
    )

    by_bin: Dict[
        str,
        List[Dict[str, Any]],
    ] = defaultdict(list)

    for record in eligible:
        by_bin[
            record.get(
                "core_length_bin",
                "",
            )
        ].append(record)

    for pool in by_bin.values():
        rng.shuffle(pool)

    bin_names = [
        "L1_60_99",
        "L2_100_159",
        "L3_160_239",
        "L4_240_plus",
    ]

    counts: Dict[str, int] = {}
    assigned = 0

    for bin_name in bin_names:
        count = min(
            len(by_bin[bin_name]),
            round(
                size
                * len(by_bin[bin_name])
                / len(eligible)
            ),
        )

        counts[bin_name] = count
        assigned += count

    while assigned < size:
        possible = [
            bin_name
            for bin_name in bin_names
            if (
                counts[bin_name]
                < len(by_bin[bin_name])
            )
        ]

        if not possible:
            break

        chosen_bin = min(
            possible,
            key=lambda bin_name: (
                counts[bin_name]
                / max(
                    1,
                    len(by_bin[bin_name]),
                )
            ),
        )

        counts[chosen_bin] += 1
        assigned += 1

    selected = [
        record
        for bin_name in bin_names
        for record
        in by_bin[bin_name][
            :counts[bin_name]
        ]
    ]

    if len(selected) < size:
        selected_ids = {
            record["candidate_id"]
            for record in selected
        }

        remaining = [
            record
            for record in eligible
            if record["candidate_id"]
            not in selected_ids
        ]

        rng.shuffle(remaining)

        selected.extend(
            remaining[
                :size - len(selected)
            ]
        )

    return selected[:size]


def proportional_allocations(
    quotas: Dict[str, int],
    target: int,
) -> Dict[str, int]:
    """按照A组卷册配额分配A/B匹配数量。"""
    total = sum(quotas.values())

    raw = {
        key: (
            target
            * value
            / total
        )
        for key, value in quotas.items()
    }

    result = {
        key: min(
            quotas[key],
            int(raw[key]),
        )
        for key in quotas
    }

    while sum(result.values()) < target:
        possible = [
            key
            for key in quotas
            if result[key] < quotas[key]
        ]

        if not possible:
            break

        chosen_key = max(
            possible,
            key=lambda key: (
                raw[key]
                - result[key]
            ),
        )

        result[chosen_key] += 1

    return result


def select_a_with_b_matching(
    a_candidates: Sequence[Dict[str, Any]],
    b_selected: Sequence[Dict[str, Any]],
    quotas: Dict[str, int],
    allow_overlap: bool,
) -> List[Dict[str, Any]]:
    """
    为B组匹配核心文本长度接近的A组，
    再按卷册配额补足A组。
    """
    matched_allocations = (
        proportional_allocations(
            quotas,
            min(
                len(b_selected),
                sum(quotas.values()),
            ),
        )
    )

    b_headwords = {
        record.get(
            "headword",
            "",
        )
        for record in b_selected
        if record.get("headword")
    }

    a_pool = [
        record
        for record in a_candidates
        if (
            record.get(
                "auto_eligible"
            )
            == "Yes"
            and record.get(
                "volume_name"
            )
            in quotas
            and (
                allow_overlap
                or record.get(
                    "headword"
                )
                not in b_headwords
            )
        )
    ]

    selected: List[Dict[str, Any]] = []
    used_ids = set()
    used_keys = set()

    selected_count: Counter = Counter()
    matched_count: Counter = Counter()

    for pair_index, b_record in enumerate(
        b_selected,
        start=1,
    ):
        available_volumes = [
            volume
            for volume in quotas
            if (
                matched_count[volume]
                < matched_allocations[volume]
            )
        ]

        if not available_volumes:
            break

        available_candidates = []

        for a_record in a_pool:
            unique_key = (
                a_record["volume_name"],
                a_record["headword"],
                a_record.get(
                    "pinyin",
                    "",
                ),
                a_record.get(
                    "fanqie",
                    "",
                ),
                a_record.get(
                    "rhyme",
                    "",
                ),
            )

            if (
                a_record["candidate_id"]
                not in used_ids
                and unique_key
                not in used_keys
                and a_record["volume_name"]
                in available_volumes
            ):
                available_candidates.append(
                    (
                        a_record,
                        unique_key,
                    )
                )

        if not available_candidates:
            raise ValueError(
                "A组候选不足，"
                "无法完成A/B长度匹配。"
            )

        (
            chosen_a,
            chosen_key,
        ) = min(
            available_candidates,
            key=lambda item: (
                abs(
                    int(
                        item[0][
                            "core_body_length"
                        ]
                    )
                    - int(
                        b_record[
                            "core_body_length"
                        ]
                    )
                ),
                matched_count[
                    item[0]["volume_name"]
                ],
                item[0]["candidate_id"],
            ),
        )

        pair_id = (
            f"PAIR_{pair_index:03d}"
        )

        chosen_a["pair_id"] = pair_id
        chosen_a["analysis_subset"] = (
            "matched_AB"
        )

        b_record["pair_id"] = pair_id
        b_record["analysis_subset"] = (
            "matched_AB"
        )

        selected.append(chosen_a)
        used_ids.add(
            chosen_a["candidate_id"]
        )
        used_keys.add(chosen_key)

        selected_count[
            chosen_a["volume_name"]
        ] += 1

        matched_count[
            chosen_a["volume_name"]
        ] += 1

    for volume, quota in quotas.items():
        need = (
            quota
            - selected_count[volume]
        )

        if need <= 0:
            continue

        available_candidates = []

        for a_record in a_pool:
            unique_key = (
                a_record["volume_name"],
                a_record["headword"],
                a_record.get(
                    "pinyin",
                    "",
                ),
                a_record.get(
                    "fanqie",
                    "",
                ),
                a_record.get(
                    "rhyme",
                    "",
                ),
            )

            if (
                a_record["volume_name"]
                == volume
                and a_record["candidate_id"]
                not in used_ids
                and unique_key
                not in used_keys
            ):
                available_candidates.append(
                    (
                        a_record,
                        unique_key,
                    )
                )

        rng = random.Random(
            f"{SELECTION_SEED}:"
            f"A_EXTRA:{volume}"
        )

        rng.shuffle(
            available_candidates
        )

        available_candidates.sort(
            key=lambda item: (
                item[0].get("split_rule")
                != "natural_sense_boundary"
            )
        )

        if (
            len(available_candidates)
            < need
        ):
            raise ValueError(
                f"{volume}合格A组"
                "候选不足。"
            )

        for (
            chosen_a,
            chosen_key,
        ) in available_candidates[:need]:
            chosen_a[
                "analysis_subset"
            ] = "A_primary_extra"

            selected.append(chosen_a)
            used_ids.add(
                chosen_a["candidate_id"]
            )
            used_keys.add(chosen_key)

            selected_count[volume] += 1

    if (
        len(selected)
        != sum(quotas.values())
    ):
        raise ValueError(
            "A组最终数量与配额不一致。"
        )

    return selected


def select_c_by_quotas(
    candidates: Sequence[Dict[str, Any]],
    quotas: Dict[str, int],
) -> List[Dict[str, Any]]:
    """
    按来源配额抽取C组，
    并尽量覆盖不同篇章。
    """
    selected: List[Dict[str, Any]] = []

    for source_name, quota in quotas.items():
        pool = [
            record
            for record in candidates
            if (
                record.get(
                    "source_name"
                )
                == source_name
                and record.get(
                    "auto_eligible"
                )
                == "Yes"
            )
        ]

        by_section: Dict[
            str,
            List[Dict[str, Any]],
        ] = defaultdict(list)

        for record in pool:
            by_section[
                record.get(
                    "entry_header",
                    "",
                )
            ].append(record)

        rng = random.Random(
            f"{SELECTION_SEED}:"
            f"C:{source_name}"
        )

        sections = list(by_section)
        rng.shuffle(sections)

        for records in by_section.values():
            rng.shuffle(records)

        chosen: List[Dict[str, Any]] = []
        level = 0

        while len(chosen) < quota:
            progress = False

            for section in sections:
                if (
                    level
                    < len(by_section[section])
                ):
                    chosen.append(
                        by_section[section][level]
                    )

                    progress = True

                    if len(chosen) == quota:
                        break

            if not progress:
                break

            level += 1

        if len(chosen) < quota:
            raise ValueError(
                f"C组{source_name}"
                f"只有{len(chosen)}条，"
                f"无法抽取{quota}条。"
            )

        selected.extend(chosen)

    return selected


def assign_ids(
    a_records: Sequence[Dict[str, Any]],
    b_records: Sequence[Dict[str, Any]],
    c_records: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """生成A001、B001和C001格式的样本编号。"""
    output: List[Dict[str, Any]] = []

    for prefix, records in (
        ("A", a_records),
        ("B", b_records),
        ("C", c_records),
    ):
        for index, record in enumerate(
            records,
            start=1,
        ):
            record["sample_id"] = (
                f"{prefix}{index:03d}"
            )

            record["selected"] = "Yes"
            output.append(record)

    return output


OUTPUT_COLUMNS = [
    "sample_id",
    "candidate_id",
    "group",
    "source_name",
    "source_group_role",
    "volume_name",
    "source_file",
    "source_locator",
    "source_url",
    "entry_index_in_file",
    "entry_type",
    "headword",
    "pinyin",
    "fanqie",
    "rhyme",
    "simplified",
    "radical",
    "radical_in_source_text",
    "sense_count",
    "source_rows",
    "original_record_ids",
    "original_senseids",
    "senses_json",
    "entry_header",
    "full_entry_header",
    "core_body_text",
    "example_text",
    "entry_core_text",
    "entry_full_text",
    "core_body_length",
    "core_body_cjk_length",
    "example_length",
    "full_text_length",
    "core_length_bin",
    "supplement_selection_basis",
    "cleaning_actions",
    "split_rule",
    "split_sense_index",
    "target_split_ratio",
    "core_visible_prefix",
    "core_reference_suffix",
    "full_visible_prefix",
    "full_reference_suffix",
    "core_prefix_length",
    "core_suffix_length",
    "full_prefix_length",
    "full_suffix_length",
    "pair_id",
    "analysis_subset",
    "auto_eligible",
    "manual_verified",
    "selected",
    "exclusion_reason",
    "notes",
]


def clean_for_output(
    record: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if not key.startswith("_")
    }


def safe_excel_value(
    value: Any,
) -> Any:
    if value is None:
        return ""

    if isinstance(
        value,
        (
            dict,
            list,
            tuple,
            set,
        ),
    ):
        value = json.dumps(
            value,
            ensure_ascii=False,
        )

    if isinstance(value, str):
        value = ILLEGAL_CHARACTERS_RE.sub(
            "",
            value,
        )

        if len(value) > 32767:
            value = (
                value[:32764]
                + "..."
            )

    return value


def style_header(
    sheet,
    column_count: int,
) -> None:
    fill = PatternFill(
        fill_type="solid",
        fgColor="1F4E78",
    )

    font = Font(
        bold=True,
        color="FFFFFF",
    )

    alignment = Alignment(
        horizontal="center",
        vertical="center",
        wrap_text=True,
    )

    for cell in sheet[1][:column_count]:
        cell.fill = fill
        cell.font = font
        cell.alignment = alignment


def write_sheet(
    workbook: Workbook,
    name: str,
    records: Sequence[Dict[str, Any]],
    columns: Sequence[str],
) -> None:
    sheet = workbook.create_sheet(
        title=name
    )

    sheet.append(
        list(columns)
    )

    for record in records:
        sheet.append(
            [
                safe_excel_value(
                    record.get(
                        column,
                        "",
                    )
                )
                for column in columns
            ]
        )

    for row in sheet.iter_rows():
        for cell in row:
            cell.alignment = Alignment(
                vertical="top",
                wrap_text=True,
            )

    style_header(
        sheet,
        len(columns),
    )

    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = (
        sheet.dimensions
    )

    width_map = {
        "source_locator": 28,
        "source_url": 32,
        "senses_json": 42,
        "entry_header": 28,
        "full_entry_header": 32,
        "core_body_text": 42,
        "example_text": 42,
        "entry_core_text": 48,
        "entry_full_text": 55,
        "core_visible_prefix": 48,
        "core_reference_suffix": 48,
        "full_visible_prefix": 55,
        "full_reference_suffix": 55,
        "notes": 34,
        "exclusion_reason": 28,
    }

    for index, column in enumerate(
        columns,
        start=1,
    ):
        sheet.column_dimensions[
            get_column_letter(index)
        ].width = width_map.get(
            column,
            14,
        )

    if (
        sheet.max_row > 1
        and "manual_verified" in columns
    ):
        column_number = (
            columns.index(
                "manual_verified"
            )
            + 1
        )

        column_letter = get_column_letter(
            column_number
        )

        validation = DataValidation(
            type="list",
            formula1='"Pending,Pass,Fail"',
            allow_blank=False,
        )

        sheet.add_data_validation(
            validation
        )

        validation.add(
            f"{column_letter}2:"
            f"{column_letter}"
            f"{sheet.max_row}"
        )


def exclusion_summary(
    groups: Dict[
        str,
        Sequence[Dict[str, Any]],
    ],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []

    for group, records in groups.items():
        counter: Counter = Counter()

        for record in records:
            if (
                record.get(
                    "auto_eligible"
                )
                == "Yes"
            ):
                counter[
                    "auto_eligible"
                ] += 1

            else:
                reasons = str(
                    record.get(
                        "exclusion_reason",
                        "",
                    )
                ).split(";")

                for reason in reasons:
                    if reason:
                        counter[reason] += 1

        output.extend(
            {
                "group": group,
                "status_or_reason": reason,
                "count": count,
            }
            for reason, count
            in counter.most_common()
        )

    return output


def codebook() -> List[Dict[str, Any]]:
    definitions = {
        "entry_core_text": (
            "主污染测试文本。"
            "A/B为辞书核心释义；"
            "C为公开经典正文。"
        ),
        "entry_full_text": (
            "辅助完整文本。"
            "A含例句，"
            "B含guangyun/exp，"
            "C与核心文本相同。"
        ),
        "core_visible_prefix": (
            "主实验发送给模型的可见前缀。"
        ),
        "core_reference_suffix": (
            "主实验隐藏参考后缀。"
        ),
        "full_visible_prefix": (
            "辅助完整词条测试前缀。"
        ),
        "full_reference_suffix": (
            "辅助完整词条测试后缀。"
        ),
        "pair_id": (
            "A/B长度匹配对子。"
        ),
        "manual_verified": (
            "正式调用模型前必须"
            "人工改为Pass。"
        ),
    }

    return [
        {
            "field": field,
            "definition": definition,
        }
        for field, definition
        in definitions.items()
    ]


def export_workbook(
    output_path: str,
    a_candidates: Sequence[Dict[str, Any]],
    b_candidates: Sequence[Dict[str, Any]],
    c_candidates: Sequence[Dict[str, Any]],
    selected: Sequence[Dict[str, Any]],
    a_quotas: Dict[str, int],
    c_quotas: Dict[str, int],
) -> None:
    workbook = Workbook()

    summary = workbook.active
    summary.title = "Summary"

    summary_rows = [
        ["项目", "结果"],
        [
            "A组目标",
            sum(a_quotas.values()),
        ],
        [
            "A组来源配额",
            json.dumps(
                a_quotas,
                ensure_ascii=False,
            ),
        ],
        [
            "B组目标",
            sum(
                record["group"] == "B"
                for record in selected
            ),
        ],
        [
            "C组目标",
            sum(c_quotas.values()),
        ],
        [
            "C组来源配额",
            json.dumps(
                c_quotas,
                ensure_ascii=False,
            ),
        ],
        [
            "最终样本总数",
            len(selected),
        ],
        [
            "selection_seed",
            SELECTION_SEED,
        ],
        [
            "split_seed",
            SPLIT_SEED,
        ],
        [
            "A候选/合格",
            (
                f"{len(a_candidates)}/"
                f"{sum(record.get('auto_eligible') == 'Yes' for record in a_candidates)}"
            ),
        ],
        [
            "B候选/合格",
            (
                f"{len(b_candidates)}/"
                f"{sum(record.get('auto_eligible') == 'Yes' for record in b_candidates)}"
            ),
        ],
        [
            "C候选/合格",
            (
                f"{len(c_candidates)}/"
                f"{sum(record.get('auto_eligible') == 'Yes' for record in c_candidates)}"
            ),
        ],
        [
            "主测试字段",
            (
                "core_visible_prefix / "
                "core_reference_suffix"
            ),
        ],
        [
            "辅助测试字段",
            (
                "full_visible_prefix / "
                "full_reference_suffix"
            ),
        ],
        [
            "正式使用条件",
            (
                "selected_100中全部"
                "manual_verified均为Pass"
                "后再调用模型。"
            ),
        ],
    ]

    for row in summary_rows:
        summary.append(
            [
                safe_excel_value(value)
                for value in row
            ]
        )

    for row in summary.iter_rows():
        for cell in row:
            cell.alignment = Alignment(
                vertical="top",
                wrap_text=True,
            )

    style_header(
        summary,
        2,
    )

    summary.freeze_panes = "A2"

    summary.column_dimensions[
        "A"
    ].width = 24

    summary.column_dimensions[
        "B"
    ].width = 72

    write_sheet(
        workbook,
        "A_candidates",
        [
            clean_for_output(record)
            for record in a_candidates
            if (
                record.get(
                    "auto_eligible"
                )
                == "Yes"
            )
        ],
        OUTPUT_COLUMNS,
    )

    write_sheet(
        workbook,
        "B_candidates",
        [
            clean_for_output(record)
            for record in b_candidates
            if (
                record.get(
                    "auto_eligible"
                )
                == "Yes"
            )
        ],
        OUTPUT_COLUMNS,
    )

    write_sheet(
        workbook,
        "C_candidates",
        [
            clean_for_output(record)
            for record in c_candidates
            if (
                record.get(
                    "auto_eligible"
                )
                == "Yes"
            )
        ],
        OUTPUT_COLUMNS,
    )

    selected_sheet_name = (
        "selected_100"
        if len(selected) == 100
        else "selected_final"
    )

    write_sheet(
        workbook,
        selected_sheet_name,
        [
            clean_for_output(record)
            for record in selected
        ],
        OUTPUT_COLUMNS,
    )

    write_sheet(
        workbook,
        "Exclusion_Summary",
        exclusion_summary(
            {
                "A": a_candidates,
                "B": b_candidates,
                "C": c_candidates,
            }
        ),
        [
            "group",
            "status_or_reason",
            "count",
        ],
    )

    write_sheet(
        workbook,
        "Codebook",
        codebook(),
        [
            "field",
            "definition",
        ],
    )

    output = Path(output_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    workbook.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--a-builder",
        default=str(
            Path(__file__).with_name(
                "build_A_structured_contamination.py"
            )
        ),
        help=(
            "build_A_structured_contamination.py"
            "的文件路径。"
        ),
    )

    parser.add_argument(
        "--a-file",
        action="append",
        default=[],
        help=(
            "A组文件，格式："
            "卷册名=Excel路径；"
            "可重复提供。"
        ),
    )

    parser.add_argument(
        "--a-quota",
        default="",
        help=(
            "A组卷册配额。"
            "例如："
            "论语卷=13,孟子卷=13,"
            "孝经卷=12,毛诗序=12"
        ),
    )

    parser.add_argument(
        "--hanyu",
        required=True,
        help=(
            "汉语大词典Excel路径。"
        ),
    )

    parser.add_argument(
        "--supplement-list",
        default="",
        help=(
            "可选：实际汉语大词典"
            "补充词表Excel路径。"
        ),
    )

    parser.add_argument(
        "--b-size",
        type=int,
        default=DEFAULT_B_SIZE,
        help="B组样本数量，默认30。",
    )

    parser.add_argument(
        "--c-file",
        action="append",
        default=[],
        help=(
            "C组文件，格式："
            "来源名=Excel路径；"
            "例如论语=analects.xlsx。"
        ),
    )

    parser.add_argument(
        "--c-quota",
        default="",
        help=(
            "C组来源配额。"
            "默认论语=10,孟子=10。"
        ),
    )

    parser.add_argument(
        "--allow-ab-headword-overlap",
        action="store_true",
        help=(
            "允许A/B最终样本出现"
            "相同词头。"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "contamination_samples_final.xlsx"
        ),
    )

    args = parser.parse_args()

    a_builder = load_a_builder(
        args.a_builder
    )

    a_files = parse_named_paths(
        args.a_file
    )

    c_files = parse_named_paths(
        args.c_file
    )

    a_quotas = parse_quota(
        args.a_quota,
        DEFAULT_A_QUOTAS,
    )

    c_quotas = parse_quota(
        args.c_quota,
        DEFAULT_C_QUOTAS,
    )

    missing_a_files = (
        set(a_quotas)
        - set(a_files)
    )

    if missing_a_files:
        raise ValueError(
            "A组配额中包含未提供文件："
            f"{sorted(missing_a_files)}"
        )

    missing_c_files = (
        set(c_quotas)
        - set(c_files)
    )

    if missing_c_files:
        raise ValueError(
            "C组配额中包含未提供文件："
            f"{sorted(missing_c_files)}"
        )

    a_candidates = build_a_candidates(
        a_builder,
        a_files,
    )

    a_headwords = {
        record.get(
            "headword",
            "",
        )
        for record in a_candidates
        if record.get("headword")
    }

    supplement_words = (
        load_supplement_words(
            args.supplement_list
            or None
        )
    )

    b_candidates = build_b_candidates(
        args.hanyu,
        a_headwords,
        supplement_words,
    )

    c_candidates = build_c_candidates(
        c_files
    )

    selected_b = stratified_sample(
        b_candidates,
        args.b_size,
        "B",
    )

    selected_a = select_a_with_b_matching(
        a_candidates,
        selected_b,
        a_quotas,
        args.allow_ab_headword_overlap,
    )

    selected_c = select_c_by_quotas(
        c_candidates,
        c_quotas,
    )

    selected = assign_ids(
        selected_a,
        selected_b,
        selected_c,
    )

    export_workbook(
        args.output,
        a_candidates,
        b_candidates,
        c_candidates,
        selected,
        a_quotas,
        c_quotas,
    )

    print(
        f"完成：{args.output}；"
        f"A={len(selected_a)}，"
        f"B={len(selected_b)}，"
        f"C={len(selected_c)}，"
        f"总计={len(selected)}"
    )


if __name__ == "__main__":
    main()

