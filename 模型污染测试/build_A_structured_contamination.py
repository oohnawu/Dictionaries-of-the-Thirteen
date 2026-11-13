#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
根据校对后、字段拆分整齐的《十三经辞典》Excel构造A组污染测试数据。

预期字段（支持常见别名）：
    字词、拼音、反切音、韵/韵部、释义、词性、例句、简体字、部首

核心设计
========
每个结构化词条同时构造两套文本：

1. entry_core_text（主污染测试）
   - 单字：
       yī 一 於悉切 影質入。質部。

       ①[數]最小的正整數。……
       ②[名]一方面。
   - 多字词：
       【一日】 ①[名]某一天。②比喻程度輕。稍微。……

   不包含例句。拼音、反切和韵部只在单字词条头部出现。
   简体字和部首作为元数据保存，不自动写入辞书原文重建文本。

2. entry_full_text（辅助完整词条测试）
   - 在entry_core_text的每个义项后加入该行对应的例句；
   - 保持“义项序号 + [词性] + 释义 + 例句”的原字段顺序；
   - 单字和多字词分别采用上述两种辞书格式。

重要解释
========
- 正式污染判断以core_reference_suffix为主；
- full_reference_suffix用于辅助判断是否复现了完整词条组织和例句选择；
- 仅复现公开经典例句，不应单独认定为《十三经辞典》来源记忆；
- 本程序只做自动构造和预选，最终50条必须人工核验。
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从校对后的《十三经辞典》结构化Excel构造A组污染测试数据。"""



import argparse
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from openpyxl import Workbook, load_workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation


SELECTION_SEED = 42
SPLIT_SEED = 42

DEFAULT_MIN_CORE_BODY_LENGTH = 60
DEFAULT_MIN_CORE_SUFFIX_LENGTH = 30
DEFAULT_MIN_VISIBLE_LENGTH = 20

DEFAULT_QUOTAS = {
    "论语卷": 25,
    "孟子卷": 25,
}

CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"

CJK_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u2060\ufeff]")
SENTENCE_END_RE = re.compile(r"[。！？；]")

CONTINUATION_MARKERS = {
    "",
    "-",
    "—",
    "–",
    "―",
    "－",
    "～",
    "~",
    "同上",
    "续",
    "續",
}

TERMINAL_PUNCTUATION = set("。！？；：，、」』”’）》】)")

HEADER_ALIASES = {
    "headword": {
        "字词",
        "字詞",
        "词头",
        "詞頭",
        "headword",
        "word",
    },
    "pinyin": {
        "拼音",
        "pinyin",
    },
    "fanqie": {
        "反切音",
        "反切",
        "fanqie",
    },
    "rhyme": {
        "韵",
        "韻",
        "韵部",
        "韻部",
        "rhyme",
    },
    "definition": {
        "释义",
        "釋義",
        "义项",
        "義項",
        "definition",
        "sense",
    },
    "pos": {
        "词性",
        "詞性",
        "pos",
    },
    "example": {
        "例句",
        "引例",
        "example",
        "exp",
    },
    "simplified": {
        "简体字",
        "簡體字",
        "simplified",
    },
    "radical": {
        "部首",
        "radical",
    },
}


def normalize_header(value: Any) -> str:
    """规范Excel表头。"""
    if value is None:
        return ""

    text = unicodedata.normalize("NFKC", str(value))
    text = ZERO_WIDTH_RE.sub("", text)

    return text.strip().lower()


def normalize_text(value: Any) -> str:
    """
    保守规范化文本。

    不进行繁简转换，不删除原有频次和页码定位信息。
    """
    if value is None:
        return ""

    text = unicodedata.normalize("NFKC", str(value))
    text = ZERO_WIDTH_RE.sub("", text)
    text = CONTROL_RE.sub("", text)
    text = text.replace("\u3000", " ")
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_headword(value: Any) -> str:
    """规范词头，并识别空白、破折号和波浪号续行。"""
    text = normalize_text(value)

    if text in CONTINUATION_MARKERS:
        return ""

    text = re.sub(r"^[【\[]|[】\]]$", "", text)

    return text.strip()


def normalize_pos(value: Any) -> str:
    """去除词性字段外层的方括号。"""
    text = normalize_text(value)
    text = re.sub(r"^[\[【]|[\]】]$", "", text)

    return text.strip()


def ensure_terminal_punctuation(text: str) -> str:
    """当字段结尾没有标点时补充句号。"""
    text = text.strip()

    if not text:
        return ""

    if text[-1] in TERMINAL_PUNCTUATION:
        return text

    return text + "。"


def circled_number(index: int) -> str:
    """生成义项序号。"""
    if 1 <= index <= len(CIRCLED):
        return CIRCLED[index - 1]

    return f"({index})"


def cjk_length(text: str) -> int:
    """计算汉字字符数。"""
    return len(CJK_RE.findall(text))


def is_single_character(headword: str) -> bool:
    """判断词头是否为单个汉字。"""
    chars = CJK_RE.findall(headword)

    non_cjk = CJK_RE.sub("", headword)
    non_cjk = re.sub(r"[\s·\-—]", "", non_cjk)

    return len(chars) == 1 and not non_cjk


def read_first_sheet(
    path: str,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """使用openpyxl读取第一个工作表。"""
    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(f"文件不存在：{input_path}")

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
            sheet.iter_rows(values_only=True)
        )
    finally:
        workbook.close()

    if not values:
        raise ValueError(f"文件为空：{input_path}")

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


def resolve_columns(
    headers: Sequence[str],
) -> Dict[str, str]:
    """将实际Excel列名映射为程序内部标准列名。"""
    normalized = {
        normalize_header(header): header
        for header in headers
        if normalize_header(header)
    }

    resolved: Dict[str, str] = {}

    for canonical, aliases in HEADER_ALIASES.items():
        for alias in aliases:
            actual = normalized.get(
                normalize_header(alias)
            )

            if actual:
                resolved[canonical] = actual
                break

    missing = {
        "headword",
        "definition",
    } - set(resolved)

    if missing:
        raise ValueError(
            f"缺少必要列{sorted(missing)}；"
            f"实际列为：{list(headers)}"
        )

    for optional in (
        "pinyin",
        "fanqie",
        "rhyme",
        "pos",
        "example",
        "simplified",
        "radical",
    ):
        resolved.setdefault(optional, "")

    return resolved


def get_value(
    row: Dict[str, Any],
    columns: Dict[str, str],
    field: str,
) -> Any:
    """按照标准字段名读取实际Excel列。"""
    column = columns.get(field, "")

    if not column:
        return None

    return row.get(column)


def metadata_changed(
    current: Dict[str, Any],
    new_metadata: Dict[str, str],
) -> bool:
    """
    当同一词头的拼音、反切或韵部发生变化时，
    将其视为新的词条块。
    """
    for field in (
        "pinyin",
        "fanqie",
        "rhyme",
    ):
        old_value = current.get(field, "")
        new_value = new_metadata.get(field, "")

        if (
            old_value
            and new_value
            and old_value != new_value
        ):
            return True

    return False


def parse_structured_entries(
    volume_name: str,
    path: str,
) -> List[Dict[str, Any]]:
    """
    将结构化Excel中的多行义项组织为完整词条。

    空白、破折号或波浪号词头视为上一词条的续行。
    """
    headers, rows = read_first_sheet(path)
    columns = resolve_columns(headers)

    entries: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    entry_index = 1

    def flush() -> None:
        nonlocal current
        nonlocal entry_index

        if (
            current
            and current["headword"]
            and current["senses"]
        ):
            current["entry_index_in_file"] = entry_index
            entries.append(current)
            entry_index += 1

        current = None

    for row in rows:
        headword = normalize_headword(
            get_value(
                row,
                columns,
                "headword",
            )
        )

        metadata = {
            field: normalize_text(
                get_value(
                    row,
                    columns,
                    field,
                )
            )
            for field in (
                "pinyin",
                "fanqie",
                "rhyme",
                "simplified",
                "radical",
            )
        }

        definition = normalize_text(
            get_value(
                row,
                columns,
                "definition",
            )
        )

        pos = normalize_pos(
            get_value(
                row,
                columns,
                "pos",
            )
        )

        example = normalize_text(
            get_value(
                row,
                columns,
                "example",
            )
        )

        if headword:
            start_new = (
                current is None
                or headword != current["headword"]
                or (
                    current["senses"]
                    and metadata_changed(
                        current,
                        metadata,
                    )
                )
            )

            if start_new:
                flush()

                current = {
                    "volume_name": volume_name,
                    "source_file": Path(path).name,
                    "headword": headword,
                    **metadata,
                    "source_rows": [],
                    "senses": [],
                }

            else:
                for field, value in metadata.items():
                    if (
                        not current.get(field)
                        and value
                    ):
                        current[field] = value

        elif current is None:
            continue

        else:
            for field, value in metadata.items():
                if (
                    not current.get(field)
                    and value
                ):
                    current[field] = value

        if current is None:
            continue

        current["source_rows"].append(
            row["_excel_row"]
        )

        if definition or pos or example:
            sense_key = (
                pos,
                definition,
                example,
            )

            already_exists = any(
                (
                    sense["pos"],
                    sense["definition"],
                    sense["example"],
                )
                == sense_key
                for sense in current["senses"]
            )

            if not already_exists:
                current["senses"].append(
                    {
                        "order": (
                            len(current["senses"])
                            + 1
                        ),
                        "pos": pos,
                        "definition": definition,
                        "example": example,
                        "source_row": row["_excel_row"],
                    }
                )

    flush()

    return entries


def build_header(
    entry: Dict[str, Any],
) -> str:
    """
    按原辞书格式构造词条头。

    单字：
    yī 一 於悉切 影質入。質部。

    多字词：
    【一日】
    """
    if is_single_character(entry["headword"]):
        parts = (
            entry.get("pinyin", ""),
            entry["headword"],
            entry.get("fanqie", ""),
            entry.get("rhyme", ""),
        )

        return " ".join(
            part
            for part in parts
            if part
        )

    return f"【{entry['headword']}】"


def build_sense_blocks(
    entry: Dict[str, Any],
) -> Tuple[
    List[str],
    List[str],
    List[Dict[str, Any]],
]:
    """
    同时生成核心义项文本和完整义项文本。

    核心文本不含例句；
    完整文本在相应义项后加入例句。
    """
    core_blocks: List[str] = []
    full_blocks: List[str] = []
    senses: List[Dict[str, Any]] = []

    sense_count = len(entry["senses"])

    for index, sense in enumerate(
        entry["senses"],
        start=1,
    ):
        label = (
            circled_number(index)
            if sense_count > 1
            else ""
        )

        pos_text = (
            f"[{sense['pos']}]"
            if sense["pos"]
            else ""
        )

        definition = ensure_terminal_punctuation(
            sense["definition"]
        )

        example = sense["example"].strip()

        core_block = (
            f"{label}{pos_text}{definition}"
        )

        full_block = (
            core_block + example
        )

        core_blocks.append(core_block)
        full_blocks.append(full_block)

        senses.append(
            {
                **sense,
                "order": index,
            }
        )

    return (
        core_blocks,
        full_blocks,
        senses,
    )


def join_entry_text(
    header: str,
    blocks: Sequence[str],
    entry_type: str,
) -> str:
    """将词条头和义项正文组织为辞书格式。"""
    body = "".join(blocks)

    if entry_type == "single_character":
        return f"{header}\n\n{body}".strip()

    return f"{header} {body}".strip()


def build_candidate(
    entry: Dict[str, Any],
    candidate_index: int,
) -> Dict[str, Any]:
    """把一个结构化词条转换为A组候选记录。"""
    entry_type = (
        "single_character"
        if is_single_character(entry["headword"])
        else "word"
    )

    header = build_header(entry)

    (
        core_blocks,
        full_blocks,
        senses,
    ) = build_sense_blocks(entry)

    entry_core_text = join_entry_text(
        header,
        core_blocks,
        entry_type,
    )

    entry_full_text = join_entry_text(
        header,
        full_blocks,
        entry_type,
    )

    core_body_text = "".join(core_blocks)

    example_text = "".join(
        sense["example"]
        for sense in senses
        if sense["example"]
    )

    source_rows = entry["source_rows"]

    return {
        "candidate_id": (
            f"A_CAND_{candidate_index:05d}"
        ),
        "sample_id": "",
        "group": "A",
        "source_name": "十三经辞典",
        "volume_name": entry["volume_name"],
        "source_file": entry["source_file"],
        "source_locator": (
            f"{entry['source_file']}: "
            f"Excel rows "
            f"{min(source_rows)}-"
            f"{max(source_rows)}"
        ),
        "entry_index_in_file": (
            entry["entry_index_in_file"]
        ),
        "entry_type": entry_type,
        "headword": entry["headword"],
        "pinyin": entry.get("pinyin", ""),
        "fanqie": entry.get("fanqie", ""),
        "rhyme": entry.get("rhyme", ""),
        "simplified": entry.get(
            "simplified",
            "",
        ),
        "radical": entry.get(
            "radical",
            "",
        ),
        "radical_in_source_text": "No",
        "sense_count": len(senses),
        "source_rows": ",".join(
            map(str, source_rows)
        ),
        "senses_json": json.dumps(
            senses,
            ensure_ascii=False,
        ),
        "entry_header": header,
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
        "auto_eligible": "No",
        "manual_verified": "Pending",
        "selected": "No",
        "exclusion_reason": "",
        "notes": "",
        "_core_blocks": core_blocks,
        "_full_blocks": full_blocks,
        "_senses": senses,
    }


def sentence_boundaries(
    text: str,
) -> List[int]:
    """返回所有完整句末位置。"""
    return [
        match.end()
        for match in SENTENCE_END_RE.finditer(text)
    ]


def try_sense_boundary_split(
    record: Dict[str, Any],
    min_visible: int,
    min_suffix: int,
) -> bool:
    """优先在两个义项之间切分。"""
    core_blocks = record["_core_blocks"]
    full_blocks = record["_full_blocks"]

    if len(core_blocks) < 2:
        return False

    rng = random.Random(
        f"{SPLIT_SEED}:"
        f"{record['candidate_id']}"
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
        core_prefix = join_entry_text(
            record["entry_header"],
            core_blocks[:split_index],
            record["entry_type"],
        )

        core_suffix = "".join(
            core_blocks[split_index:]
        ).strip()

        full_prefix = join_entry_text(
            record["entry_header"],
            full_blocks[:split_index],
            record["entry_type"],
        )

        full_suffix = "".join(
            full_blocks[split_index:]
        ).strip()

        if (
            len(core_prefix) < min_visible
            or len(core_suffix) < min_suffix
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

    if not options:
        return False

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

    record.update(
        {
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
            "core_prefix_length": len(
                core_prefix
            ),
            "core_suffix_length": len(
                core_suffix
            ),
            "full_prefix_length": len(
                full_prefix
            ),
            "full_suffix_length": len(
                full_suffix
            ),
        }
    )

    return True


def try_single_sense_split(
    record: Dict[str, Any],
    min_visible: int,
    min_suffix: int,
) -> bool:
    """
    单义项词条无法在义项间切分时，
    尝试在完整句末切分。
    """
    if len(record["_senses"]) != 1:
        return False

    sense = record["_senses"][0]

    definition = ensure_terminal_punctuation(
        sense["definition"]
    )

    valid_positions = [
        position
        for position in sentence_boundaries(
            definition
        )
        if (
            position >= min_visible
            and (
                len(definition)
                - position
                >= min_suffix
            )
        )
    ]

    if not valid_positions:
        return False

    rng = random.Random(
        f"{SPLIT_SEED}:"
        f"{record['candidate_id']}"
    )

    target_ratio = rng.uniform(
        0.40,
        0.60,
    )

    target_position = round(
        len(definition)
        * target_ratio
    )

    split_index = min(
        valid_positions,
        key=lambda position: abs(
            position
            - target_position
        ),
    )

    pos_text = (
        f"[{sense['pos']}]"
        if sense["pos"]
        else ""
    )

    prefix_piece = (
        pos_text
        + definition[:split_index]
    )

    suffix_piece = (
        definition[split_index:]
        .strip()
    )

    core_prefix = join_entry_text(
        record["entry_header"],
        [prefix_piece],
        record["entry_type"],
    )

    full_prefix = core_prefix

    full_suffix = (
        suffix_piece
        + sense["example"]
    ).strip()

    if (
        len(core_prefix) < min_visible
        or len(suffix_piece) < min_suffix
    ):
        return False

    record.update(
        {
            "split_rule": (
                "sentence_boundary_"
                "within_single_sense"
            ),
            "split_sense_index": 1,
            "target_split_ratio": round(
                target_ratio,
                4,
            ),
            "core_visible_prefix": (
                core_prefix
            ),
            "core_reference_suffix": (
                suffix_piece
            ),
            "full_visible_prefix": (
                full_prefix
            ),
            "full_reference_suffix": (
                full_suffix
            ),
            "core_prefix_length": len(
                core_prefix
            ),
            "core_suffix_length": len(
                suffix_piece
            ),
            "full_prefix_length": len(
                full_prefix
            ),
            "full_suffix_length": len(
                full_suffix
            ),
        }
    )

    return True


def evaluate_candidate(
    record: Dict[str, Any],
    min_core_body: int,
    min_visible: int,
    min_suffix: int,
) -> None:
    """判断候选词条是否能够进入自动合格池。"""
    reasons: List[str] = []

    if (
        not record["headword"]
        or not CJK_RE.search(
            record["headword"]
        )
    ):
        reasons.append(
            "invalid_headword"
        )

    if record["sense_count"] < 1:
        reasons.append(
            "missing_sense"
        )

    if (
        record["core_body_length"]
        < min_core_body
    ):
        reasons.append(
            "core_definition_too_short"
        )

    if "�" in record["entry_core_text"]:
        reasons.append(
            "replacement_character"
        )

    if not reasons:
        split_success = (
            try_sense_boundary_split(
                record,
                min_visible,
                min_suffix,
            )
            or try_single_sense_split(
                record,
                min_visible,
                min_suffix,
            )
        )

        if not split_success:
            reasons.append(
                "cannot_create_valid_core_split"
            )

    record["exclusion_reason"] = (
        ";".join(reasons)
    )

    record["auto_eligible"] = (
        "Yes"
        if not reasons
        else "No"
    )


def parse_quota(
    value: str,
) -> Dict[str, int]:
    """解析卷册配额。"""
    if not value:
        return dict(DEFAULT_QUOTAS)

    quotas: Dict[str, int] = {}

    for item in value.split(","):
        name, count = item.split(
            "=",
            1,
        )

        quotas[name.strip()] = int(
            count.strip()
        )

    return quotas


def parse_named_paths(
    values: Sequence[str],
) -> Dict[str, str]:
    """解析“名称=文件路径”形式的参数。"""
    result: Dict[str, str] = {}

    for value in values:
        if "=" not in value:
            raise ValueError(
                "必须使用"
                f"“卷册名=文件路径”：{value}"
            )

        name, path = value.split(
            "=",
            1,
        )

        result[name.strip()] = (
            path.strip()
        )

    return result


def select_records(
    candidates: Sequence[Dict[str, Any]],
    quotas: Dict[str, int],
) -> List[Dict[str, Any]]:
    """
    按卷册配额选择A组样本。

    优先选择能够在自然义项边界切分的词条。
    """
    selected: List[Dict[str, Any]] = []
    seen_keys = set()

    for volume, quota in quotas.items():
        pool = [
            record
            for record in candidates
            if (
                record["volume_name"]
                == volume
                and record["auto_eligible"]
                == "Yes"
            )
        ]

        natural = [
            record
            for record in pool
            if (
                record["split_rule"]
                == "natural_sense_boundary"
            )
        ]

        fallback = [
            record
            for record in pool
            if (
                record["split_rule"]
                != "natural_sense_boundary"
            )
        ]

        rng = random.Random(
            f"{SELECTION_SEED}:{volume}"
        )

        rng.shuffle(natural)
        rng.shuffle(fallback)

        chosen: List[Dict[str, Any]] = []

        for record in natural + fallback:
            unique_key = (
                record["volume_name"],
                record["headword"],
                record["pinyin"],
                record["fanqie"],
                record["rhyme"],
            )

            if unique_key in seen_keys:
                continue

            chosen.append(record)
            seen_keys.add(unique_key)

            if len(chosen) == quota:
                break

        if len(chosen) < quota:
            raise ValueError(
                f"{volume}只有"
                f"{len(chosen)}条"
                "合格唯一候选，"
                f"无法抽取{quota}条。"
            )

        selected.extend(chosen)

    for index, record in enumerate(
        selected,
        start=1,
    ):
        record["sample_id"] = (
            f"A{index:03d}"
        )

        record["selected"] = "Yes"

    return selected


OUTPUT_COLUMNS = [
    "sample_id",
    "candidate_id",
    "group",
    "source_name",
    "volume_name",
    "source_file",
    "source_locator",
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
    "senses_json",
    "entry_header",
    "core_body_text",
    "example_text",
    "entry_core_text",
    "entry_full_text",
    "core_body_length",
    "core_body_cjk_length",
    "example_length",
    "full_text_length",
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
    "auto_eligible",
    "manual_verified",
    "selected",
    "exclusion_reason",
    "notes",
]


def clean_record_for_output(
    record: Dict[str, Any],
) -> Dict[str, Any]:
    """删除仅供程序内部使用的下划线字段。"""
    return {
        key: value
        for key, value in record.items()
        if not key.startswith("_")
    }


def safe_excel_value(
    value: Any,
) -> Any:
    """转换为openpyxl可以安全写入的单元格值。"""
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
    """设置表头样式。"""
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
    """使用openpyxl写入一个工作表。"""
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
        "sample_id": 11,
        "candidate_id": 15,
        "volume_name": 12,
        "source_file": 18,
        "source_locator": 28,
        "entry_type": 16,
        "headword": 18,
        "pinyin": 14,
        "fanqie": 16,
        "rhyme": 24,
        "simplified": 12,
        "radical": 12,
        "source_rows": 18,
        "senses_json": 45,
        "entry_header": 30,
        "core_body_text": 42,
        "example_text": 45,
        "entry_core_text": 48,
        "entry_full_text": 55,
        "core_visible_prefix": 48,
        "core_reference_suffix": 48,
        "full_visible_prefix": 55,
        "full_reference_suffix": 55,
        "exclusion_reason": 28,
        "notes": 30,
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

        validation.error = (
            "请选择Pending、Pass或Fail。"
        )

        validation.errorTitle = (
            "无效值"
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
    candidates: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """统计A组候选排除原因。"""
    counter: Counter = Counter()

    for record in candidates:
        if record["auto_eligible"] == "Yes":
            counter["auto_eligible"] += 1

        else:
            for reason in record[
                "exclusion_reason"
            ].split(";"):
                if reason:
                    counter[reason] += 1

    return [
        {
            "status_or_reason": reason,
            "count": count,
        }
        for reason, count
        in counter.most_common()
    ]


def codebook() -> List[Dict[str, Any]]:
    """输出核心字段说明。"""
    definitions = {
        "entry_core_text": (
            "不含例句的核心词条文本，"
            "是主污染测试文本。"
        ),
        "entry_full_text": (
            "加入对应例句的完整词条文本，"
            "用于辅助分析。"
        ),
        "core_visible_prefix": (
            "主测试发送给模型的前缀。"
        ),
        "core_reference_suffix": (
            "主测试隐藏参考后缀。"
        ),
        "full_visible_prefix": (
            "完整词条辅助测试前缀。"
        ),
        "full_reference_suffix": (
            "完整词条辅助测试后缀。"
        ),
        "manual_verified": (
            "正式实验前应人工改为Pass。"
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
    candidates: Sequence[Dict[str, Any]],
    selected: Sequence[Dict[str, Any]],
    quotas: Dict[str, int],
    input_files: Dict[str, str],
) -> None:
    """使用openpyxl输出A组候选和预选结果。"""
    workbook = Workbook()

    summary = workbook.active
    summary.title = "Summary"

    summary_rows = [
        ["项目", "结果"],
        [
            "A组正式目标",
            sum(quotas.values()),
        ],
        [
            "来源配额",
            json.dumps(
                quotas,
                ensure_ascii=False,
            ),
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
            "候选词条总数",
            len(candidates),
        ],
        [
            "自动合格候选",
            sum(
                record["auto_eligible"]
                == "Yes"
                for record in candidates
            ),
        ],
        [
            "已预选",
            len(selected),
        ],
        [
            "输入文件",
            json.dumps(
                input_files,
                ensure_ascii=False,
            ),
        ],
        [
            "主测试文本",
            (
                "entry_core_text / "
                "core_reference_suffix"
            ),
        ],
        [
            "辅助测试文本",
            (
                "entry_full_text / "
                "full_reference_suffix"
            ),
        ],
        [
            "人工核验要求",
            (
                "selected_A50中全部样本"
                "正式使用前须将"
                "manual_verified改为Pass。"
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
    ].width = 70

    eligible = [
        clean_record_for_output(record)
        for record in candidates
        if record["auto_eligible"] == "Yes"
    ]

    selected_output = [
        clean_record_for_output(record)
        for record in selected
    ]

    write_sheet(
        workbook,
        "A_candidates",
        eligible,
        OUTPUT_COLUMNS,
    )

    write_sheet(
        workbook,
        "selected_A50",
        selected_output,
        OUTPUT_COLUMNS,
    )

    write_sheet(
        workbook,
        "Exclusion_Summary",
        exclusion_summary(candidates),
        [
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
        "--file",
        action="append",
        default=[],
        help=(
            "卷册名=结构化Excel路径；"
            "可重复提供。"
        ),
    )

    parser.add_argument(
        "--quota",
        default="",
        help=(
            "例如："
            "论语卷=13,孟子卷=13,"
            "孝经卷=12,毛诗序=12"
        ),
    )

    parser.add_argument(
        "--min-core-body",
        type=int,
        default=(
            DEFAULT_MIN_CORE_BODY_LENGTH
        ),
    )

    parser.add_argument(
        "--min-visible",
        type=int,
        default=(
            DEFAULT_MIN_VISIBLE_LENGTH
        ),
    )

    parser.add_argument(
        "--min-core-suffix",
        type=int,
        default=(
            DEFAULT_MIN_CORE_SUFFIX_LENGTH
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "A_group_"
            "contamination_samples.xlsx"
        ),
    )

    args = parser.parse_args()

    input_files = parse_named_paths(
        args.file
    )

    quotas = parse_quota(
        args.quota
    )

    if not input_files:
        raise ValueError(
            "至少需要一个--file参数。"
        )

    missing_files = (
        set(quotas)
        - set(input_files)
    )

    if missing_files:
        raise ValueError(
            "配额中包含未提供的卷册："
            f"{sorted(missing_files)}"
        )

    entries: List[Dict[str, Any]] = []

    for volume_name, path in input_files.items():
        print(
            f"读取：{volume_name} -> {path}"
        )

        entries.extend(
            parse_structured_entries(
                volume_name,
                path,
            )
        )

    candidates: List[Dict[str, Any]] = []

    for index, entry in enumerate(
        entries,
        start=1,
    ):
        record = build_candidate(
            entry,
            index,
        )

        evaluate_candidate(
            record,
            args.min_core_body,
            args.min_visible,
            args.min_core_suffix,
        )

        candidates.append(record)

    selected = select_records(
        candidates,
        quotas,
    )

    export_workbook(
        args.output,
        candidates,
        selected,
        quotas,
        input_files,
    )

    print(
        f"完成：{args.output}；"
        f"候选={len(candidates)}；"
        "合格="
        f"{sum(record['auto_eligible'] == 'Yes' for record in candidates)}；"
        f"预选={len(selected)}"
    )


if __name__ == "__main__":
    main()

