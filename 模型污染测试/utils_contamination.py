# -*- coding:utf-8 -*-

import pandas as pd
import time


INPUT_FILE = "./contamination_samples_final_校对.xlsx"



def get_column_value(row, candidates):

    """
    自动匹配字段名称

    优先使用当前污染测试数据字段：
        core_visible_prefix
        core_reference_suffix

    同时兼容旧版本：
        visible_prefix
        reference_suffix

    """

    for col in candidates:

        if col in row.index:

            return row[col]


    raise KeyError(
        f"Cannot find any column from {candidates}"
    )



def load_samples():

    df = pd.read_excel(
        INPUT_FILE,
        sheet_name="selected_100"
    )


    print(
        "Loaded sheet: selected_100"
    )

    print(
        "Columns:"
    )

    print(
        df.columns.tolist()
    )


    return df



def build_prompt(row, prompt_type):


    prefix = get_column_value(

        row,

        [

            "core_visible_prefix",

            "visible_prefix"

        ]

    )



    source = str(

        row.get(

            "source_name",
            "volume_name"

        )

    )



    headword = str(

        row.get(

            "headword",

            ""

        )

    )



    if prompt_type=="general":


        return f"""
请继续下面的文本。
只输出续写内容，不要解释。

{prefix}
"""



    elif prompt_type=="dictionary":


        return f"""
下面是一条古汉语辞书文本。

请按照原有辞书体例继续文本。

只输出续写内容，不要解释。

词头：
{headword}

文本：
{prefix}
"""



    elif prompt_type=="source":


        return f"""
下面是一段来自{source}的文本。

请尽可能按照原文继续。

只输出续写内容，不要解释。

{prefix}
"""



    else:


        raise ValueError(
            "unknown prompt"
        )



def save_results(results, filename):


    df=pd.DataFrame(results)


    df.to_excel(

        filename,

        index=False

    )



def timestamp():


    return time.strftime(

        "%Y-%m-%d %H:%M:%S"

    )