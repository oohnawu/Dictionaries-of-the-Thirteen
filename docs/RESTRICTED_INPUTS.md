# Restricted input schemas

The JSON configurations name resources relative to the excluded `data/restricted/` directory. Place only legally obtained resources there and never commit them.

| Resource filename | Fields used by the archived code |
| --- | --- |
| merged_部首.xlsx | 字词, 部首 |
| merge_拼音&反切.xlsx | 字词, 拼音, 反切音 |
| merged_词性&例句&释义.xlsx | 字词, 词性, 释义, 例句_校准, 包含例句的段落, 书名, 章节 |

Column names are part of the archived input contract. DTC-derived evidence and HDC gold definitions must not be mixed into one retrievable source. The knowledge-base builder refuses HDC source paths and gold-answer keys. The generated index remains restricted even though its builder code is MIT-licensed.

Run formal commands from the formal directory so that its config is selected:

```bash
cd qwen3_8b_rag_formal_server
python build_definition_rag_kb.py --config baseline_config.json
python run_qwen3_8b_rag.py --help
```

The first command builds an index only if the separately supplied resource exists. A generation run is not demonstrated by default because it may incur API charges. Set `DASHSCOPE_API_KEY` in the shell, not in a script or committed `.env`.
