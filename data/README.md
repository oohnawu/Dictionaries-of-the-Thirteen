# Evaluation and numerical data

Evaluation inputs are stored in `../eval_data/json`. Each task has 200 records under each of two prompting modes. The mode variants have the same ordered headword/reference pairs. There are 1,200 task-level evaluation instances and 2,400 formatted prompt records, not 2,400 independent evaluation instances. A repeated headword in a contextual task does not necessarily indicate a duplicate instance.

| Original directory | Task |
| --- | --- |
| 部首识别 | radical recognition |
| 词性识别 | POS tagging |
| 拼音注音 | pinyin annotation |
| 反切注音 | fanqie annotation |
| 释义能力 | definition generation |
| 语境理解 | contextual sense disambiguation |

Each JSON array record contains exactly `instruction`, `input` and `output`. The three strings are preserved byte-for-byte in the original JSON files. References follow the source dictionaries and the historical dataset construction; a radical label should not be assumed to represent every dictionary classification system.

`testset_manifest.json` lists counts, unique-headword counts and SHA-256 hashes. `summaries` contains numerical exports, including selected Qwen3-sft rows exported from the first worksheet of archived metric workbooks. Source workbooks are not included. The figure-data directory under `论文图表更新_20模型` contains the main model statistics. Local provenance columns have been removed, not metric values.

`data/restricted` is deliberately absent. Do not create public commits containing complete dictionary resources there. See [data rights](../DATA_RIGHTS.md).
