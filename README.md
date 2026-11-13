# Evaluating large language models for Chinese classics lexicography

Evaluation datasets and research code for **Evaluating and adapting large language models for Chinese classics lexicography: a multi-task study**.

The six tasks are radical recognition, POS tagging, pinyin annotation, fanqie annotation, definition generation and contextual sense disambiguation. Retrieval-augmented generation is a comparison configuration, not a new retrieval architecture proposed by this research.

## What is included

- Six evaluation tasks, each with 200 instances. Twelve JSON files provide zero-shot and five-shot prompt variants of the same 1,200 evaluation instances; they are not 2,400 independent samples.
- Lookup, headword-majority and BM25 baselines, and the formal Qwen3-8B RAG comparison code.
- Original-compatible automatic scoring, real-human-rating analysis, overlap and continuation-probe analysis, sample-size statistics and OCR process statistics.
- Numerical model, baseline, human-evaluation and contamination-probe summaries. No individual expert worksheets or comments are included.
- Latest manuscript-style plotting code and public-data plotting entry points.

The original directory names are retained where this avoids altering imports and data alignment. Chinese filenames require a UTF-8-capable environment.

## Quick start

Python 3.10 or newer is recommended. Run commands from the repository root.

```bash
python -m venv .venv
# Activate the virtual environment using the command appropriate to your shell.
python -m pip install -r requirements/core.txt
python tools/validate_release.py
python tools/score_model_outputs.py --help
```

The dependency files describe the released modules. They are **not a reconstructed lockfile of the original experiments**. Sentence-BERT requires `requirements/semantic.txt`; its weights must be installed or downloaded separately. For GPU inference, install a PyTorch build appropriate to your machine.

## Evaluation data

Files are under `eval_data/json/<original task label>/<0-shot or 5-shot>/`. Each contains a JSON array with `instruction`, `input` and `output` strings. `input` is the target headword; contextual information and demonstrations are in `instruction`; `output` is the reference answer.

See [the data guide](data/README.md), [test-set manifest](data/testset_manifest.json) and [data rights notice](DATA_RIGHTS.md). Test files are copied without changing their prompts, order or answers. The depositor confirmed permission to publicly redistribute these test contents. That confirmation is not a new MIT licence for third-party dictionary text.

## Score your model outputs

Create a JSON array in the same order as the selected test file. Each prediction record must contain a `prediction` string. If `input` is supplied, it must match the corresponding reference headword. An example prediction record is `{"input": "目标字头", "prediction": "模型回答"}`.

```bash
python tools/score_model_outputs.py --task radical --predictions my_predictions.json --references "eval_data/json/部首识别/0-shot/部首识别_eval.json" --output outputs/radical_metrics.json

python tools/score_model_outputs.py --task definition --profile legacy --predictions my_predictions.json --references "eval_data/json/释义能力/0-shot/释义能力_eval.json" --skip-sbert --output outputs/definition_metrics.json
```

Remove `--skip-sbert` only after installing semantic dependencies and obtaining `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Use `--sbert-model` for a local model directory. Scoring commands do not call a generation API.

**Historical scoring profiles differ.** `legacy` preserves the original-compatible reference-substring accuracy criterion, including removal of 部 for radical recognition, and the historical generation metric implementation. `rag` preserves the supplementary RAG scorer's preprocessing and multiset ROUGE implementation. Classification with `--profile exact` uses the deterministic baseline's normalised exact comparison. See [scoring details](docs/SCORING.md). Do not describe these as one identical exact-match or ROUGE pipeline.

The archived batch evaluator under `补充评测_指标整合` expects the historical collection of 36 result JSON files. For a new prediction file, use `tools/score_model_outputs.py` instead.

## Baselines and RAG

```bash
python run_six_baselines.py --help
python qwen3_8b_rag_formal_server/build_definition_rag_kb.py --help
python qwen3_8b_rag_formal_server/run_qwen3_8b_rag.py --help
```

Full baselines and RAG require legally obtained dictionary resources. They are not included. Root and formal-directory `baseline_config.json` files point to the excluded `data/restricted/` directory; see [restricted-input schema](docs/RESTRICTED_INPUTS.md).

For the formal API comparison, set `DASHSCOPE_API_KEY` in your shell and run from `qwen3_8b_rag_formal_server` using its own configuration. Do not run a generation command unless you intend to incur API charges. Existing configurations retain `max_new_tokens=512` and `max_input_tokens=8192`; they have not been changed to match a different manuscript claim. Definition evidence top-k is 20 and contextual evidence top-k is 5.

IDs: POS-B1 is headword-majority; POS-B2 is contextual BM25. C0/C1/C2 are closed-book/direct BM25/BM25-based RAG for contextual sense disambiguation. D0/D1 are closed-book/direct BM25; D2 uses examples, contexts and sources; D3 uses definitions, POS labels and sources; D4 uses full evidence. Definition RAG uses DTC evidence, while HDC gold definitions are used for scoring rather than retrieval evidence.

## Human evaluation

Only aggregate human-evaluation results are distributed in `data/summaries/human_evaluation_summary.json`. They describe 1,000 model outputs, three ratings per output and 3,000 rating assignments. Individual worksheets, comments, identities, text-bearing manifests and simulation outputs are excluded. Aggregate tables alone do not allow recalculation of alpha or Spearman coefficients.

Users with lawfully obtained individual rating data can use the archived audit and analysis scripts. Real-rating filename defaults and 10,000 bootstrap resamples are set in the release. Quality dimensions use 1–5; hallucination severity uses 0–2 and is not included in the quality composite. The implementation reports population standard deviations (`ddof=0`). Detailed commands and limitations are in [the reproduction guide](docs/REPRODUCIBILITY.md).

## Plot numerical summaries

```bash
python -m pip install -r requirements/figures.txt
python "论文图表更新_20模型_最新版论文风格/generate_figures_latest_style.py"
python tools/plot_supplementary.py
```

These plotting commands use included numerical exports rather than dictionary text or private expert files. They write new figures only. Model display names are retained as reported, including the correction to Baichuan4; service/checkpoint identifiers should be distinguished from display names.

## Scope and exclusions

No full dictionary scans, OCR corpus, fine-tuning data, RAG index, raw retrieved evidence, model weights, historical API keys or manuscript files are included. The complete original 20-model generation runners and LoRA training launch configuration were not present in the selected archive, so this is not a one-command reproduction of every historical experiment. The continuation-analysis code requires separately obtained probe inputs. Some archived statistics scripts also require excluded source workbooks.

`tools/validate_release.py` checks packaged file hashes, test-set integrity and common secret/path patterns. It does not independently certify copyright permission or guarantee identical output from remote model services.

## Licence

[MIT](LICENSE) applies to author-owned software. Dataset text, numerical data, model weights and third-party dependencies are outside that software licence. See [DATA_RIGHTS.md](DATA_RIGHTS.md). No additional rights to DTC or HDC are granted.

## 中文使用说明

本仓库含六项任务各200条测试样本，zero-shot和five-shot只是同一批样本的两种提示版本。测试JSON按原文件保留；代码MIT许可不覆盖第三方辞书文本。完整辞书、训练数据和检索库未公开。

首次使用先安装 `requirements/core.txt`，运行 `python tools/validate_release.py` 检查数据与文件。计算自己的预测结果请使用 `tools/score_model_outputs.py`。必须区分原模型兼容评分、规范化精确匹配与补充RAG评分，不应将三者混称为同一算法。现存RAG配置保留512输出token，未擅自改成4096。

人工评价只公开汇总表，不包括专家原始评分和评语。绘图使用公开数值数据，可单独运行。任何调用模型API的操作都需要使用者提供自己的密钥，并自行承担费用。本仓库没有一并提供全部20模型原始推理和LoRA训练工程。
