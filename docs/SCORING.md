# Scoring profiles

## Original-compatible profile

The archived v4 evaluator in `补充评测_指标整合/evaluate_added_models.py` preserves the historical implementation. Classification is case-sensitive reference-substring containment, not strict exact matching. Radical recognition removes 部 from both strings first. Generation scoring uses the legacy extraction patterns, zhconv conversion, jieba segmentation retaining punctuation, set n-gram ROUGE F1 and the archived LCS implementation. The legacy denominator includes its epsilon adjustment.

## Supplementary retrieval profile

The supplementary scorer in `qwen3_8b_rag_formal_server/evaluate_supplementary_baselines.py` uses NFKC/OpenCC simplification, punctuation-filtered jieba tokens and multiset n-gram ROUGE F1. These preprocessing and overlap rules are not identical to the historical evaluator. Existing reported results are not silently recalculated or replaced in this release.

## Exact classification profile

The deterministic baseline reports equality after the `simplified_answer_key` transformation: NFKC, OpenCC traditional-to-simplified conversion and whitespace removal. This profile must not be substituted for historical substring results while retaining the same accuracy values.

## Semantic similarity

Both scorer families use the multilingual MiniLM Sentence-BERT model with their respective text extraction and normalisation. Their public wrappers call the archived functions and preserve their clipping/empty-text behaviour. Model weights and software versions can affect numerical reproduction and must be documented for a new run.

## Human statistics

The analysis uses ordinal Krippendorff's alpha per task and dimension. Composite scores are means of the documented dimensions; hallucination is separate. Spearman bootstrap resampling is at the output-row level in the archived implementation; it does not introduce a new prompt-cluster bootstrap. The bootstrap seed is 20260731 with per-analysis offsets, and formal analyses use 10,000 resamples. Standard deviations use `statistics.pstdev`.

The public aggregate human summary is not sufficient to independently reproduce coefficients. Original individual scores and a compatible manifest are required and are not distributed here.
