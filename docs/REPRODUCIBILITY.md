# Reproduction scope

## What can run from public inputs

Test-set validation, integrity checks, scoring externally supplied predictions, plotting numerical summaries and small code unit tests can run with the relevant dependencies installed. Scoring semantic similarity requires separate model weights. Remote generation is never executed by validation commands.

## What needs additional inputs

Dictionary lookup and RAG require restricted dictionaries. Continuation-probe construction and analysis require the original dictionary/classical-text samples and generated outputs. Original-batch metric integration requires the archived historical result collection and summary workbooks. OCR and corpus statistics require source process records. Those input files are not bundled.

## Real human-rating analysis

Users who possess appropriately authorised individual data can use:

```bash
python "补充人工评测/analyze_human_evaluation.py" --manifest derived_data/human_eval_manifest.json --ratings derived_data/real_human_ratings.json --output derived_data/real_human_analysis_reproduced.json --bootstrap 10000 --seed 20260731
```

Relative manifest, rating and output paths in this archived script are resolved against its own script directory. The input manifest requires a `units` array; ratings require matching unit IDs, pseudonymous expert IDs and six dimension fields. Exactly three ratings per output are validated. Missing source files are an explicit error, not substituted with simulated values.

The release changes the default filenames to real-rating files and the default bootstrap count to 10,000. It does not alter the alpha, composite, standard-deviation or correlation algorithms. It does not distribute individual scores, comments, full manifests or simulations.

## Archived settings

The released RAG configurations retain the archive's 512-token output cap and 8192-token input cap. This release does not claim that every historical inference used one common cap. No complete historical package lock or original LoRA launch project was found in the selected archive. The code provenance manifest records the selected source hashes and bounded changes. These limitations should be distinguished from changes to an evaluation dataset.

## Figure inputs

The latest-style figure script reads the main 20-model CSVs and three released SFT-only numerical CSVs. `tools/plot_supplementary.py` points the archived supplemental plotting functions at sanitised aggregate JSON/CSV inputs. Source metric values are not recomputed during plotting.
