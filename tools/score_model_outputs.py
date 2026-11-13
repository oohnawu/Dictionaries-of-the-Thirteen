"""Score one prediction JSON using explicitly selected archived scoring rules."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = {"radical": "部首识别", "pos": "词性识别", "pinyin": "拼音注音", "fanqie": "反切注音", "definition": "释义能力", "context": "语境理解"}


def load_module(name: str, relative: str):
    path = ROOT / relative
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def aligned_records(predictions: object, references: object) -> list[dict]:
    if not isinstance(predictions, list) or not isinstance(references, list) or not references:
        raise ValueError("Predictions and non-empty references must be JSON arrays")
    if len(predictions) != len(references):
        raise ValueError("Prediction/reference counts differ")
    aligned = []
    for index, (pred, ref) in enumerate(zip(predictions, references), 1):
        if not isinstance(pred, dict) or not isinstance(ref, dict):
            raise ValueError(f"Record {index} is not an object")
        if not isinstance(pred.get("prediction"), str):
            raise ValueError(f"Record {index} needs a prediction string")
        if not isinstance(ref.get("output"), str) or not ref["output"].strip():
            raise ValueError(f"Reference {index} has no non-empty output")
        if "input" in pred and pred["input"] != ref.get("input"):
            raise ValueError(f"Headword alignment mismatch at record {index}")
        aligned.append({**ref, "prediction": pred["prediction"]})
    return aligned


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--profile", choices=["legacy", "rag", "exact"], default="legacy")
    parser.add_argument("--output", type=Path, default=Path("outputs/metrics.json"))
    parser.add_argument("--skip-sbert", action="store_true")
    parser.add_argument("--sbert-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--sbert-device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.output.resolve() in {args.predictions.resolve(), args.references.resolve()} or (ROOT / "eval_data").resolve() in args.output.resolve().parents:
        raise ValueError("Scoring output must not overwrite predictions or evaluation references")
    records = aligned_records(json.loads(args.predictions.read_text(encoding="utf-8-sig")), json.loads(args.references.read_text(encoding="utf-8-sig")))
    task = TASKS[args.task]
    classification = args.task in {"radical", "pos", "pinyin", "fanqie"}
    summary = {"task": task, "profile": args.profile, "instances": len(records)}
    legacy = load_module("legacy_scorer", "补充评测_指标整合/evaluate_added_models.py")
    if classification:
        if args.profile == "rag":
            raise ValueError("Use legacy or exact for classification tasks")
        if args.profile == "legacy":
            summary["accuracy"] = legacy.accuracy_metrics(records, task=task)
        else:
            common = load_module("exact_common", "baseline_common.py")
            correct = sum(common.simplified_answer_key(r["prediction"]) == common.simplified_answer_key(r["output"]) for r in records)
            summary["accuracy"] = {"correct": correct, "count": len(records), "accuracy_percent": correct / len(records) * 100, "matching_rule": "NFKC + OpenCC t2s + whitespace removal, exact equality"}
    else:
        if args.profile == "exact":
            raise ValueError("Exact is a classification profile")
        if args.profile == "legacy":
            jieba, zhconv = legacy.require_original_dependencies()
            summary["rouge"] = legacy.rouge_metrics(records, jieba_module=jieba, zhconv_module=zhconv)
            if not args.skip_sbert:
                model = legacy.load_sbert(args.sbert_model, device=args.sbert_device)
                summary["sbert"] = legacy.sbert_metrics(records, model=model, zhconv_module=zhconv, batch_size=args.batch_size)
        else:
            rag = load_module("retrieval_scorer", "qwen3_8b_rag_formal_server/evaluate_supplementary_baselines.py")
            rows = [{"prediction": r["prediction"], "reference": r["output"]} for r in records]
            for row in rows:
                rag.add_rouge(row)
            if not args.skip_sbert:
                model = rag.load_sbert(args.sbert_model, device=args.sbert_device, require_device=False)
                rag.add_sbert(rows, model=model, batch_size=args.batch_size)
            summary["metrics"] = rag.average_metrics(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"instances": len(records), "profile": args.profile, "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
