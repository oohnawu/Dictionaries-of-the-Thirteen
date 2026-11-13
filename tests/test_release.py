"""Small artificial unit fixtures; these are not experimental expert ratings."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def module(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


class ReleaseTests(unittest.TestCase):
    def test_dataset_integrity(self):
        validate = module("release_validator", "tools/validate_release.py")
        self.assertEqual(validate.validate()["unique_task_instances"], 1200)

    def test_legacy_radical_and_containment(self):
        scorer = module("legacy_test", "补充评测_指标整合/evaluate_added_models.py")
        rows = [{"output": "一部", "prediction": "部首是一"}, {"output": "木部", "prediction": "水"}]
        result = scorer.accuracy_metrics(rows, task="部首识别")
        self.assertEqual(result["correct"], 1)
        self.assertEqual(result["accuracy_percent"], 50)

    def test_prediction_alignment(self):
        wrapper = module("score_wrapper_test", "tools/score_model_outputs.py")
        with self.assertRaises(ValueError):
            wrapper.aligned_records([{"input": "乙", "prediction": "x"}], [{"input": "甲", "output": "y"}])
        records = wrapper.aligned_records([{"prediction": "x", "output": "untrusted"}], [{"input": "甲", "output": "trusted"}])
        self.assertEqual(records[0]["output"], "trusted")

    def test_perfect_ordinal_agreement(self):
        human = module("human_test", "补充人工评测/analyze_human_evaluation.py")
        self.assertAlmostEqual(human.krippendorff_alpha_ordinal([[1, 1, 1], [2, 2, 2], [4, 4, 4]]), 1.0)


if __name__ == "__main__":
    unittest.main()
