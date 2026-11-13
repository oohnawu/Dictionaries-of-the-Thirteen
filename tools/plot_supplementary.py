"""Plot supplementary figures from released numerical summaries only."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    path = ROOT / "论文图表更新_20模型/generate_paper_figures.py"
    spec = importlib.util.spec_from_file_location("supplementary_plotter", path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.CI_SOURCE = ROOT / "论文图表更新_20模型/图表数据/20模型_准确率_Wilson95CI.csv"
    module.HUMAN_SOURCE = ROOT / "data/summaries/human_evaluation_summary.json"
    module.BASELINE_SOURCE = ROOT / "data/summaries/supplementary_baseline_summary.json"
    module.DATA_DIR = ROOT / "outputs/supplementary_figure_data"
    module.SUPP_FIG_DIR = ROOT / "outputs/supplementary_figures"
    module.MAIN_FIG_DIR = ROOT / "outputs/unused_main_figures"
    module.ensure_dirs()
    module.plot_accuracy_ci()
    module.plot_human_statistics()
    module.plot_baseline_rag()
    print("Supplementary figures saved under outputs/supplementary_figures")


if __name__ == "__main__":
    main()
