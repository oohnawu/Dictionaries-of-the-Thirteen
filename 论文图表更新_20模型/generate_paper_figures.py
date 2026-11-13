from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
MAIN_FIG_DIR = ROOT / "论文主图"
SUPP_FIG_DIR = ROOT / "审稿补充图"
DATA_DIR = ROOT / "图表数据"

SOURCE_WORKBOOK = PROJECT / "补充评测_整合结果" / "原始与新增模型指标整合_修正版.xlsx"
CI_SOURCE = PROJECT / "样本量与置信区间统计说明" / "结果文件" / "准确率_Wilson95CI.csv"
HUMAN_SOURCE = PROJECT / "补充人工评测" / "人工专家评测_正式统计结果.json"
BASELINE_SOURCE = PROJECT / "qwen3_8b_rag_formal_server" / "evaluation_results" / "supplementary_baseline_metrics.json"

MODEL_MAPPING = {
    "Qwen3-8B": "Qwen3-8B",
    "Qwen3-14B": "Qwen3-14B",
    "chatglm3-6b": "chatglm3-6b",
    "glm-4-9b-chat": "glm4-9b-chat",
    "Baichuan2-7B-Chat": "Baichuan2-7B-Chat",
    "Baichuan2-13B-Chat": "Baichuan2-13B-Chat",
    "Xunzi-Qwen1.5-7B-Chat": "Xunzi-Qwen1.5-7B-Chat",
    "Llama3-Chinese-8B-Instruct": "Llama3-Chinese-8B-Instruct",
    "MiniCPM3-4B": "MiniCPM3-4B",
    "internlm2_5-7b-chat": "internlm2.5-7b-chat",
    "internlm2_5-20b-chat": "internlm2.5-20b-chat",
    "deepseek-llm-7b-chat": "DeepSeek-llm-7b-chat",
    "deepseek-moe-16b-chat": "DeepSeek-moe-16b-chat",
    "deepseek-r1-250528": "DeepSeek-R1",
    "qwen_api": "Qwen-max",
    "baichuan_api": "Baichuan4",
    "glm_api": "GLM-4",
    "gpt4o_api": "GPT-4o",
    "deepseek-v4-pro": "DeepSeek-V4-Pro",
    "qwen3.7-max": "Qwen3.7-max",
}

MODEL_ORDER = list(MODEL_MAPPING.values())
RAW_MODEL_ORDER = list(MODEL_MAPPING.keys())

# 按用户给出的许可类型：前14个为开源模型，后6个为闭源/API模型。
OPEN_SOURCE_COUNT = 14

TASK_EN = {
    "反切注音": "Fanqie Annotation",
    "拼音注音": "Pinyin Annotation",
    "词性识别": "POS Tagging",
    "部首识别": "Radical Recognition",
    "语境理解": "Contextual Sense Disambiguation",
    "释义生成": "Definition Generation",
}

COLORS = {
    "0-shot": "#2d4069",
    "5-shot": "#e18060",
    "improvement": "#a277af",
    "rouge": "#2d4069",
    "sbert": "#e18060",
    "context": "#2d4069",
    "definition": "#e18060",
}

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 12,
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def ensure_dirs() -> None:
    for path in (MAIN_FIG_DIR, SUPP_FIG_DIR, DATA_DIR):
        path.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def load_main_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    acc = pd.read_excel(SOURCE_WORKBOOK, sheet_name="Accuracy整合")
    rouge = pd.read_excel(SOURCE_WORKBOOK, sheet_name="ROUGE整合")
    sim = pd.read_excel(SOURCE_WORKBOOK, sheet_name="语义相似度整合")

    def prep(df: pd.DataFrame) -> pd.DataFrame:
        out = df[df["模型名"].isin(RAW_MODEL_ORDER) & df["评测模式"].isin(["0-shot", "5-shot"])].copy()
        out["模型显示名"] = out["模型名"].map(MODEL_MAPPING)
        out["模型显示名"] = pd.Categorical(out["模型显示名"], categories=MODEL_ORDER, ordered=True)
        return out.sort_values(["任务名", "评测模式", "模型显示名"]).reset_index(drop=True)

    acc, rouge, sim = prep(acc), prep(rouge), prep(sim)

    expected_acc = len(MODEL_ORDER) * 4 * 2
    expected_sem = len(MODEL_ORDER) * 2 * 2
    assert len(acc) == expected_acc, f"Accuracy rows: expected {expected_acc}, got {len(acc)}"
    assert len(rouge) == expected_sem, f"ROUGE rows: expected {expected_sem}, got {len(rouge)}"
    assert len(sim) == expected_sem, f"Similarity rows: expected {expected_sem}, got {len(sim)}"

    for name, df, tasks in [
        ("Accuracy", acc, ["反切注音", "拼音注音", "词性识别", "部首识别"]),
        ("ROUGE", rouge, ["语境理解", "释义生成"]),
        ("Similarity", sim, ["语境理解", "释义生成"]),
    ]:
        counts = df.groupby(["模型显示名", "任务名", "评测模式"], observed=False).size()
        missing = [
            (model, task, mode)
            for model in MODEL_ORDER
            for task in tasks
            for mode in ("0-shot", "5-shot")
            if counts.get((model, task, mode), 0) != 1
        ]
        assert not missing, f"{name} missing or duplicated cells: {missing[:10]}"

    acc.to_csv(DATA_DIR / "20模型_准确率.csv", index=False, encoding="utf-8-sig")
    rouge.to_csv(DATA_DIR / "20模型_ROUGE.csv", index=False, encoding="utf-8-sig")
    sim.to_csv(DATA_DIR / "20模型_语义相似度.csv", index=False, encoding="utf-8-sig")
    return acc, rouge, sim


def pivot_task(df: pd.DataFrame, task: str, metric: str) -> pd.DataFrame:
    sub = df[df["任务名"] == task].copy()
    pivot = sub.pivot(index="模型显示名", columns="评测模式", values=metric).reindex(MODEL_ORDER)
    assert pivot[["0-shot", "5-shot"]].notna().all().all(), f"Missing value in {task}/{metric}"
    pivot["Improvement"] = pivot["5-shot"] - pivot["0-shot"]
    return pivot


def add_bar_labels(ax: plt.Axes, bars, y_max: float) -> None:
    offset = max(0.15, y_max * 0.006)
    for bar in bars:
        height = float(bar.get_height())
        if math.isnan(height):
            continue
        if height <= max(0.8, y_max * 0.025):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + offset,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                rotation=90,
                color="black",
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + offset,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="black",
            )


def plot_two_task_composite(
    df: pd.DataFrame,
    tasks: list[str],
    metric: str,
    y_label: str,
    improvement_label: str,
    output_name: str,
) -> None:
    pivots = [pivot_task(df, task, metric) for task in tasks]
    ymax = min(100.0, max(float(p[["0-shot", "5-shot"]].max().max()) for p in pivots) * 1.14)
    improvements = np.concatenate([p["Improvement"].to_numpy() for p in pivots])
    xmin = math.floor((float(improvements.min()) - 2.0) / 5.0) * 5.0
    xmax = math.ceil((float(improvements.max()) + 3.0) / 5.0) * 5.0
    if xmin == xmax:
        xmin -= 5
        xmax += 5

    fig = plt.figure(figsize=(24, 20))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.8, 1.2], hspace=0.40, wspace=0.12)
    axes_bar = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    axes_imp = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    x = np.arange(len(MODEL_ORDER))
    width = 0.42

    left_order = pivots[0].sort_values("Improvement", ascending=True).index.tolist()
    for i, (task, pivot) in enumerate(zip(tasks, pivots)):
        ax = axes_bar[i]
        bars0 = ax.bar(
            x - width / 2,
            pivot["0-shot"],
            width,
            color=COLORS["0-shot"],
            edgecolor="black",
            linewidth=1,
            label="0-shot",
        )
        bars5 = ax.bar(
            x + width / 2,
            pivot["5-shot"],
            width,
            color=COLORS["5-shot"],
            edgecolor="black",
            linewidth=1,
            label="5-shot",
        )
        add_bar_labels(ax, bars0, ymax)
        add_bar_labels(ax, bars5, ymax)
        ax.axvline(OPEN_SOURCE_COUNT - 0.5, color="red", linestyle="--", linewidth=2)
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_ORDER, rotation=45, ha="right", fontsize=14)
        ax.set_ylim(0, ymax)
        ax.set_ylabel(y_label if i == 0 else "", fontsize=20, weight="bold")
        ax.set_title(
            f"{TASK_EN[task]} - 0-shot vs 5-shot Performance Comparison",
            fontsize=20,
            pad=20,
            weight="bold",
        )
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        if i == 0:
            ax.legend(
                title="Evaluation Mode",
                bbox_to_anchor=(0.90, 0.98),
                loc="upper left",
                fontsize=16,
                title_fontsize=16,
            )

        imp = pivot.reindex(left_order)
        ax2 = axes_imp[i]
        bars = ax2.barh(
            np.arange(len(imp)),
            imp["Improvement"],
            color=COLORS["improvement"],
            edgecolor="black",
            linewidth=0.5,
            height=0.6,
        )
        for bar in bars:
            value = float(bar.get_width())
            pad = max((xmax - xmin) * 0.008, 0.08)
            xpos = value + pad if value >= 0 else value - pad
            ax2.text(
                xpos,
                bar.get_y() + bar.get_height() / 2,
                f"{value:+.1f}",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=9,
                weight="bold",
            )
        ax2.set_yticks(np.arange(len(imp)))
        ax2.set_yticklabels(imp.index if i == 0 else [""] * len(imp), fontsize=13)
        ax2.set_xlim(xmin, xmax)
        ax2.axvline(0, color="gray", linestyle="--", linewidth=0.7)
        ax2.grid(axis="x", linestyle="--", alpha=0.5)
        ax2.set_title(
            f"{TASK_EN[task]} - {improvement_label} Improvement (5-shot vs 0-shot)",
            fontsize=20,
            pad=18,
            weight="bold",
        )
        ax2.set_xlabel("Increase in Percentage Points (%)", fontsize=20, weight="bold")

    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.05, top=0.95, wspace=0.02, hspace=0.35)
    save_figure(fig, MAIN_FIG_DIR / output_name)


def build_heatmap(acc: pd.DataFrame, rouge: pd.DataFrame, sim: pd.DataFrame) -> pd.DataFrame:
    columns = ["部首识别", "词性识别", "拼音注音", "反切注音", "释义生成", "语境理解"]
    matrix = pd.DataFrame(index=MODEL_ORDER, columns=columns, dtype=float)
    for task in columns[:4]:
        p = pivot_task(acc, task, "准确率（%）")
        matrix[task] = p[["0-shot", "5-shot"]].mean(axis=1)
    for task in columns[4:]:
        rp = pivot_task(rouge, task, "ROUGE-Avg(%)")
        sp = pivot_task(sim, task, "平均语义相似度(%)")
        matrix[task] = (rp[["0-shot", "5-shot"]].mean(axis=1) + sp[["0-shot", "5-shot"]].mean(axis=1)) / 2
    matrix.index.name = "Model Name"
    matrix.to_csv(DATA_DIR / "20模型_六任务热力图得分.csv", encoding="utf-8-sig")
    return matrix


def plot_heatmap(matrix: pd.DataFrame) -> None:
    renamed = matrix.rename(columns={k: TASK_EN[k] for k in matrix.columns})
    fig, ax = plt.subplots(figsize=(15, 13))
    sns.heatmap(
        renamed,
        annot=True,
        fmt=".1f",
        cmap=sns.color_palette("PuRd", as_cmap=True),
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Overall Performance Score (%)"},
        vmin=0,
        vmax=100,
        annot_kws={"size": 15, "weight": "bold"},
        ax=ax,
    )
    ax.axhline(OPEN_SOURCE_COUNT, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Task Type", fontsize=20, weight="bold")
    ax.set_ylabel("Model Name", fontsize=20, weight="bold")
    ax.tick_params(axis="x", labelsize=14, rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.tick_params(axis="y", labelsize=14)
    cbar = ax.collections[0].colorbar
    cbar.set_label("Overall Performance Score (%)", size=20, weight="bold")
    cbar.ax.tick_params(labelsize=16)
    fig.tight_layout()
    save_figure(fig, MAIN_FIG_DIR / "05_大模型多任务性能热力图_20模型")


def plot_accuracy_ci() -> pd.DataFrame:
    ci = pd.read_csv(CI_SOURCE)
    ci = ci[
        ci["模型名"].isin(RAW_MODEL_ORDER)
        & ci["评测模式"].isin(["0-shot", "5-shot"])
        & ci["任务名"].isin(["部首识别", "词性识别", "拼音注音", "反切注音"])
    ].copy()
    ci["模型显示名"] = ci["模型名"].map(MODEL_MAPPING)
    expected = len(MODEL_ORDER) * 4 * 2
    assert len(ci) == expected, f"CI rows: expected {expected}, got {len(ci)}"
    ci.to_csv(DATA_DIR / "20模型_准确率_Wilson95CI.csv", index=False, encoding="utf-8-sig")

    tasks = ["部首识别", "词性识别", "拼音注音", "反切注音"]
    fig, axes = plt.subplots(2, 2, figsize=(22, 19), sharex=True)
    y = np.arange(len(MODEL_ORDER))
    offsets = {"0-shot": -0.16, "5-shot": 0.16}
    for ax, task in zip(axes.flat, tasks):
        for mode in ["0-shot", "5-shot"]:
            sub = ci[(ci["任务名"] == task) & (ci["评测模式"] == mode)].set_index("模型显示名").reindex(MODEL_ORDER)
            means = sub["准确率(%)"].to_numpy(dtype=float)
            low = sub["Wilson95CI下限(%)"].to_numpy(dtype=float)
            high = sub["Wilson95CI上限(%)"].to_numpy(dtype=float)
            ax.errorbar(
                means,
                y + offsets[mode],
                xerr=np.vstack([np.maximum(means - low, 0.0), np.maximum(high - means, 0.0)]),
                fmt="o",
                markersize=5,
                capsize=2,
                linewidth=1.2,
                color=COLORS[mode],
                label=mode,
            )
        ax.axhline(OPEN_SOURCE_COUNT - 0.5, color="red", linestyle="--", linewidth=1.5)
        ax.set_yticks(y)
        ax.set_yticklabels(MODEL_ORDER if ax in axes[:, 0] else [""] * len(MODEL_ORDER), fontsize=11)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        ax.set_title(f"{TASK_EN[task]}: Accuracy with Wilson 95% CI", fontsize=17, weight="bold")
        ax.set_xlabel("Accuracy (%)", fontsize=15, weight="bold")
    axes[0, 0].legend(title="Evaluation Mode", fontsize=12, title_fontsize=12, loc="lower right")
    fig.tight_layout(pad=2.5)
    save_figure(fig, SUPP_FIG_DIR / "06_四项准确率任务_Wilson95CI_20模型")
    return ci


def plot_human_statistics() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = json.loads(HUMAN_SOURCE.read_text(encoding="utf-8"))
    consistency = pd.DataFrame(data["consistency"])
    correlations = pd.DataFrame(data["correlations"])
    consistency.to_csv(DATA_DIR / "人工评测_KrippendorffAlpha.csv", index=False, encoding="utf-8-sig")
    correlations.to_csv(DATA_DIR / "人工评测_Spearman相关与Bootstrap95CI.csv", index=False, encoding="utf-8-sig")

    dim_order = [
        "semantic_correctness",
        "completeness",
        "conciseness",
        "contextual_appropriateness",
        "lexicographical_style",
        "hallucination_severity",
    ]
    dim_labels = ["Semantic\ncorrectness", "Completeness", "Conciseness", "Contextual\nappropriateness", "Lexicographical\nstyle", "Hallucination\nseverity"]
    task_colors = {"语境理解": COLORS["context"], "释义生成": COLORS["definition"]}

    fig, axes = plt.subplots(1, 2, figsize=(22, 8.5))
    x = np.arange(len(dim_order))
    width = 0.38
    for i, task in enumerate(["语境理解", "释义生成"]):
        sub = consistency[consistency["task"] == task].set_index("dimension").reindex(dim_order)
        bars = axes[0].bar(
            x + (i - 0.5) * width,
            sub["krippendorff_alpha_ordinal"],
            width,
            label=TASK_EN[task],
            color=task_colors[task],
            edgecolor="black",
            linewidth=0.7,
        )
        for bar in bars:
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.018,
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
    axes[0].axhline(0.667, color="#666666", linestyle="--", linewidth=1.2, label="0.667 reference")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(dim_labels, fontsize=11)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("Krippendorff's alpha (ordinal)", fontsize=15, weight="bold")
    axes[0].set_title("Inter-rater Reliability by Evaluation Dimension", fontsize=18, weight="bold")
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)
    axes[0].legend(fontsize=11, loc="upper left")

    corr = correlations[
        correlations["human_metric"].eq("human_overall_quality")
        & correlations["automatic_metric"].isin(["rouge_avg_percent", "sbert_similarity"])
    ].copy()
    groups = ["合并", "语境理解", "释义生成"]
    group_labels = ["Combined", "Contextual Sense\nDisambiguation", "Definition\nGeneration"]
    x2 = np.arange(len(groups))
    for i, metric in enumerate(["rouge_avg_percent", "sbert_similarity"]):
        sub = corr[corr["automatic_metric"] == metric].set_index("group").reindex(groups)
        means = sub["spearman_rho"].to_numpy(dtype=float)
        lows = sub["ci95_low"].to_numpy(dtype=float)
        highs = sub["ci95_high"].to_numpy(dtype=float)
        xpos = x2 + (i - 0.5) * width
        bars = axes[1].bar(
            xpos,
            means,
            width,
            color=COLORS["rouge"] if metric == "rouge_avg_percent" else COLORS["sbert"],
            edgecolor="black",
            linewidth=0.7,
            label="ROUGE-Avg" if metric == "rouge_avg_percent" else "Sentence-BERT",
        )
        axes[1].errorbar(
            xpos,
            means,
            yerr=np.vstack([means - lows, highs - means]),
            fmt="none",
            ecolor="black",
            capsize=4,
            linewidth=1.2,
        )
        for bar, value in zip(bars, means):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.025,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=11,
            )
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(group_labels, fontsize=12)
    axes[1].set_ylim(0, 0.75)
    axes[1].set_ylabel("Spearman's rho", fontsize=15, weight="bold")
    axes[1].set_title("Automatic Metrics vs Human Overall Quality\n(Bootstrap 95% CI, 10,000 iterations)", fontsize=18, weight="bold")
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)
    axes[1].legend(fontsize=12, loc="upper left")
    fig.tight_layout(pad=2.0)
    save_figure(fig, SUPP_FIG_DIR / "07_人工评测一致性与自动指标相关性")
    return consistency, correlations


def plot_baseline_rag() -> pd.DataFrame:
    data = json.loads(BASELINE_SOURCE.read_text(encoding="utf-8"))
    acc_metrics = data["reused_accuracy"]["metrics"]
    structured = [
        ("Radical\nopen-book", acc_metrics["radical_openbook"]["accuracy_percent"]),
        ("Pinyin\nopen-book", acc_metrics["pinyin_openbook"]["accuracy_percent"]),
        ("Fanqie\nopen-book", acc_metrics["fanqie_openbook"]["accuracy_percent"]),
        ("POS-B1\nMajority", acc_metrics["pos_majority"]["accuracy_percent"]),
        ("POS-B2\nContextual BM25", acc_metrics["pos_context_bm25"]["accuracy_percent"]),
    ]

    d0 = data["reused_d0"]["metrics"]
    context = [
        ("C0\nClosed-book", d0["context_d0"]["rouge_avg_f1_percent"], d0["context_d0"]["sbert_cosine_percent"]),
        (
            "C1\nDirect BM25",
            data["semantic_reports"]["context_bm25"]["groups"]["all"]["rouge_avg_f1_percent"],
            data["semantic_reports"]["context_bm25"]["groups"]["all"]["sbert_cosine_percent"],
        ),
        (
            "C2\nBM25-based RAG",
            data["semantic_reports"]["context_rag"]["groups"]["all"]["rouge_avg_f1_percent"],
            data["semantic_reports"]["context_rag"]["groups"]["all"]["sbert_cosine_percent"],
        ),
    ]
    definition = [
        ("D0\nClosed-book", d0["definition_d0"]["rouge_avg_f1_percent"], d0["definition_d0"]["sbert_cosine_percent"]),
        (
            "D1\nDirect BM25",
            data["semantic_reports"]["definition_bm25"]["groups"]["all"]["rouge_avg_f1_percent"],
            data["semantic_reports"]["definition_bm25"]["groups"]["all"]["sbert_cosine_percent"],
        ),
    ]
    for key, label in [
        ("definition_d2", "D2\nExample-only"),
        ("definition_d3", "D3\nDefinition+POS"),
        ("definition_d4", "D4\nFull evidence"),
    ]:
        group = data["semantic_reports"][key]["groups"]["all"]
        definition.append((label, group["rouge_avg_f1_percent"], group["sbert_cosine_percent"]))

    rows = []
    for label, value in structured:
        rows.append({"panel": "structured", "configuration": label.replace("\n", " "), "accuracy_percent": value})
    for panel, configurations in (("context", context), ("definition", definition)):
        for label, rouge, sbert in configurations:
            rows.append(
                {
                    "panel": panel,
                    "configuration": label.replace("\n", " "),
                    "rouge_avg_percent": rouge,
                    "sbert_percent": sbert,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(DATA_DIR / "补充基线与RAG图表数据.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(24, 8.5),
        gridspec_kw={"width_ratios": [1.05, 0.75, 1.15]},
    )
    labels = [x[0] for x in structured]
    values = [x[1] for x in structured]
    bars = axes[0].bar(np.arange(len(labels)), values, color=[COLORS["0-shot"], COLORS["0-shot"], COLORS["0-shot"], COLORS["5-shot"], COLORS["5-shot"]], edgecolor="black")
    axes[0].set_xticks(np.arange(len(labels)))
    axes[0].set_xticklabels(labels, fontsize=11)
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Accuracy (%)", fontsize=15, weight="bold")
    axes[0].set_title("Rule-based and Retrieval Baselines", fontsize=18, weight="bold")
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)
    for bar, value in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:.1f}", ha="center", fontsize=11)

    for axis, configurations, title in (
        (axes[1], context, "Contextual Sense Disambiguation"),
        (axes[2], definition, "Definition Generation"),
    ):
        semantic_labels = [item[0] for item in configurations]
        rouge_values = [item[1] for item in configurations]
        sbert_values = [item[2] for item in configurations]
        x = np.arange(len(semantic_labels))
        width = 0.38
        rouge_bars = axis.bar(
            x - width / 2,
            rouge_values,
            width,
            color=COLORS["rouge"],
            edgecolor="black",
            label="ROUGE-Avg",
        )
        sbert_bars = axis.bar(
            x + width / 2,
            sbert_values,
            width,
            color=COLORS["sbert"],
            edgecolor="black",
            label="Sentence-BERT",
        )
        axis.set_xticks(x)
        axis.set_xticklabels(semantic_labels, fontsize=10)
        axis.set_ylim(0, 75)
        axis.set_ylabel("Score (%)", fontsize=15, weight="bold")
        axis.set_title(title, fontsize=17, weight="bold")
        axis.grid(axis="y", linestyle="--", alpha=0.3)
        axis.legend(fontsize=11)
        for semantic_bars in (rouge_bars, sbert_bars):
            for bar in semantic_bars:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{bar.get_height():.1f}",
                    ha="center",
                    fontsize=9,
                    rotation=90,
                )
    fig.tight_layout(pad=2.0)
    save_figure(fig, SUPP_FIG_DIR / "08_补充基线与跨辞书RAG结果")
    return out


def write_manifest(acc: pd.DataFrame, rouge: pd.DataFrame, sim: pd.DataFrame) -> None:
    manifest = {
        "source_workbook": str(SOURCE_WORKBOOK),
        "model_count": len(MODEL_ORDER),
        "model_order": MODEL_ORDER,
        "open_source_models": MODEL_ORDER[:OPEN_SOURCE_COUNT],
        "closed_or_api_models": MODEL_ORDER[OPEN_SOURCE_COUNT:],
        "name_corrections": {"baichuan_api": "Baichuan4", "previous_incorrect_label": "Baichuan3"},
        "row_counts": {"accuracy": len(acc), "rouge": len(rouge), "similarity": len(sim)},
        "expected_row_counts": {"accuracy": 160, "rouge": 80, "similarity": 80},
        "data_complete": True,
        "figure_style": {
            "font": "Times New Roman",
            "zero_shot_color": COLORS["0-shot"],
            "five_shot_color": COLORS["5-shot"],
            "improvement_color": COLORS["improvement"],
            "raster_dpi": 300,
            "vector_format": "SVG",
        },
    }
    (DATA_DIR / "图表数据审计清单.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    ensure_dirs()
    acc, rouge, sim = load_main_data()
    write_manifest(acc, rouge, sim)

    plot_two_task_composite(
        acc,
        ["反切注音", "拼音注音"],
        "准确率（%）",
        "Accuracy (%)",
        "Accuracy",
        "01_反切与拼音注音任务_性能对比与提升_20模型",
    )
    plot_two_task_composite(
        acc,
        ["词性识别", "部首识别"],
        "准确率（%）",
        "Accuracy (%)",
        "Accuracy",
        "02_词性与部首识别任务_性能对比与提升_20模型",
    )
    plot_two_task_composite(
        rouge,
        ["语境理解", "释义生成"],
        "ROUGE-Avg(%)",
        "ROUGE-Avg (%)",
        "ROUGE",
        "03_语境理解与释义生成_ROUGE分数对比与提升_20模型",
    )
    plot_two_task_composite(
        sim,
        ["语境理解", "释义生成"],
        "平均语义相似度(%)",
        "Semantic Similarity (%)",
        "Similarity",
        "04_语境理解与释义生成_语义相似度对比与提升_20模型",
    )
    matrix = build_heatmap(acc, rouge, sim)
    plot_heatmap(matrix)
    plot_accuracy_ci()
    plot_human_statistics()
    plot_baseline_rag()
    print("All figures and data tables generated successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
