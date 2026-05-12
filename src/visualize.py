import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from config import FIGS

PRIMARY = "#003049"
ACCENT  = "#F77F00"
GRAY    = "#DADADA"

PALETTE = [PRIMARY, ACCENT, "#C1121F", "#669BBC", "#A8DADC"]
sns.set_theme(style="whitegrid", font_scale=1.0)


def plot_class_distribution(dfs):
    """Bar chart of class distribution across all datasets."""
    fig, axes = plt.subplots(1, len(dfs), figsize=(14, 3.5), sharey=False)
    for ax, (name, df) in zip(axes, dfs.items()):
        counts = df["binary_label"].value_counts().sort_index()
        bars = ax.bar(["Benign", "Attack"], counts.values,
                      color=[PRIMARY, ACCENT], edgecolor="white", width=0.5)
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.set_ylabel("Count" if name == list(dfs.keys())[0] else "")
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
        )
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                    f"{val:,}", ha="center", va="bottom", fontsize=7)
    plt.suptitle("Class Distribution per Dataset", fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = FIGS / "class_distribution.pdf"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out.name}")


def plot_ablation_comparison(results_df):
    """
    Grouped bar chart: F1 score per dataset × ablation, for each model.
    """
    datasets = results_df["dataset"].unique()
    ablations = ["conventional", "entropy_only", "combined"]
    models = results_df["model"].unique()
    colors = {"conventional": GRAY, "entropy_only": ACCENT, "combined": PRIMARY}

    fig, axes = plt.subplots(1, len(models), figsize=(14, 4.5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    x = np.arange(len(datasets))
    w = 0.25

    for ax, model in zip(axes, models):
        for i, abl in enumerate(ablations):
            sub = results_df[(results_df["model"] == model) &
                             (results_df["ablation"] == abl)]
            sub = sub.set_index("dataset").reindex(datasets)
            heights = sub["f1"].values
            errs    = sub["f1_std"].values
            bars = ax.bar(x + (i - 1) * w, heights, w,
                          label=abl, color=colors[abl],
                          edgecolor="white", linewidth=0.5,
                          yerr=errs, capsize=2, error_kw={"linewidth": 0.8})
        ax.set_title(model, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Weighted F1-score" if model == models[0] else "")
        ax.legend(fontsize=8, framealpha=0.7)

    plt.suptitle("Ablation Study: F1 by Feature Set and Dataset",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = FIGS / "ablation_comparison.pdf"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out.name}")


def plot_entropy_distribution(dfs, mdes):
    """
    Violin plots of MDE score per class, across datasets.
    """
    n = len(dfs)
    fig, axes = plt.subplots(1, n, figsize=(14, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (name, df), mde in zip(axes, dfs.items(), mdes.values()):
        if "mde_score" not in mde.columns:
            ax.set_visible(False)
            continue
        plot_df = pd.DataFrame({
            "MDE Score": mde["mde_score"].values,
            "Class": df["binary_label"].map({0: "Benign", 1: "Attack"}).values,
        })
        sns.violinplot(data=plot_df, x="Class", y="MDE Score",
                       palette={"Benign": PRIMARY, "Attack": ACCENT},
                       inner="quartile", linewidth=0.8, ax=ax, cut=0)
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("MDE Score" if name == list(dfs.keys())[0] else "")

    plt.suptitle("MDE Score Distribution: Benign vs Attack",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = FIGS / "entropy_distribution.pdf"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out.name}")


def plot_cross_dataset_entropy_heatmap(dfs, mdes):
    """
    Heatmap: mean entropy feature value per attack class across all datasets
    (using only MDE features that appear in all datasets: mde_score, dir_entropy_bytes,
    dir_entropy_pkts, jsd columns).
    """
    common_feats = ["mde_score"]
    rows = []
    for name, df in dfs.items():
        mde = mdes[name]
        for feat in common_feats:
            if feat not in mde.columns:
                continue
            benign_mean = mde.loc[df["binary_label"] == 0, feat].mean()
            attack_mean = mde.loc[df["binary_label"] == 1, feat].mean()
            rows.append({"Dataset": name, "Feature": feat,
                         "Benign": benign_mean, "Attack": attack_mean,
                         "Delta": attack_mean - benign_mean})

    if not rows:
        return

    heat_df = pd.DataFrame(rows).pivot(index="Dataset", columns="Feature", values="Delta")
    fig, ax = plt.subplots(figsize=(max(4, len(heat_df.columns) * 1.5), 4))
    sns.heatmap(heat_df, annot=True, fmt=".3f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("MDE Score Δ (Attack − Benign) per Dataset",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    out = FIGS / "entropy_delta_heatmap.pdf"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out.name}")


def plot_model_comparison(results_df):
    """
    Final summary: LightGBM vs RF on combined features, F1 per dataset.
    """
    sub = results_df[results_df["ablation"] == "combined"].copy()
    datasets = sub["dataset"].unique()
    x = np.arange(len(datasets))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (model, color) in enumerate(zip(["LightGBM", "RandomForest"],
                                            [PRIMARY, ACCENT])):
        m_df = sub[sub["model"] == model].set_index("dataset").reindex(datasets)
        ax.bar(x + (i - 0.5) * w, m_df["f1"], w,
               label=model, color=color, edgecolor="white",
               yerr=m_df["f1_std"], capsize=3, error_kw={"linewidth": 0.8})

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Weighted F1-score (combined features)")
    ax.set_title("LightGBM vs Random Forest — Combined MDE Feature Set",
                 fontsize=10, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    out = FIGS / "model_comparison.pdf"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out.name}")
