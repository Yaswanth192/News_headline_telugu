"""
src/experiments/plot_learning_curve.py
======================================
Phase 4 — Plot Low-Resource Learning Curves.

Reads outputs/metrics/low_resource_summary.csv and creates:
  - outputs/plots/learning_curve_rouge.png
  - outputs/plots/learning_curve_chrf.png
  - outputs/plots/learning_curve_relevance.png
"""

import pathlib
import sys
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent


def plot_curves():
    csv_path = PROJECT_ROOT / "outputs" / "metrics" / "low_resource_summary.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist.")
        return

    df = pd.read_csv(csv_path)
    df = df.sort_values(by="Proportion (%)")

    out_dir = PROJECT_ROOT / "outputs" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. ROUGE Plot
    plt.figure(figsize=(8, 5))
    plt.plot(df["Proportion (%)"], df["ROUGE-1"], marker="o", linewidth=2, label="ROUGE-1", color="#1f77b4")
    plt.plot(df["Proportion (%)"], df["ROUGE-2"], marker="s", linewidth=2, label="ROUGE-2", color="#ff7f0e")
    plt.plot(df["Proportion (%)"], df["ROUGE-L"], marker="^", linewidth=2, label="ROUGE-L", color="#2ca02c")
    plt.title("Telugu Headline Generation — ROUGE vs Training Data Proportion", fontsize=12, fontweight="bold")
    plt.xlabel("Training Data Proportion (%)", fontsize=11)
    plt.ylabel("Score", fontsize=11)
    plt.xticks(df["Proportion (%)"])
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    rouge_plot = out_dir / "learning_curve_rouge.png"
    plt.savefig(rouge_plot, dpi=300)
    plt.close()
    print(f"Saved: {rouge_plot}")

    # 2. chrF Plot
    plt.figure(figsize=(8, 5))
    plt.plot(df["Proportion (%)"], df["chrF"], marker="D", linewidth=2, color="#d62728", label="chrF")
    plt.title("Telugu Headline Generation — chrF vs Training Data Proportion", fontsize=12, fontweight="bold")
    plt.xlabel("Training Data Proportion (%)", fontsize=11)
    plt.ylabel("chrF Score", fontsize=11)
    plt.xticks(df["Proportion (%)"])
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    chrf_plot = out_dir / "learning_curve_chrf.png"
    plt.savefig(chrf_plot, dpi=300)
    plt.close()
    print(f"Saved: {chrf_plot}")

    # 3. Relevance Stacked Bar Plot
    plt.figure(figsize=(8, 5))
    bar_width = 4
    x = df["Proportion (%)"]
    plt.bar(x, df["HREL Count"], width=bar_width, label="HREL (Factual Main Event)", color="#2ca02c", alpha=0.85)
    plt.bar(x, df["MREL Count"], width=bar_width, bottom=df["HREL Count"], label="MREL (Secondary Event)", color="#ff7f0e", alpha=0.85)
    if "LREL Count" in df.columns:
        plt.bar(x, df["LREL Count"], width=bar_width, bottom=df["HREL Count"] + df["MREL Count"], label="LREL (Misleading/Clickbait)", color="#d62728", alpha=0.85)
    plt.title("Headline Relevance Distribution (mBERT Classifier)", fontsize=12, fontweight="bold")
    plt.xlabel("Training Data Proportion (%)", fontsize=11)
    plt.ylabel("Count of Generated Headlines (out of 50)", fontsize=11)
    plt.xticks(df["Proportion (%)"])
    plt.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.legend()
    plt.tight_layout()
    rel_plot = out_dir / "learning_curve_relevance.png"
    plt.savefig(rel_plot, dpi=300)
    plt.close()
    print(f"Saved: {rel_plot}")


if __name__ == "__main__":
    plot_curves()
