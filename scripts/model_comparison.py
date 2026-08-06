"""
Generate a side-by-side model comparison chart for the best-performing models.

By default this reads:
  models/saved_artifacts/candidate_results.json

And writes:
  scripts/model_comparison.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "models" / "saved_artifacts" / "candidate_results.json"
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("model_comparison.png")

FONT_TICK = 12
FONT_LABEL = 13
FONT_TITLE = 14

R2_COLORS = {
    "cv": "#c8d4c8",
    "train": "#1a3a3a",
    "holdout": "#2a9d8f",
}
RMSE_COLORS = ["#2a9d8f", "#6a8fad", "#a8bfc8", "#c1714f", "#a89060", "#7f8f9a", "#8f6f5a"]


def _short_model_name(name: str) -> str:
    compact = (
        name.replace(" Regression", "")
        .replace("Gradient Boosting", "Gradient\nBoosting")
        .replace("Random Forest", "Random\nForest")
        .replace("Extra Trees", "Extra\nTrees")
    )
    return compact


def _load_top_models(results_path: Path, top_n: int):
    with results_path.open("r", encoding="utf-8") as file:
        results = json.load(file)

    if not isinstance(results, list) or not results:
        raise ValueError(f"No candidate model results found in {results_path}")

    ranked = sorted(results, key=lambda item: item["holdout_metrics"]["r2"], reverse=True)[:top_n]
    return ranked


def _build_chart_payload(top_models):
    models = [_short_model_name(item["model_name"]) for item in top_models]
    cv_r2 = [item["cv_r2_mean"] for item in top_models]
    train_r2 = [item["train_original_metrics"]["r2"] for item in top_models]
    hold_r2 = [item["holdout_metrics"]["r2"] for item in top_models]
    rmse = [item["holdout_metrics"]["rmse"] for item in top_models]
    rmse_colors = [RMSE_COLORS[idx % len(RMSE_COLORS)] for idx in range(len(top_models))]
    return models, cv_r2, train_r2, hold_r2, rmse, rmse_colors


def create_model_comparison(results_path: Path, output_path: Path, top_n: int):
    top_models = _load_top_models(results_path, top_n)
    models, cv_r2, train_r2, hold_r2, rmse, rmse_colors = _build_chart_payload(top_models)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), facecolor="white")

    x = np.arange(len(models))
    w = 0.30

    # Chart 1: R² grouped bars
    b1 = ax1.bar(x - w, cv_r2, width=w, color=R2_COLORS["cv"], zorder=3)
    b2 = ax1.bar(x, train_r2, width=w, color=R2_COLORS["train"], zorder=3)
    b3 = ax1.bar(x + w, hold_r2, width=w, color=R2_COLORS["holdout"], zorder=3)

    r2_min = min(cv_r2 + train_r2 + hold_r2)
    y_low = max(0.0, min(0.9, r2_min - 0.02))
    ax1.set_title("R² Across Evaluation Stages", fontsize=FONT_TITLE+10, pad=8)
    ax1.set_ylim(y_low, 1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=FONT_TICK+5)
    ax1.tick_params(axis="y", labelsize=FONT_TICK)
    ax1.yaxis.grid(True, color="#e0e0e0", zorder=0)
    ax1.set_axisbelow(True)
    ax1.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax1.tick_params(left=False, bottom=False)

    legend_patches = [
        mpatches.Patch(color=R2_COLORS["cv"], label="CV R²"),
        mpatches.Patch(color=R2_COLORS["train"], label="Train R²"),
        mpatches.Patch(color=R2_COLORS["holdout"], label="Holdout R²"),
    ]
    ax1.legend(
        handles=legend_patches,
        fontsize=FONT_TICK,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=3,
    )

    for bars in (b1, b2, b3):
        for bar in bars:
            val = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                val - 0.004,
                f"{val:.3f}",
                ha="center",
                va="top",
                fontsize=FONT_TICK - 2,
                color="#222",
                zorder=4,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 0.2},
            )

    # Chart 2: Holdout RMSE
    sorted_data = sorted(zip(models, rmse, rmse_colors),
                     key=lambda x: x[1],
                     reverse=True)

    models, rmse, rmse_colors = map(list, zip(*sorted_data))

    x = np.arange(len(models))*1.4
    bars = ax2.bar(x, rmse, width=1.0, color=rmse_colors, zorder=3)
    rmse_max = max(rmse)
    rmse_padding = max(0.002, rmse_max * 0.1)

    for bar, val in zip(bars, rmse):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + rmse_padding * 0.2,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=FONT_TICK,
            color="#222",
        )


    ax2.set_title("Holdout RMSE", fontsize=FONT_TITLE+10, pad=8)
    ax2.set_ylim(0, rmse_max + rmse_padding)
    ax2.set_ylabel("mg/L", fontsize=FONT_LABEL)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=FONT_TICK+5)
    ax2.tick_params(axis="x", pad = 10)
    ax2.tick_params(axis="y", labelsize=FONT_TICK)
    ax2.yaxis.grid(True, color="#e0e0e0", zorder=0)
    ax2.set_axisbelow(True)
    ax2.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax2.tick_params(left=False, bottom=False)

    print(mpl.rcParams["font.family"])
    print(mpl.rcParams["font.sans-serif"])
    plt.rcParams["font.family"] = "Arial"

    plt.tight_layout()
    plt.savefig(output_path, dpi=1000, bbox_inches="tight", facecolor="white", transparent=False)
    if "agg" not in plt.get_backend().lower():
        plt.show()
    print(f"Saved: {output_path}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Create a best-model comparison bar chart.")
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help=f"Path to candidate_results.json (default: {DEFAULT_RESULTS_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path to save the output PNG (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top models (by holdout R²) to plot.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.top_n < 1:
        raise ValueError("--top-n must be at least 1")
    create_model_comparison(args.results, args.output, args.top_n)


if __name__ == "__main__":
    main()
