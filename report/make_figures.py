"""Draw the two report figures from the saved result CSVs (no training).

Run from the repository root: python3 report/make_figures.py
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent / "figures"

# Small figures that fit one column of the two-column paper.
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 7.5, "legend.fontsize": 6.5})


def read_rows(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def learning_curves():
    rows = read_rows(RESULTS / "boosting_learning_curve.csv")
    best = {
        row["depth"]: row
        for row in read_rows(RESULTS / "boosting_depth_comparison.csv")
    }
    # Ordered depths use one blue from light to dark; dashed = training.
    blues = {"1": "#6baed6", "3": "#2171b5", "5": "#08306b"}

    fig, ax = plt.subplots(figsize=(3.4, 1.8))
    for depth, color in blues.items():
        curve = [row for row in rows if row["depth"] == depth]
        trees = [int(row["trees"]) for row in curve]
        ax.plot(
            trees,
            [float(row["validation_loss"]) for row in curve],
            color=color,
            linewidth=1.2,
            label=f"depth {depth}, validation",
        )
        ax.plot(
            trees,
            [float(row["training_loss"]) for row in curve],
            color=color,
            linewidth=1.0,
            linestyle="--",
            label=f"depth {depth}, training",
        )
        # Dot at the number of trees with the lowest validation loss.
        ax.plot(
            int(best[depth]["n_trees"]),
            float(best[depth]["validation_loss"]),
            marker="o",
            markersize=4,
            color=color,
        )
    ax.set_ylim(0.17, 0.235)
    ax.set_xlabel("Number of trees (learning rate 0.1)")
    ax.set_ylabel("Mean logistic loss")
    ax.grid(alpha=0.2)
    # Legend in two rows below the curves: color = depth, line style = data set.
    depth_handles = [
        Line2D([], [], color=color, linewidth=1.2, label=f"depth {depth}")
        for depth, color in blues.items()
    ]
    style_handles = [
        Line2D([], [], color="gray", linewidth=1.2, label="validation"),
        Line2D([], [], color="gray", linewidth=1.0, linestyle="--", label="training"),
    ]
    # Columns are filled top to bottom, so this order gives the rows
    # "depth 1, depth 3, depth 5" and "validation, training".
    handles = [
        depth_handles[0],
        style_handles[0],
        depth_handles[1],
        style_handles[1],
        depth_handles[2],
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="lower left",
        ncol=3,
        columnspacing=1.2,
        handlelength=1.8,
    )
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "learning_curves.pdf")
    plt.close(fig)


def feature_importance(top=10):
    rows = read_rows(RESULTS / "boosting_feature_importance.csv")[:top][::-1]
    shares = [100 * float(row["share"]) for row in rows]

    fig, ax = plt.subplots(figsize=(3.4, 1.6))
    bars = ax.barh(
        [row["feature"] for row in rows], shares, color="#2171b5", height=0.6
    )
    ax.tick_params(axis="y", labelsize=6.5)
    ax.bar_label(
        bars, labels=[f"{share:.1f}%" for share in shares], padding=2, fontsize=6.5
    )
    ax.set_xlim(0, max(shares) * 1.18)
    ax.set_xlabel("Share of total split gain (%)")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "feature_importance.pdf")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    learning_curves()
    feature_importance()
    print("Saved figures in", OUT)
