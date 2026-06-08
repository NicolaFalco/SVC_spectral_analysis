import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


# ── Shared style ───────────────────────────────────────────────────────────────
FIGURE_DPI = 300
FIGURE_SIZE = (8, 6)
OUTPUT_DIR = "svc_output/figures"


def _get_color_map(classes):
    """Assign a consistent color to each unique class."""
    unique_classes = sorted(classes.unique())
    cmap = cm.get_cmap("tab10", len(unique_classes))
    return {cls: cmap(i) for i, cls in enumerate(unique_classes)}


def _make_legend(color_map, ax, ellipse_alpha=0.15):
    """Add a class legend to the axes."""
    handles = [
        Line2D([0], [0], marker="o", color="w", label=cls,
               markerfacecolor=color, markersize=8)
        for cls, color in color_map.items()
    ]
    ax.legend(handles=handles, title="Class", framealpha=0.9, loc="best")


def _confidence_ellipse(x, y, ax, color, n_std=2.0, alpha=0.15):
    """
    Draw a confidence ellipse for a set of 2D points.

    Args:
        x, y:  Arrays of x and y coordinates
        ax:    Matplotlib axes to draw on
        color: Fill and edge color
        n_std: Number of standard deviations for the ellipse radius (default: 2 → ~95%)
        alpha: Fill transparency
    """
    if len(x) < 3:
        # Need at least 3 points to compute a meaningful ellipse
        return

    cov = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by largest eigenvalue
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Angle of the ellipse
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

    # Width and height are 2 * n_std * sqrt(eigenvalue)
    width, height = 2 * n_std * np.sqrt(eigenvalues)

    ellipse = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        linewidth=1.5,
        linestyle="--",
        zorder=0        # Draw behind the scatter points
    )
    ax.add_patch(ellipse)


def _save_figure(fig, filename):
    """Save figure to the output directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_scores(scores_df: pd.DataFrame, class_col: str,
                pc_x: str, pc_y: str,
                explained_variance: np.ndarray,
                n_std: float = 2.0):
    """
    Scatter plot of two PCA score components, colored by class,
    with confidence ellipses.

    Args:
        scores_df:          DataFrame with PC scores and class column
        class_col:          Column name for the class label
        pc_x:               PC to plot on X axis (e.g., 'PC1')
        pc_y:               PC to plot on Y axis (e.g., 'PC2')
        explained_variance: Array of explained variance ratios
        n_std:              Number of std deviations for ellipse (default: 2 → ~95%)
    """
    pc_idx = {"PC1": 0, "PC2": 1, "PC3": 2}
    var_x = explained_variance[pc_idx[pc_x]] * 100
    var_y = explained_variance[pc_idx[pc_y]] * 100

    color_map = _get_color_map(scores_df[class_col])

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for cls, group in scores_df.groupby(class_col):
        color = color_map[cls]

        # ── Confidence ellipse ─────────────────────────────────────────────
        _confidence_ellipse(
            group[pc_x].values, group[pc_y].values,
            ax=ax, color=color, n_std=n_std
        )

        # ── Scatter points ─────────────────────────────────────────────────
        ax.scatter(
            group[pc_x], group[pc_y],
            color=color,
            s=60, alpha=0.85, edgecolors="white", linewidths=0.5,
            label=cls, zorder=2
        )

    ax.axhline(0, color="grey", linewidth=0.7, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.7, linestyle="--")
    ax.set_xlabel(f"{pc_x} ({var_x:.1f}%)", fontsize=12)
    ax.set_ylabel(f"{pc_y} ({var_y:.1f}%)", fontsize=12)
    ax.set_title(f"PCA Scores — {pc_x} vs {pc_y}", fontsize=13)
    _make_legend(color_map, ax)

    plt.tight_layout()
    _save_figure(fig, f"pca_{pc_x}_vs_{pc_y}.png")


def plot_explained_variance(explained_variance: np.ndarray):
    """
    Cumulative explained variance plot.

    Args:
        explained_variance: Array of explained variance ratios from PCA
    """
    cumulative = np.cumsum(explained_variance) * 100
    n_components = len(explained_variance)
    pcs = [f"PC{i+1}" for i in range(n_components)]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    ax.plot(pcs, cumulative, marker="o", color="steelblue",
            linewidth=2, markersize=6, markerfacecolor="white",
            markeredgewidth=2)

    # 80% and 95% reference lines
    for threshold, style in [(80, "--"), (95, ":")]:
        ax.axhline(threshold, color="coral", linewidth=1.2,
                   linestyle=style, label=f"{threshold}% variance")

    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Cumulative Explained Variance (%)", fontsize=12)
    ax.set_title("PCA — Cumulative Explained Variance", fontsize=13)
    ax.set_ylim(0, 105)
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    _save_figure(fig, "pca_explained_variance.png")