"""
SVC Analysis — PCA on averaged spectral signatures
====================================================

Usage:
    svc-analysis --input data.csv --class-col species --gr plant

Options:
    --input       Path to the input CSV file (required)
    --class-col   Column name for class labels, e.g. species, label (required)
    --gr          Column name to group/average by, e.g. plant, plot (required)
    --n-pcs       Number of PCA components to compute (default: 10)
    --n-std       Confidence ellipse radius in std deviations (default: 2.0 → ~95%)
    --title       Title prefix added to figure filenames and plot titles (optional)

Examples:
    # Default settings (2.0 std ellipse, 10 PCs)
    svc-analysis --input data.csv --class-col species --gr plant

    # With a custom title
    svc-analysis --input data.csv --class-col species --gr plant --title Experiment_01

    # Tighter ellipse (1 std → ~39% confidence)
    svc-analysis --input data.csv --class-col species --gr plant --n-std 1.0

    # Wider ellipse (3 std → ~99% confidence)
    svc-analysis --input data.csv --class-col species --gr plant --n-std 3.0

    # Compute more PCs
    svc-analysis --input data.csv --class-col species --gr plant --n-pcs 20

Output:
    Figures are saved to: svc_output/figures/
        - {title}_pca_PC1_vs_PC2.png       (or pca_PC1_vs_PC2.png if no title)
        - {title}_pca_PC1_vs_PC3.png
        - {title}_pca_explained_variance.png
"""

import argparse
import sys
from .pca import load_and_average, run_pca
from .plots import plot_scores, plot_explained_variance


def main():
    parser = argparse.ArgumentParser(
        description="SVC Spectral Analysis — PCA on averaged spectral signatures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--input",     required=True,  help="Path to input CSV file")
    parser.add_argument("--class-col", required=True,  help="Column name for class labels (e.g. species, label)")
    parser.add_argument("--gr",        required=True,  help="Column name to group/average by (e.g. plant, plot)")
    parser.add_argument("--n-pcs",     type=int,   default=10,  help="Number of PCA components to compute (default: 10)")
    parser.add_argument("--n-std",     type=float, default=2.0, help="Confidence ellipse radius in std deviations (default: 2.0 → ~95%%)")
    parser.add_argument("--title",     default=None,             help="Title prefix added to figure filenames and plot titles (optional)")

    args = parser.parse_args()

    print(f"\n── SVC Analysis ──────────────────────────────────────")
    print(f"  Input file : {args.input}")
    print(f"  Class col  : {args.class_col}")
    print(f"  Group col  : {args.gr}")
    print(f"  Ellipse    : {args.n_std} std ({_std_to_confidence(args.n_std):.0f}% confidence)")
    print(f"  Title      : {args.title if args.title else 'None'}")
    print(f"──────────────────────────────────────────────────────\n")

    # ── Load and average ───────────────────────────────────────────────────────
    print("→ Loading and averaging spectra by group...")
    try:
        averaged_df = load_and_average(args.input, args.gr, args.class_col)
    except ValueError as e:
        print(f"\n[Error] {e}")
        sys.exit(1)

    print(f"  Groups found  : {len(averaged_df)}")
    print(f"  Classes found : {sorted(averaged_df[args.class_col].unique())}\n")

    # ── Run PCA ───────────────────────────────────────────────────────────────
    print("→ Running PCA...")
    try:
        scores_df, explained_variance, pca = run_pca(
            averaged_df, args.class_col, args.gr, n_components=args.n_pcs
        )
    except ValueError as e:
        print(f"\n[Error] {e}")
        sys.exit(1)

    for i, var in enumerate(explained_variance):
        print(f"  PC{i+1}: {var*100:.2f}%")

    # ── Generate plots ─────────────────────────────────────────────────────────
    print("\n→ Saving figures...")
    plot_scores(scores_df, args.class_col, "PC1", "PC2", explained_variance, n_std=args.n_std, title=args.title)
    plot_scores(scores_df, args.class_col, "PC1", "PC3", explained_variance, n_std=args.n_std, title=args.title)
    plot_explained_variance(explained_variance, title=args.title)

    print("\n✅ Done!")


def _std_to_confidence(n_std: float) -> float:
    """Convert number of standard deviations to approximate confidence percentage."""
    from scipy.stats import chi2
    return chi2.cdf(n_std ** 2, df=2) * 100


if __name__ == "__main__":
    main()