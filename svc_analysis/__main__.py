import argparse
import sys
from .pca import load_and_average, run_pca
from .plots import plot_scores, plot_explained_variance


def main():
    parser = argparse.ArgumentParser(
        description="SVC Spectral Analysis — PCA on averaged spectral signatures"
    )
    parser.add_argument("--input",     required=True,  help="Path to input CSV file")
    parser.add_argument("--class-col", required=True,  help="Column name for class labels (e.g. species, label)")
    parser.add_argument("--gr",        required=True,  help="Column name to group/average by (e.g. plant, plot)")
    parser.add_argument("--n-pcs",     type=int,   default=10,  help="Number of PCA components to compute (default: 10)")
    parser.add_argument("--n-std",     type=float, default=2.0, help="Confidence ellipse radius in std deviations (default: 2.0 → ~95%%)")

    args = parser.parse_args()

    print(f"\n── SVC Analysis ──────────────────────────────────────")
    print(f"  Input file : {args.input}")
    print(f"  Class col  : {args.class_col}")
    print(f"  Group col  : {args.gr}")
    print(f"  Ellipse    : {args.n_std} std ({_std_to_confidence(args.n_std):.0f}% confidence)")
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
    plot_scores(scores_df, args.class_col, "PC1", "PC2", explained_variance, n_std=args.n_std)
    plot_scores(scores_df, args.class_col, "PC1", "PC3", explained_variance, n_std=args.n_std)
    plot_explained_variance(explained_variance)

    print("\n✅ Done!")


def _std_to_confidence(n_std: float) -> float:
    """Convert number of standard deviations to approximate confidence percentage."""
    from scipy.stats import chi2
    # For a 2D ellipse, confidence = CDF of chi2 with 2 degrees of freedom
    return chi2.cdf(n_std ** 2, df=2) * 100


if __name__ == "__main__":
    main()