import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def load_and_average(csv_path: str, group_col: str, class_col: str) -> pd.DataFrame:
    """
    Load the CSV, average spectra by group, and retain the class label.
    
    Args:
        csv_path:   Path to the input CSV file
        group_col:  Column name to group by (e.g., 'plant', 'plot')
        class_col:  Column name for the class label (e.g., 'species', 'label')
    
    Returns:
        DataFrame with one row per group, spectral columns averaged,
        and the class label preserved.
    """
    df = pd.read_csv(csv_path)

    # ── Validate columns ───────────────────────────────────────────────────────
    for col in [group_col, class_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in CSV. "
                             f"Available columns: {list(df.columns)}")

    # ── Identify spectral columns (numeric, excluding metadata) ────────────────
    non_spectral = {"Spec_scan", group_col, class_col}
    spectral_cols = [
        c for c in df.columns
        if c not in non_spectral and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not spectral_cols:
        raise ValueError("No numeric spectral columns found in the CSV.")

    # ── Validate one class per group ───────────────────────────────────────────
    class_per_group = df.groupby(group_col)[class_col].nunique()
    ambiguous = class_per_group[class_per_group > 1]
    if not ambiguous.empty:
        raise ValueError(
            f"The following groups have more than one class label, "
            f"which is not allowed:\n{ambiguous}"
        )

    # ── Average spectra by group, keep class label ─────────────────────────────
    class_map = df.groupby(group_col)[class_col].first()
    averaged = df.groupby(group_col)[spectral_cols].mean()
    averaged[class_col] = class_map

    return averaged.reset_index()


def run_pca(averaged_df: pd.DataFrame, class_col: str, group_col: str, n_components: int = 10):
    """
    Standardize spectral data and run PCA.

    Args:
        averaged_df:  Output of load_and_average()
        class_col:    Column name for the class label
        group_col:    Column name for the group
        n_components: Number of PCA components to compute

    Returns:
        scores_df:          DataFrame with PC scores + class + group columns
        explained_variance: Array of explained variance ratios per component
        pca:                Fitted sklearn PCA object
    """
    non_spectral = {class_col, group_col}
    spectral_cols = [
        c for c in averaged_df.columns
        if c not in non_spectral and pd.api.types.is_numeric_dtype(averaged_df[c])
    ]

    X = averaged_df[spectral_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_components = min(n_components, X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_scaled)

    # ── Build scores DataFrame ─────────────────────────────────────────────────
    score_cols = [f"PC{i+1}" for i in range(n_components)]
    scores_df = pd.DataFrame(scores, columns=score_cols)
    scores_df[class_col] = averaged_df[class_col].values
    scores_df[group_col] = averaged_df[group_col].values

    return scores_df, pca.explained_variance_ratio_, pca