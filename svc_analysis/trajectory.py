"""
svc_analysis/trajectory.py
==========================
Centroid trajectories in PC space across days.

For each species (level1), fits a single PCA on all days combined, then
computes:
  - treatment-level centroids per day
  - per-plant trajectories for plants present on ≥2 days
  - a completeness table showing how many days each plant was measured
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _spec_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith('w_')]


# ── Main function ──────────────────────────────────────────────────────────────

def compute_trajectories(
    spectra_df:   pd.DataFrame,
    metadata_df:  pd.DataFrame,
    cols:         dict,
    output_dir:   str | Path,
    n_components: int = 10,
) -> dict:
    """
    For each level1 (species), fit a single PCA on all days combined, then
    compute the per-treatment centroid in PC space for each day.

    Fitting PCA on all days combined ensures trajectories are comparable
    across days — each day's data lives in the same PC space.

    Parameters
    ----------
    spectra_df   : stacked spectra DataFrame
    metadata_df  : stacked metadata DataFrame
    cols         : column mapping with keys level1, level2, level3, scan_id, date
    output_dir   : directory for CSV outputs
    n_components : number of PCA components to compute

    Returns
    -------
    results : dict
        results[level1] with keys:
            'scores'             — full scores DataFrame
                                   (scan_id, date, level2, level3, PC1..PCn)
            'centroids'          — centroid DataFrame
                                   (date, level3, PC1..PCn)
            'explained_variance' — array of explained variance ratios
            'pca'                — fitted PCA object
            'scaler'             — fitted StandardScaler
            'plant_trajectories' — per-plant mean scores DataFrame
                                   (date, level2, level3, PC1..PCn)
                                   only plants with ≥2 days
            'completeness'       — DataFrame (level2, level3, n_days, complete)
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    l1_col   = cols['level1']
    l2_col   = cols['level2']
    l3_col   = cols['level3']
    scan_col = cols['scan_id']
    date_col = cols['date']

    merged = spectra_df.merge(metadata_df, on=scan_col, how='inner')
    log.info('Trajectory — merged shape: %s', merged.shape)

    scols         = _spec_cols(merged)
    n_days_total  = merged[date_col].nunique()
    results: dict = {}

    for l1_val, l1_group in merged.groupby(l1_col):
        log.info('── Trajectory  Level1: %s ──────────────────────', l1_val)

        l1_group = l1_group.reset_index(drop=True)
        safe_l1 = str(l1_val).replace(' ', '_').replace('/', '_').replace('\\', '_')

        # ── Fit PCA on all days combined ───────────────────────────────────────
        X         = l1_group[scols].values.astype(float)
        n_comp    = min(n_components, X.shape[0], X.shape[1])
        scaler    = StandardScaler()
        X_scaled  = scaler.fit_transform(X)
        pca       = PCA(n_components=n_comp)
        scores    = pca.fit_transform(X_scaled)

        score_cols = [f'PC{i+1}' for i in range(n_comp)]
        scores_df  = pd.DataFrame(scores, columns=score_cols)
        scores_df[scan_col] = l1_group[scan_col].values
        scores_df[date_col] = l1_group[date_col].values
        scores_df[l2_col]   = l1_group[l2_col].values
        scores_df[l3_col]   = l1_group[l3_col].values

        log.info(
            '   Explained variance PC1=%.1f%%  PC2=%.1f%%',
            pca.explained_variance_ratio_[0] * 100,
            pca.explained_variance_ratio_[1] * 100 if n_comp > 1 else 0,
        )

        # ── Treatment centroids per day ────────────────────────────────────────
        centroids = (
            scores_df
            .groupby([date_col, l3_col])[score_cols]
            .mean()
            .reset_index()
            .sort_values(date_col)
        )

        # ── Per-plant trajectories (plants with ≥2 days only) ──────────────────
        plant_days      = scores_df.groupby(l2_col)[date_col].nunique()
        eligible_plants = plant_days[plant_days >= 2].index

        log.info(
            '   Plants with ≥2 days: %d / %d',
            len(eligible_plants),
            plant_days.shape[0],
        )

        plant_traj = (
            scores_df[scores_df[l2_col].isin(eligible_plants)]
            .groupby([date_col, l2_col, l3_col])[score_cols]
            .mean()
            .reset_index()
            .sort_values([l2_col, date_col])
        )

        # ── Completeness table ─────────────────────────────────────────────────
        completeness = (
            scores_df
            .groupby(l2_col)[date_col]
            .nunique()
            .reset_index()
            .rename(columns={date_col: 'n_days'})
        )
        completeness['complete'] = completeness['n_days'] == n_days_total
        completeness = completeness.merge(
            l1_group[[l2_col, l3_col]].drop_duplicates(),
            on  = l2_col,
            how = 'left',
        )

        # ── Explained variance DataFrame ───────────────────────────────────────
        ev_df = pd.DataFrame({
            'PC':                 score_cols,
            'explained_variance': pca.explained_variance_ratio_,
            'cumulative':         np.cumsum(pca.explained_variance_ratio_),
        })

        # ── Save CSVs ──────────────────────────────────────────────────────────
        scores_df.to_csv(
            output_dir / f'{safe_l1}_trajectory_scores.csv', index=False)
        centroids.to_csv(
            output_dir / f'{safe_l1}_trajectory_centroids.csv', index=False)
        plant_traj.to_csv(
            output_dir / f'{safe_l1}_trajectory_plants.csv', index=False)
        completeness.to_csv(
            output_dir / f'{safe_l1}_trajectory_completeness.csv', index=False)
        ev_df.to_csv(
            output_dir / f'{safe_l1}_trajectory_ev.csv', index=False)

        results[l1_val] = {
            'scores':             scores_df,
            'centroids':          centroids,
            'explained_variance': pca.explained_variance_ratio_,
            'pca':                pca,
            'scaler':             scaler,
            'plant_trajectories': plant_traj,
            'completeness':       completeness,
        }

    return results