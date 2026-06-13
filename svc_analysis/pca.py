"""
svc_analysis/pca.py
===================
PCA analysis for the hyperspectral pipeline.

Three analyses are performed per level1 (species):

  1. Per-day PCA   — one PCA per (level1 × date), points coloured by level3
                     (treatment). Uses a PCA fitted on that day's data only.

  2. Loadings      — PC1 and PC2 loadings vs wavelength, per (level1 × date).

  3. Trajectory    — one PCA fitted on ALL days combined per level1, then each
                     plant (level2) that appears on ≥2 days is tracked across
                     days in the shared PC space.
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


def _wavelengths(cols: list[str]) -> np.ndarray:
    return np.array([float(c.split('_')[1]) for c in cols])


# ── Main function ──────────────────────────────────────────────────────────────

def run_pca(
    spectra_df:   pd.DataFrame,
    metadata_df:  pd.DataFrame,
    cols:         dict,
    output_dir:   str | Path,
    n_components: int = 10,
) -> dict:
    """
    Run PCA analyses for each level1 (species).

    Parameters
    ----------
    spectra_df   : stacked spectra DataFrame (columns: scan_id, w_<wl>, …)
    metadata_df  : stacked metadata DataFrame
    cols         : column mapping with keys level1, level2, level3, scan_id, date
    output_dir   : directory for CSV outputs
    n_components : number of PCA components to retain

    Returns
    -------
    results : dict
        results[level1] with keys:

        'per_day' : dict[date] with keys:
            'scores_df'          — PC scores + level2 + level3 + date columns
            'explained_variance' — array of explained variance ratios
            'pca'                — fitted PCA object
            'scaler'             — fitted StandardScaler
            'loadings_df'        — DataFrame (wavelength, PC1, PC2, …)

        'trajectory' : dict with keys:
            'scores_df'          — PC scores for all days in shared space,
                                   includes level2, level3, date columns
            'explained_variance' — array from the shared PCA
            'pca'                — shared fitted PCA object
            'scaler'             — shared fitted StandardScaler
            'multi_day_plants'   — list of level2 IDs present on ≥2 days
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scols       = _spec_cols(spectra_df)
    wavelengths = _wavelengths(scols)

    l1_col   = cols['level1']
    l2_col   = cols['level2']
    l3_col   = cols['level3']
    scan_col = cols['scan_id']
    date_col = cols['date']

    merged = spectra_df.merge(metadata_df, on=scan_col, how='inner')
    log.info('PCA — merged shape: %s', merged.shape)

    results: dict = {}

    for l1_val, l1_group in merged.groupby(l1_col):
        log.info('── PCA  Level1: %s ─────────────────────────────', l1_val)

        safe_l1 = str(l1_val).replace(' ', '_').replace('/', '_').replace('\\', '_')
        results[l1_val] = {'per_day': {}, 'trajectory': {}}

        # ── 1 & 2. Per-day PCA + loadings ─────────────────────────────────────
        for date_val, date_group in l1_group.groupby(date_col):
            log.info('   Per-day PCA  date=%s  n=%d', date_val, len(date_group))

            X = date_group[scols].values.astype(float)

            if len(X) < 3:
                log.warning(
                    '   Skipping %s / %s — too few samples (%d)',
                    l1_val, date_val, len(X),
                )
                continue

            n_comp = min(n_components, X.shape[0], X.shape[1])

            scaler    = StandardScaler()
            X_scaled  = scaler.fit_transform(X)
            pca       = PCA(n_components=n_comp)
            scores    = pca.fit_transform(X_scaled)

            # scores DataFrame
            score_cols = [f'PC{i+1}' for i in range(n_comp)]
            scores_df  = pd.DataFrame(scores, columns=score_cols)
            scores_df[l2_col]   = date_group[l2_col].values
            scores_df[l3_col]   = date_group[l3_col].values
            scores_df[date_col] = date_val

            # loadings DataFrame — shape (n_wavelengths, n_comp)
            loadings_df = pd.DataFrame(
                pca.components_.T,          # (n_features, n_components)
                columns = score_cols,
            )
            loadings_df.insert(0, 'wavelength', wavelengths)

            # save CSVs
            safe_date = str(date_val).replace('-', '').replace('/', '_').replace('\\', '_')
            stem      = f'{safe_l1}_{safe_date}'

            scores_df.to_csv(output_dir / f'{stem}_pca_scores.csv', index=False)
            loadings_df.to_csv(output_dir / f'{stem}_pca_loadings.csv', index=False)

            log.info(
                '   Explained variance PC1=%.1f%%  PC2=%.1f%%',
                pca.explained_variance_ratio_[0] * 100,
                pca.explained_variance_ratio_[1] * 100 if n_comp > 1 else 0,
            )

            results[l1_val]['per_day'][date_val] = {
                'scores_df':          scores_df,
                'explained_variance': pca.explained_variance_ratio_,
                'pca':                pca,
                'scaler':             scaler,
                'loadings_df':        loadings_df,
            }

        # ── 3. Trajectory — shared PCA across all days ─────────────────────────
        log.info('   Trajectory PCA  (all days combined)  n=%d', len(l1_group))

        X_all = l1_group[scols].values.astype(float)

        if len(X_all) < 3:
            log.warning('   Too few samples for trajectory PCA — skipping.')
            continue

        n_comp_traj  = min(n_components, X_all.shape[0], X_all.shape[1])
        shared_scaler = StandardScaler()
        X_all_scaled  = shared_scaler.fit_transform(X_all)
        shared_pca    = PCA(n_components=n_comp_traj)
        scores_all    = shared_pca.fit_transform(X_all_scaled)

        score_cols = [f'PC{i+1}' for i in range(n_comp_traj)]
        traj_df    = pd.DataFrame(scores_all, columns=score_cols)
        traj_df[l2_col]   = l1_group[l2_col].values
        traj_df[l3_col]   = l1_group[l3_col].values
        traj_df[date_col] = l1_group[date_col].values

        # find plants present on ≥2 days
        days_per_plant  = traj_df.groupby(l2_col)[date_col].nunique()
        multi_day_plants = days_per_plant[days_per_plant >= 2].index.tolist()

        log.info(
            '   Plants with ≥2 days: %d / %d',
            len(multi_day_plants),
            days_per_plant.shape[0],
        )

        # save trajectory scores
        traj_df.to_csv(
            output_dir / f'{safe_l1}_trajectory_scores.csv', index=False,
        )

        results[l1_val]['trajectory'] = {
            'scores_df':          traj_df,
            'explained_variance': shared_pca.explained_variance_ratio_,
            'pca':                shared_pca,
            'scaler':             shared_scaler,
            'multi_day_plants':   multi_day_plants,
        }

    return results