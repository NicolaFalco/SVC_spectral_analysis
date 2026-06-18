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

def aggregate_plant_day(
    spectra_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    cols: dict,
) -> pd.DataFrame:
    """
    Collapse raw scans → one *average spectrum* per plant per calendar day.

    Returns a DataFrame that already contains the hierarchy columns
    (level1, level2, level3, date) plus every spectral column (w_…).
    The function also creates a synthetic ``scan_id`` so that the rest of the
    pipeline (which expects a ``scan_id`` column) continues to work.
    """
    # -----------------------------------------------------------------
    # 1️⃣ Merge spectra + metadata (identical to the original code)
    # -----------------------------------------------------------------
    merged = spectra_df.merge(metadata_df, on=cols['scan_id'], how='inner')

    # -----------------------------------------------------------------
    # 2️⃣ Keep only the calendar date (drop the time part)
    # -----------------------------------------------------------------
    merged[cols['date']] = pd.to_datetime(merged[cols['date']]).dt.date

    # -----------------------------------------------------------------
    # 3️⃣ Identify spectral columns
    # -----------------------------------------------------------------
    spec_cols = [c for c in merged.columns if c.startswith('w_')]

    # -----------------------------------------------------------------
    # 4️⃣ Group by Species (level1), Plant (level2), Date and Treatment (level3)
    #    and compute the mean spectrum for each group.
    # -----------------------------------------------------------------
    agg = (
        merged.groupby([
            cols['level1'],
            cols['level2'],
            cols['date'],
            cols['level3']
        ])[spec_cols]
        .mean()
        .reset_index()
    )

    # -----------------------------------------------------------------
    # 5️⃣ Build a synthetic scan_id for downstream ``merge`` calls
    # -----------------------------------------------------------------
    agg[cols['scan_id']] = (
        agg[cols['level2']].astype(str) + '_' +
        agg[cols['date']].astype(str)
    )
    return agg


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
    dict – see the original docstring for the exact layout.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # 1️⃣  Merge the two tables **first** (the aggregated data already
    #     contains the spectral columns) and grab the spectral columns.
    # -----------------------------------------------------------------
    merged = spectra_df.merge(metadata_df, on=cols['scan_id'], how='inner')
    spec_cols = _spec_cols(merged)                # ← NEW – get w_… columns from merged DF
    wavelengths = _wavelengths(spec_cols)          # ← NEW – array of numeric wavelengths

    # -----------------------------------------------------------------
    # 2️⃣  Pull out the hierarchy column names for convenience
    # -----------------------------------------------------------------
    l1_col   = cols['level1']
    l2_col   = cols['level2']
    l3_col   = cols['level3']
    scan_col = cols['scan_id']
    date_col = cols['date']

    log.info('PCA — merged shape: %s', merged.shape)

    results: dict = {}

    for l1_val, l1_group in merged.groupby(l1_col):
        log.info('── PCA  Level1: %s ─────────────────────────────', l1_val)

        safe_l1 = str(l1_val).replace(' ', '_')
        results[l1_val] = {'per_day': {}, 'trajectory': {}}

        # -----------------------------------------------------------------
        # 1️⃣ & 2️⃣  Per‑day PCA (averaged plant spectra) + loadings
        # -----------------------------------------------------------------
        for date_val, date_group in l1_group.groupby(date_col):
            log.info('   Per‑day PCA (averaged plants)  date=%s  n=%d',
                     date_val, len(date_group))

            X = date_group[spec_cols].values.astype(float)

            # Need at least 3 plants to compute a PCA
            if len(X) < 3:
                log.warning('   Skipping %s / %s – too few plants (%d)',
                            l1_val, date_val, len(X))
                continue

            n_comp = min(n_components, X.shape[0], X.shape[1])

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            pca = PCA(n_components=n_comp)
            scores = pca.fit_transform(X_scaled)

            # -------------------------------------------------------------
            # Build scores DataFrame – keep plant ID and treatment for colour
            # -------------------------------------------------------------
            score_cols = [f'PC{i+1}' for i in range(n_comp)]
            scores_df = pd.DataFrame(scores, columns=score_cols)
            scores_df[l2_col] = date_group[l2_col].values
            scores_df[l3_col] = date_group[l3_col].values
            scores_df[date_col] = date_val

            # -------------------------------------------------------------
            # Loadings (wavelength vs PC)
            # -------------------------------------------------------------
            loadings_df = pd.DataFrame(
                pca.components_.T,
                columns=score_cols,
            )
            loadings_df.insert(0, 'wavelength', wavelengths)

            # -------------------------------------------------------------
            # Save CSVs (use the safe stem helper)
            # -------------------------------------------------------------
            safe_date = str(date_val).replace('-', '')
            stem = f'{safe_l1}_{safe_date}'
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

        # -----------------------------------------------------------------
        # 3️⃣  Trajectory – shared PCA across all days (unchanged)
        # -----------------------------------------------------------------
        log.info('   Trajectory PCA (all days combined)  n=%d', len(l1_group))

        X_all = l1_group[spec_cols].values.astype(float)

        if len(X_all) < 3:
            log.warning('   Too few samples for trajectory PCA — skipping.')
            continue

        n_comp_traj = min(n_components, X_all.shape[0], X_all.shape[1])
        shared_scaler = StandardScaler()
        X_all_scaled = shared_scaler.fit_transform(X_all)
        shared_pca = PCA(n_components=n_comp_traj)
        scores_all = shared_pca.fit_transform(X_all_scaled)

        score_cols = [f'PC{i+1}' for i in range(n_comp_traj)]
        traj_df = pd.DataFrame(scores_all, columns=score_cols)
        traj_df[scan_col] = l1_group[scan_col].values
        traj_df[date_col] = l1_group[date_col].values
        traj_df[l2_col] = l1_group[l2_col].values
        traj_df[l3_col] = l1_group[l3_col].values

        # Find plants present on ≥2 days
        days_per_plant = traj_df.groupby(l2_col)[date_col].nunique()
        multi_day_plants = days_per_plant[days_per_plant >= 2].index.tolist()

        log.info(
            '   Plants with ≥2 days: %d / %d',
            len(multi_day_plants), days_per_plant.shape[0],
        )

        results[l1_val]['trajectory'] = {
            'scores_df':          traj_df,
            'explained_variance': shared_pca.explained_variance_ratio_,
            'pca':                shared_pca,
            'scaler':             shared_scaler,
            'multi_day_plants':   multi_day_plants,
        }

    # -------------------------------------------------------------
    # **IMPORTANT** – make sure we actually *return* the dict!
    # -------------------------------------------------------------
    return results