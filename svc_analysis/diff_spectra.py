"""
svc_analysis/diff_spectra.py
============================
Difference spectra per treatment relative to each plant's baseline.

For each species and treatment, computes the mean reflectance difference
between each day and the first recorded day of each individual plant,
then averages across plants within each treatment group.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _spec_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith('w_')]


def _wavelengths(cols: list[str]) -> np.ndarray:
    return np.array([float(c.split('_')[1]) for c in cols])


# ── Main function ──────────────────────────────────────────────────────────────

def compute_diff_spectra(
    spectra_df:  pd.DataFrame,
    metadata_df: pd.DataFrame,
    cols:        dict,
    output_dir:  str | Path,
) -> dict:
    """
    Compute mean difference spectra per species, treatment, and day
    relative to each plant's first recorded day (earliest date = baseline).

    Parameters
    ----------
    spectra_df  : stacked spectra DataFrame
    metadata_df : stacked metadata DataFrame
    cols        : column mapping with keys level1, level2, level3, scan_id, date
    output_dir  : directory for CSV outputs

    Returns
    -------
    results : dict
        results[level1][level3] with keys:
            'wavelengths' — np.ndarray of wavelength values
            <date>        — pd.Series of mean diff spectrum for that date
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    l1_col   = cols['level1']
    l2_col   = cols['level2']
    l3_col   = cols['level3']
    scan_col = cols['scan_id']
    date_col = cols['date']

    merged = spectra_df.merge(metadata_df, on=scan_col, how='inner')
    log.info('Diff spectra — merged shape: %s', merged.shape)

    scols       = _spec_cols(merged)
    wavelengths = _wavelengths(scols)

    results: dict = {}

    for l1_val, l1_group in merged.groupby(l1_col):
        log.info('── Diff spectra  Level1: %s ────────────────────', l1_val)
        results[l1_val] = {}
        safe_l1 = str(l1_val).replace(' ', '_').replace('/', '_').replace('\\', '_')

        for l3_val, tr_group in l1_group.groupby(l3_col):
            log.info('   Treatment: %s  (n=%d)', l3_val, len(tr_group))
            results[l1_val][l3_val] = {'wavelengths': wavelengths}

            # ── Per-plant baseline (earliest date) ─────────────────────────────
            plant_baselines: dict[str, pd.Series] = {}
            for plant, pl_df in tr_group.groupby(l2_col):
                first_day = sorted(pl_df[date_col].unique())[0]
                baseline  = pl_df[pl_df[date_col] == first_day][scols].mean()
                plant_baselines[plant] = baseline

            # ── Per-day mean difference across plants ──────────────────────────
            days = sorted(tr_group[date_col].unique())
            diff_rows = []   # for CSV export

            for day in days:
                day_df    = tr_group[tr_group[date_col] == day]
                day_diffs = []

                for plant, pl_df in day_df.groupby(l2_col):
                    if plant not in plant_baselines:
                        continue
                    plant_mean = pl_df[scols].mean()
                    diff       = plant_mean - plant_baselines[plant]
                    day_diffs.append(diff)

                if day_diffs:
                    mean_diff = pd.concat(day_diffs, axis=1).mean(axis=1)
                    mean_diff.index = scols
                    log.info('   Day %s — %d plants contributed', day, len(day_diffs))
                else:
                    mean_diff = pd.Series(np.nan, index=scols)
                    log.warning('   Day %s — no valid plants, NaN spectrum', day)

                results[l1_val][l3_val][day] = mean_diff

                diff_rows.append({
                    date_col: day,
                    **dict(zip(wavelengths, mean_diff.values)),
                })

            # ── Save CSV ───────────────────────────────────────────────────────
            safe_l3 = str(l3_val).replace(' ', '_').replace('/', '_').replace('\\', '_')
            csv_path = output_dir / f'{safe_l1}_{safe_l3}_diff_spectra.csv'
            pd.DataFrame(diff_rows).to_csv(csv_path, index=False)
            log.info('   Saved: %s', csv_path)

    return results


# ── Flatten to tidy DataFrame ──────────────────────────────────────────────────

def diff_spectra_to_dataframe(results: dict, cols: dict) -> pd.DataFrame:
    """
    Flatten the nested diff spectra results dict into a tidy DataFrame.
    Non-destructive — does not modify the results dict.

    Parameters
    ----------
    results : dict
        Output of compute_diff_spectra().
    cols : dict
        Column mapping, used for output column naming.

    Returns
    -------
    pd.DataFrame
        Columns: level1, level3, date, wavelength, mean_diff
    """
    l1_col   = cols['level1']
    l3_col   = cols['level3']
    date_col = cols['date']

    rows = []
    for l1_val, treatments in results.items():
        for l3_val, day_data in treatments.items():
            wavelengths = day_data.get('wavelengths')
            if wavelengths is None:
                continue
            for key, diff_series in day_data.items():
                if key == 'wavelengths':
                    continue
                for wl, val in zip(wavelengths, diff_series.values):
                    rows.append({
                        l1_col:    l1_val,
                        l3_col:    l3_val,
                        date_col:  key,
                        'wavelength': wl,
                        'mean_diff':  val,
                    })

    return pd.DataFrame(rows)