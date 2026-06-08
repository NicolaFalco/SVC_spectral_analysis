"""
export.py

Build reflectance matrix, smooth, detect outliers, and export CSV files.
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def build_reflectance_matrix(target, reference):
    """
    Align all spectra to a common wavelength grid and compute
    reflectance (target / reference).

    Parameters
    ----------
    target : list of dict
    reference : list of dict

    Returns
    -------
    reflec_mat : np.ndarray, shape (n_spectra, n_wavelengths)
    var_wave_name : np.ndarray
    """
    wave_lengths = np.array([len(t['wavelength']) for t in target])
    unique_lens  = np.unique(wave_lengths)

    # Common grid = longest wavelength vector found
    longest_idx  = np.where(wave_lengths == unique_lens[-1])[0][0]
    var_wave_name = target[longest_idx]['wavelength']

    n = len(target)
    reflec_mat = np.full((n, len(var_wave_name)), np.nan)

    for k in range(n):
        t_data = target[k]['data'].copy()
        r_data = reference[k]['data'].copy()

        if len(target[k]['wavelength']) == len(var_wave_name):
            # Wavelength grids already match
            with np.errstate(divide='ignore', invalid='ignore'):
                reflec_mat[k, :] = np.where(r_data != 0, t_data / r_data, np.nan)
        else:
            # Map shorter wavelength vector onto the common grid
            short_wave = target[k]['wavelength']
            new_tar = np.full(len(var_wave_name), np.nan)
            new_ref = np.full(len(var_wave_name), np.nan)

            for j in range(len(short_wave)):
                idx = np.argmin(np.abs(var_wave_name - short_wave[j]))
                new_tar[idx] = t_data[j]
                new_ref[idx] = r_data[j]

            with np.errstate(divide='ignore', invalid='ignore'):
                reflec_mat[k, :] = np.where(new_ref != 0, new_tar / new_ref, np.nan)

    return reflec_mat, var_wave_name


def smooth_reflectance(reflec_mat, window_length=5, polyorder=2):
    """
    Apply Savitzky-Golay smoothing row-wise, skipping NaN regions.

    Parameters
    ----------
    reflec_mat : np.ndarray
    window_length : int
    polyorder : int

    Returns
    -------
    smoothed : np.ndarray
    """
    smoothed = reflec_mat.copy()
    for i in range(reflec_mat.shape[0]):
        row = reflec_mat[i, :]
        valid = ~np.isnan(row)
        if valid.sum() > window_length:
            smoothed[i, valid] = savgol_filter(
                row[valid], window_length=window_length, polyorder=polyorder)
    return smoothed


def build_metadata_table(target):
    """
    Build a DataFrame with per-spectrum metadata.

    Parameters
    ----------
    target : list of dict

    Returns
    -------
    df : pd.DataFrame
        Columns: Spec_scan, Datetime, Latitude, Longitude
    """
    records = []
    for t in target:
        gps = t['header'].get('GPSCoordinates', [float('nan'), float('nan')])
        if isinstance(gps, str):
            lat, lon = float('nan'), float('nan')
        else:
            lat = gps[0] if len(gps) > 0 else float('nan')
            lon = gps[1] if len(gps) > 1 else float('nan')

        records.append({
            'Spec_scan': t['name'],
            'Datetime':  t['datetime'],
            'Latitude':  lat,
            'Longitude': lon,
        })

    return pd.DataFrame(records)


def build_reflectance_table(reflec_mat, var_wave_name):
    """
    Build a DataFrame from the reflectance matrix.
    Column names formatted as w_<wavelength> (e.g. w_350_5).

    Parameters
    ----------
    reflec_mat : np.ndarray
    var_wave_name : np.ndarray

    Returns
    -------
    df : pd.DataFrame
    """
    col_names = [
        'w_' + str(round(float(w), 2)).replace('.', '_')
        for w in var_wave_name
    ]
    return pd.DataFrame(reflec_mat, columns=col_names)


def detect_outliers_mean(data):
    """
    Detect outliers using a 3-sigma mean-based rule.
    Equivalent to MATLAB isoutlier(..., 'mean').

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)

    Returns
    -------
    is_out : np.ndarray of bool, same shape as data
    """
    mean = np.nanmean(data, axis=0)
    std  = np.nanstd(data, axis=0)
    with np.errstate(invalid='ignore'):
        is_out = np.abs(data - mean) > 3 * std
    return is_out


def remove_outliers(spec_t, reflec_mat, vi_tab=None):
    """
    Remove rows identified as outliers in the reflectance
    matrix or VI table.

    Parameters
    ----------
    spec_t : pd.DataFrame
    reflec_mat : np.ndarray
    vi_tab : pd.DataFrame or None

    Returns
    -------
    spec_t_noout : pd.DataFrame
    idout : np.ndarray of int
        Row indices of removed outliers.
    """
    out_band   = detect_outliers_mean(reflec_mat)
    idout_band = np.where(out_band.sum(axis=1) > 0)[0]

    if vi_tab is not None and not vi_tab.empty:
        out_vi   = detect_outliers_mean(vi_tab.values.astype(float))
        idout_vi = np.where(out_vi.sum(axis=1) > 0)[0]
        idout    = np.unique(np.concatenate([idout_band, idout_vi]))
    else:
        idout = idout_band

    spec_t_noout = spec_t.drop(index=idout).reset_index(drop=True)
    return spec_t_noout, idout


def remove_nan_inf(spec_t_noout, reflec_mat_cl):
    """
    Remove rows containing NaN or Inf in the reflectance matrix.

    Parameters
    ----------
    spec_t_noout : pd.DataFrame
    reflec_mat_cl : np.ndarray
        Reflectance matrix already stripped of outlier rows.

    Returns
    -------
    spec_t_cl : pd.DataFrame
    """
    bad_rows = np.unique(
        np.where(np.isnan(reflec_mat_cl) | np.isinf(reflec_mat_cl))[0]
    )
    if len(bad_rows) > 0:
        spec_t_cl = spec_t_noout.drop(index=bad_rows).reset_index(drop=True)
    else:
        spec_t_cl = spec_t_noout.copy()
    return spec_t_cl


def build_vi_table(reflec_mat, var_wave_name, vi_func, meta_df):
    """
    Build a DataFrame of vegetation indices with metadata prepended.

    Parameters
    ----------
    reflec_mat : np.ndarray
    var_wave_name : np.ndarray
    vi_func : callable
    meta_df : pd.DataFrame

    Returns
    -------
    vi_df : pd.DataFrame
    """
    vi_rows = []
    for i in range(reflec_mat.shape[0]):
        vi_rows.append(vi_func(reflec_mat[i, :], var_wave_name))
    vi_df = pd.DataFrame(vi_rows)
    return pd.concat([meta_df.reset_index(drop=True),
                      vi_df.reset_index(drop=True)], axis=1)


def export_tables(target, reference, output_dir, base_name,
                  smooth=False, vi_func=None):
    """
    Full pipeline: build reflectance matrix, detect outliers, clean, export CSVs.

    Order of operations:
        Raw:      raw data → VI → outliers (reflectance + VI) → NaN/Inf clean
        Smoothed: smooth   → VI → outliers (reflectance + VI) → NaN/Inf clean

    Saves the following CSV files:

        Always:
            SVC_<base_name>_metadata.csv       — metadata only
            SVC_<base_name>.csv                — Spec_scan + reflectance, raw
            SVC_<base_name>_noOut.csv          — raw, outliers removed
            SVC_<base_name>_noOut_cl.csv       — raw, outliers + NaN/Inf removed
                                                 (only if NaN/Inf found)

        With --vi:
            SVC_<base_name>_vi.csv             — Spec_scan + VI, raw

        With --sm:
            SVC_<base_name>_sm.csv             — Spec_scan + reflectance, smoothed
            SVC_<base_name>_sm_noOut.csv       — smoothed, outliers removed
            SVC_<base_name>_sm_noOut_cl.csv    — smoothed, outliers + NaN/Inf removed
                                                 (only if NaN/Inf found)

        With --sm and --vi:
            SVC_<base_name>_sm_vi.csv          — Spec_scan + VI, smoothed

    Parameters
    ----------
    target : list of dict
    reference : list of dict
    output_dir : str
    base_name : str
    smooth : bool
    vi_func : callable or None

    Returns
    -------
    spec_t : pd.DataFrame
    """
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Build reflectance matrix
    # ------------------------------------------------------------------
    print("Building reflectance matrix...")
    reflec_mat, var_wave_name = build_reflectance_matrix(target, reference)
    print(f"  Shape: {reflec_mat.shape[0]} spectra x "
          f"{reflec_mat.shape[1]} wavelengths")

    # ------------------------------------------------------------------
    # 2. Metadata + ID column
    # ------------------------------------------------------------------
    print("Building metadata table...")
    meta_df = build_metadata_table(target)

    meta_path = os.path.join(output_dir, f"SVC_{base_name}_metadata.csv")
    meta_df.to_csv(meta_path, index=False)
    print(f"Saved: {meta_path}")

    id_col = meta_df[['Spec_scan']].reset_index(drop=True)

    # ------------------------------------------------------------------
    # 3. Raw pipeline: raw → VI → outliers → NaN/Inf
    # ------------------------------------------------------------------
    print("\n--- Raw pipeline ---")

    # 3a. Save raw reflectance (before any cleaning)
    reflec_tab_raw = build_reflectance_table(reflec_mat, var_wave_name)
    spec_t_raw = pd.concat(
        [id_col, reflec_tab_raw.reset_index(drop=True)], axis=1)

    out_path_raw = os.path.join(output_dir, f"SVC_{base_name}.csv")
    spec_t_raw.to_csv(out_path_raw, index=False)
    print(f"Saved: {out_path_raw}")

    # 3b. VI on raw data
    if vi_func is not None:
        print("Computing vegetation indices (raw)...")
        vi_vals_raw = pd.DataFrame([
            vi_func(reflec_mat[i, :], var_wave_name)
            for i in range(reflec_mat.shape[0])
        ])
        vi_tab_raw = pd.concat(
            [id_col, vi_vals_raw.reset_index(drop=True)], axis=1)
        vi_path = os.path.join(output_dir, f"SVC_{base_name}_vi.csv")
        vi_tab_raw.to_csv(vi_path, index=False)
        print(f"Saved: {vi_path}")
    else:
        vi_vals_raw = pd.DataFrame()

    # 3c. Outlier removal on raw data (reflectance + VI)
    print("Detecting and removing outliers (raw)...")
    spec_t_noout, idout = remove_outliers(spec_t_raw, reflec_mat, vi_vals_raw)
    print(f"  Removed {len(idout)} outlier row(s).")

    out_path_noout = os.path.join(output_dir, f"SVC_{base_name}_noOut.csv")
    spec_t_noout.to_csv(out_path_noout, index=False)
    print(f"Saved: {out_path_noout}")

    # 3d. NaN/Inf clean on outlier-removed raw data
    reflec_mat_noout = np.delete(reflec_mat, idout, axis=0)
    spec_t_cl = remove_nan_inf(spec_t_noout, reflec_mat_noout)

    if len(spec_t_cl) < len(spec_t_noout):
        n_removed = len(spec_t_noout) - len(spec_t_cl)
        print(f"  Removed {n_removed} NaN/Inf row(s) from raw table.")
        out_path_cl = os.path.join(output_dir, f"SVC_{base_name}_noOut_cl.csv")
        spec_t_cl.to_csv(out_path_cl, index=False)
        print(f"Saved: {out_path_cl}")
    else:
        print("  No NaN/Inf rows found in raw table.")

    spec_t = spec_t_noout

    # ------------------------------------------------------------------
    # 4. Smoothed pipeline: smooth → VI → outliers → NaN/Inf
    # ------------------------------------------------------------------
    if smooth:
        print("\n--- Smoothed pipeline ---")
        print("Smoothing reflectance (Savitzky-Golay, window=5, poly=2)...")
        reflec_mat_sm = smooth_reflectance(reflec_mat)

        # 4a. Save smoothed reflectance (before any cleaning)
        reflec_tab_sm = build_reflectance_table(reflec_mat_sm, var_wave_name)
        spec_t_sm = pd.concat(
            [id_col, reflec_tab_sm.reset_index(drop=True)], axis=1)

        out_path_sm = os.path.join(output_dir, f"SVC_{base_name}_sm.csv")
        spec_t_sm.to_csv(out_path_sm, index=False)
        print(f"Saved: {out_path_sm}")

        # 4b. VI on smoothed data
        if vi_func is not None:
            print("Computing vegetation indices (smoothed)...")
            vi_vals_sm = pd.DataFrame([
                vi_func(reflec_mat_sm[i, :], var_wave_name)
                for i in range(reflec_mat_sm.shape[0])
            ])
            vi_tab_sm = pd.concat(
                [id_col, vi_vals_sm.reset_index(drop=True)], axis=1)
            vi_path_sm = os.path.join(output_dir, f"SVC_{base_name}_sm_vi.csv")
            vi_tab_sm.to_csv(vi_path_sm, index=False)
            print(f"Saved: {vi_path_sm}")
        else:
            vi_vals_sm = pd.DataFrame()

        # 4c. Outlier removal on smoothed data (reflectance + VI)
        print("Detecting and removing outliers (smoothed)...")
        spec_t_sm_noout, idout_sm = remove_outliers(
            spec_t_sm, reflec_mat_sm, vi_vals_sm)
        print(f"  Removed {len(idout_sm)} outlier row(s).")

        out_path_sm_noout = os.path.join(
            output_dir, f"SVC_{base_name}_sm_noOut.csv")
        spec_t_sm_noout.to_csv(out_path_sm_noout, index=False)
        print(f"Saved: {out_path_sm_noout}")

        # 4d. NaN/Inf clean on smoothed + outlier-removed data
        reflec_mat_sm_cl = np.delete(reflec_mat_sm, idout_sm, axis=0)
        spec_t_sm_cl = remove_nan_inf(spec_t_sm_noout, reflec_mat_sm_cl)

        if len(spec_t_sm_cl) < len(spec_t_sm_noout):
            n_removed = len(spec_t_sm_noout) - len(spec_t_sm_cl)
            print(f"  Removed {n_removed} NaN/Inf row(s) from smoothed table.")
            out_path_sm_cl = os.path.join(
                output_dir, f"SVC_{base_name}_sm_noOut_cl.csv")
            spec_t_sm_cl.to_csv(out_path_sm_cl, index=False)
            print(f"Saved: {out_path_sm_cl}")
        else:
            print("  No NaN/Inf rows found in smoothed table.")

        spec_t = spec_t_sm_noout

    print("\nAll done.")
    return spec_t