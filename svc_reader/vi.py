"""
vi.py

Narrow-band vegetation index calculations.

These functions are designed to be passed as `vi_func` to export_tables().

Usage
-----
from svc_reader.vi import narrowVI
from svc_reader.export import export_tables

export_tables(target, reference, output_dir, base_name, vi_func=narrowVI)
"""

import numpy as np


def _band(reflec, wavelengths, target_wl, tol=5.0):
    """
    Extract a single band value closest to target_wl.

    Parameters
    ----------
    reflec : np.ndarray, shape (n_bands,)
    wavelengths : np.ndarray, shape (n_bands,)
    target_wl : float
    tol : float
        Maximum allowed distance (nm) from target_wl.

    Returns
    -------
    value : float or np.nan
    """
    idx = np.argmin(np.abs(wavelengths - target_wl))
    if np.abs(wavelengths[idx] - target_wl) > tol:
        return np.nan
    return float(reflec[idx])


def narrowVI(reflec, wavelengths):
    """
    Compute a set of narrow-band vegetation indices from a single spectrum.

    Parameters
    ----------
    reflec : np.ndarray, shape (n_bands,)
        Reflectance values (single spectrum row).
    wavelengths : np.ndarray, shape (n_bands,)
        Corresponding wavelength values (nm).

    Returns
    -------
    vi : dict
        Dictionary of vegetation index values.
    """
    wavelengths = np.asarray(wavelengths, dtype=float)
    reflec      = np.asarray(reflec,      dtype=float)

    def b(wl):
        return _band(reflec, wavelengths, wl)

    vi = {}

    # --- Broadband greenness ---
    # NDVI: (R800 - R670) / (R800 + R670)
    r800, r670 = b(800), b(670)
    vi['NDVI'] = (r800 - r670) / (r800 + r670) if (r800 + r670) != 0 else np.nan

    # EVI: 2.5 * (R800 - R670) / (R800 + 6*R670 - 7.5*R475 + 1)
    r475 = b(475)
    denom = r800 + 6 * r670 - 7.5 * r475 + 1
    vi['EVI'] = 2.5 * (r800 - r670) / denom if denom != 0 else np.nan

    # SAVI: (R800 - R670) / (R800 + R670 + 0.5) * 1.5
    denom = r800 + r670 + 0.5
    vi['SAVI'] = ((r800 - r670) / denom * 1.5) if denom != 0 else np.nan

    # --- Red-edge ---
    # NDRE: (R750 - R705) / (R750 + R705)
    r750, r705 = b(750), b(705)
    vi['NDRE'] = (r750 - r705) / (r750 + r705) if (r750 + r705) != 0 else np.nan

    # Chl_re: R750 / R705 - 1
    vi['Chl_re'] = (r750 / r705 - 1) if r705 != 0 else np.nan

    # REP: Red-edge inflection point (linear 4-point method, Guyot & Baret 1988)
    r670, r700, r740, r780 = b(670), b(700), b(740), b(780)
    try:
        rre = (r670 + r780) / 2.0
        vi['REP'] = 700 + 40 * ((rre - r700) / (r740 - r700))
    except (ZeroDivisionError, TypeError):
        vi['REP'] = np.nan

    # --- Carotenoids / anthocyanins ---
    # CRI1: 1/R510 - 1/R550
    r510, r550 = b(510), b(550)
    vi['CRI1'] = (1.0 / r510 - 1.0 / r550) if (r510 != 0 and r550 != 0) else np.nan

    # CRI2: 1/R510 - 1/R700
    r700 = b(700)
    vi['CRI2'] = (1.0 / r510 - 1.0 / r700) if (r510 != 0 and r700 != 0) else np.nan

    # ARI1: 1/R550 - 1/R700
    vi['ARI1'] = (1.0 / r550 - 1.0 / r700) if (r550 != 0 and r700 != 0) else np.nan

    # ARI2: R800 * (1/R550 - 1/R700)
    vi['ARI2'] = (r800 * vi['ARI1']) if not np.isnan(vi['ARI1']) else np.nan

    # --- Water / structure ---
    # WBI: R900 / R970
    r900, r970 = b(900), b(970)
    vi['WBI'] = (r900 / r970) if r970 != 0 else np.nan

    # NDWI: (R860 - R1240) / (R860 + R1240)
    r860, r1240 = b(860), b(1240)
    vi['NDWI'] = (r860 - r1240) / (r860 + r1240) if (r860 + r1240) != 0 else np.nan

    # MSI: R1600 / R820
    r1600, r820 = b(1600), b(820)
    vi['MSI'] = (r1600 / r820) if r820 != 0 else np.nan

    # --- Chlorophyll content ---
    # MTCI: (R754 - R709) / (R709 - R681)
    r754, r709, r681 = b(754), b(709), b(681)
    denom = r709 - r681
    vi['MTCI'] = (r754 - r709) / denom if denom != 0 else np.nan

    # TCARI: 3 * [(R700 - R670) - 0.2*(R700 - R550)*(R700/R670)]
    r700, r670, r550 = b(700), b(670), b(550)
    vi['TCARI'] = (
        3 * ((r700 - r670) - 0.2 * (r700 - r550) * (r700 / r670))
        if r670 != 0 else np.nan
    )

    # OSAVI: (R800 - R670) / (R800 + R670 + 0.16)
    denom = r800 + r670 + 0.16
    vi['OSAVI'] = (r800 - r670) / denom if denom != 0 else np.nan

    # TCARI/OSAVI
    vi['TCARI_OSAVI'] = (
        vi['TCARI'] / vi['OSAVI']
        if (not np.isnan(vi['TCARI']) and vi['OSAVI'] not in (0, np.nan))
        else np.nan
    )

    return vi