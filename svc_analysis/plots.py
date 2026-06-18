"""
svc_analysis/plots.py
=====================
All plotting functions for the hyperspectral analysis pipeline.

Each function is independent and receives only what it needs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _spec_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith('w_')]


def _wavelengths(cols: list[str]) -> np.ndarray:
    return np.array([float(c.split('_')[1]) for c in cols])


def _safe_stem(l1: str, date: str) -> str:
    """Clean strings to be used as filenames (removes spaces, slashes, colons)."""
    s_l1 = str(l1).replace(' ', '_').replace('/', '_').replace('\\', '_').replace(':', '_')
    s_date = str(date).replace(' ', '_').replace('/', '_').replace('\\', '_').replace(':', '_').replace('-', '')
    return f'{s_l1}_{s_date}'


def _palette(name: str, n: int) -> list:
    """Return n colours from the named matplotlib colormap."""
    cmap = plt.get_cmap(name)
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def _save(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    log.info('Saved: %s', path)



def draw_ellipse(ax, x, y, confidence: float = 0.95) -> None:
    """
    Draw a confidence ellipse around a cloud of points.

    Parameters
    ----------
    ax   : matplotlib.axes.Axes
        Axis on which to draw the ellipse.
    x, y : array‑like
        Coordinates of the points belonging to one class/treatment.
    confidence : float, optional
        Desired confidence level (default 0.95).

    Notes
    -----
    The ellipse is based on the covariance matrix of (x, y).
    For a bivariate normal distribution the chi‑square value for 95 % confidence
    is 5.991 (df=2). The scaling factor is therefore
        scale = sqrt( chi2.ppf(confidence, df=2) )
    """
    # Need at least 3 points to estimate a covariance matrix
    if len(x) < 3:
        return

    # ----- 1. Covariance matrix & eigen‑decomposition -----
    cov = np.cov(x, y)                     # 2 × 2 matrix
    eigvals, eigvecs = np.linalg.eig(cov)   # eigvals = λ1, λ2 ; eigvecs columns are eigenvectors

    # ----- 2. Sort eigenvalues (largest first) -----
    order = eigvals.argsort()[::-1]          # descending order
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # ----- 3. Lengths of the ellipse axes -----
    # For a 2‑D Gaussian the chi‑square value for the requested confidence is:
    #   chi2 = scipy.stats.chi2.ppf(confidence, df=2)
    # We avoid pulling in scipy by using the constant for 95 % (≈5.991).
    chi2_val = {0.95: 5.991, 0.99: 9.210, 0.90: 4.605}.get(confidence, 5.991)
    # Axis lengths are sqrt(λ * chi2)
    width  = 2 * np.sqrt(eigvals[0] * chi2_val)   # major axis
    height = 2 * np.sqrt(eigvals[1] * chi2_val)   # minor axis

    # ----- 4. Rotation angle -----
    # The eigenvector corresponding to the largest eigenvalue points along the
    # major axis.  The angle between that vector and the x‑axis gives the rotation.
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    # ----- 5. Draw the ellipse -----
    ellipse = Ellipse(
        xy      = (np.mean(x), np.mean(y)),
        width   = width,
        height  = height,
        angle   = angle,
        edgecolor = 'black',
        facecolor = 'none',
        lw      = 1.2,
        alpha   = 0.6,
    )
    ax.add_patch(ellipse)

# ── 1. Mean spectra ────────────────────────────────────────────────────────────

def plot_mean_spectra(
    spectra_df:   pd.DataFrame,
    metadata_df:  pd.DataFrame,
    cols:         dict,
    output_dir:   str | Path,
    title:        str = 'analysis',
    dpi:          int = 150,
    palette:      str = 'tab10',
    figsize:      tuple = (10, 4),
) -> None:
    """
    Plot the **average spectrum of each plant (level2) per day**,
    coloured by treatment (level3). One subplot per species, one figure per day.
    """
    output_dir = Path(output_dir)
    scan_col   = cols['scan_id']
    l1_col     = cols['level1']
    l2_col     = cols['level2']
    l3_col     = cols['level3']
    date_col   = cols['date']

    # -------------------------------------------------------------
    # Merge the *aggregated* spectra with the metadata (they already share scan_id)
    # -------------------------------------------------------------
    merged = spectra_df.merge(metadata_df, on=scan_col, how='inner')
    spec_cols = _spec_cols(merged)                # w_…
    wavelengths = _wavelengths(spec_cols)

    l1_vals = sorted(merged[l1_col].unique())
    dates   = sorted(merged[date_col].unique())

    for date in dates:
        day_df = merged[merged[date_col] == date]

        n_l1 = len(l1_vals)
        fig, axes = plt.subplots(
            1, n_l1,
            figsize=(figsize[0] * n_l1, figsize[1]),
            sharey=True,
        )
        if n_l1 == 1:
            axes = [axes]

        fig.suptitle(f'{title}  |  {date}', fontsize=13)

        for ax, l1 in zip(axes, l1_vals):
            sub = day_df[day_df[l1_col] == l1]

            # ---------------------------------------------------------
            # Each plant (level2) is a *single* row → plot its spectrum
            # ---------------------------------------------------------
            plants = sub[l2_col].unique()
            treatments = sub[l3_col].unique()
            colors = _palette(palette, len(treatments))

            # Build a colour map: treatment → colour
            colour_map = dict(zip(treatments, colors))

            for plant in plants:
                plant_row = sub[sub[l2_col] == plant]
                if plant_row.empty:
                    continue
                trt = plant_row[l3_col].iloc[0]          # treatment of this plant
                spec = plant_row[spec_cols].values.astype(float).ravel()
                ax.plot(wavelengths, spec,
                        color=colour_map[trt],
                        lw=1.5,
                        label=str(trt) if plant == plants[0] else None)  # label once per treatment
                # Optional: add a tiny marker at the start of the line for visibility
                ax.scatter(wavelengths[0], spec[0], color=colour_map[trt],
                           s=15, edgecolors='k', zorder=5)

            ax.set_title(str(l1), fontsize=11)
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('Reflectance')
            # Only add the legend if there is at least one treatment label
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend(title=l3_col, fontsize=8)
            ax.grid(True, alpha=0.3)

        # -------------------------------------------------------------
        # Save – use the safe stem helper (no illegal characters)
        # -------------------------------------------------------------
        stem = _safe_stem(title, date)
        _save(fig, output_dir / f'{stem}_mean_spectra.png', dpi=dpi)

# ── 2. Confusion matrices ──────────────────────────────────────────────────────

def plot_confusion_matrices(
    results:    dict,
    output_dir: str | Path,
    title:      str = 'analysis',
    dpi:        int = 150,
    cmap:       str = 'Blues',
    figsize:    tuple = (4, 3.5),
) -> None:
    """
    Plot one confusion matrix per level1 / date combination.
    """
    output_dir = Path(output_dir)

    for l1_val, dates in results.items():
        for date_val, res in dates.items():
            cm      = res.get('confusion_matrix')
            classes = res.get('classes')

            if cm is None or classes is None:
                log.warning('No confusion matrix for %s / %s — skipping.',
                            l1_val, date_val)
                continue

            n = len(classes)
            fig, ax = plt.subplots(
                figsize=(figsize[0] * max(n / 2, 1),
                         figsize[1] * max(n / 2, 1))
            )

            im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
            fig.colorbar(im, ax=ax)

            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=9)
            ax.set_yticklabels(classes, fontsize=9)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(f'{l1_val}  |  {date_val}', fontsize=11)

            thresh = cm.max() / 2.0
            for i in range(n):
                for j in range(n):
                    ax.text(
                        j, i, str(cm[i, j]),
                        ha='center', va='center',
                        color='white' if cm[i, j] > thresh else 'black',
                        fontsize=9,
                    )

            stem = _safe_stem(l1_val, date_val)
            _save(fig, output_dir / f'{stem}_confusion_matrix.png', dpi=dpi)


# ── 3. VIP scores ──────────────────────────────────────────────────────────────

def plot_vip_scores(
    results:    dict,
    output_dir: str | Path,
    title:      str = 'analysis',
    dpi:        int = 150,
    figsize:    tuple = (10, 4),
    vip_thresh: float = 1.0,
) -> None:
    """
    Plot VIP scores vs wavelength for each level1 / date combination.
    Highlights wavelengths above the VIP threshold.
    """
    output_dir = Path(output_dir)

    for l1_val, dates in results.items():
        for date_val, res in dates.items():
            vip_df = res.get('vip_df')
            if vip_df is None or vip_df.empty:
                log.warning('No VIP data for %s / %s — skipping.',
                            l1_val, date_val)
                continue

            wl  = vip_df['wavelength'].values
            vip = vip_df['vip'].values

            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(wl, vip, color='steelblue', lw=1.5, label='VIP score')
            ax.axhline(vip_thresh, color='tomato', ls='--', lw=1.2,
                       label=f'Threshold ({vip_thresh})')
            ax.fill_between(wl, vip, vip_thresh,
                            where=(vip >= vip_thresh),
                            alpha=0.25, color='tomato',
                            label='Important region')
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('VIP Score')
            ax.set_title(f'VIP Scores  |  {l1_val}  |  {date_val}', fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

            stem = _safe_stem(l1_val, date_val)
            _save(fig, output_dir / f'{stem}_vip_scores.png', dpi=dpi)


# ── 4. F1 over dates ───────────────────────────────────────────────────────────

def plot_f1_over_dates(
    results:    dict,
    output_dir: str | Path,
    title:      str = 'analysis',
    dpi:        int = 150,
    figsize:    tuple = (8, 4),
    palette:    str = 'tab10',
) -> None:
    """
    Line plot of mean F1 (± std) across dates, one line per level1.
    Uses integer x-positions to avoid matplotlib fill_between issues
    with string/date axes, then relabels ticks.
    """
    output_dir = Path(output_dir)

    records = []
    for l1_val, dates in results.items():
        for date_val, res in dates.items():
            mdf = res.get('metrics_df')
            if mdf is None or mdf.empty:
                continue
            records.append({
                'level1':  l1_val,
                'date':    date_val,
                'f1_mean': mdf['f1'].mean(),
                'f1_std':  mdf['f1'].std(ddof=0),
            })

    if not records:
        log.warning('No F1 data to plot.')
        return

    summary = pd.DataFrame(records).sort_values('date')
    all_dates = sorted(summary['date'].unique())
    date_idx  = {d: i for i, d in enumerate(all_dates)}
    l1_vals   = sorted(summary['level1'].unique())
    colors    = _palette(palette, len(l1_vals))

    fig, ax = plt.subplots(figsize=figsize)

    for color, l1 in zip(colors, l1_vals):
        sub  = summary[summary['level1'] == l1].sort_values('date')
        xs   = [date_idx[d] for d in sub['date']]
        mean = sub['f1_mean'].values
        std  = sub['f1_std'].values

        ax.plot(xs, mean, marker='o', color=color, lw=1.8, label=str(l1))
        ax.fill_between(xs, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_xticks(range(len(all_dates)))
    ax.set_xticklabels(all_dates, rotation=30, ha='right', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Date')
    ax.set_ylabel('Macro F1')
    ax.set_title(f'{title}  |  Macro F1 over Time', fontsize=11)
    ax.legend(title='Level 1', fontsize=9)
    ax.grid(True, alpha=0.3)

    _save(fig, output_dir / f'{title}_f1_over_dates.png', dpi=dpi)


# ── 5. PLS scores scatter (LV1 vs LV2) ────────────────────────────────────────

def plot_pls_scores(
    results:    dict,
    output_dir: str | Path,
    cols:       dict,
    title:      str = 'analysis',
    dpi:        int = 150,
    figsize:    tuple = (5, 4),
    palette:    str = 'tab10',
) -> None:
    """
    Scatter plot of the first two PLS score vectors (LV1 vs LV2),
    coloured by level3 class, for each level1 / date combination.
    Consumes pre-computed 'scores_df' stored by classifier.py.
    """
    output_dir = Path(output_dir)
    l3_col     = cols['level3']

    for l1_val, dates in results.items():
        for date_val, res in dates.items():
            scores_df = res.get('scores_df')

            if scores_df is None or scores_df.empty:
                log.warning('No scores_df for %s / %s — skipping.',
                            l1_val, date_val)
                continue

            if 'LV1' not in scores_df.columns or 'LV2' not in scores_df.columns:
                log.warning('scores_df missing LV1/LV2 for %s / %s — skipping.',
                            l1_val, date_val)
                continue

            classes = sorted(scores_df[l3_col].unique())
            colors  = _palette(palette, len(classes))

            fig, ax = plt.subplots(figsize=figsize)

            for color, cls in zip(colors, classes):
                mask = scores_df[l3_col] == cls
                ax.scatter(
                    scores_df.loc[mask, 'LV1'],
                    scores_df.loc[mask, 'LV2'],
                    label      = str(cls),
                    color      = color,
                    alpha      = 0.75,
                    edgecolors = 'k',
                    linewidths = 0.4,
                    s          = 50,
                )

            ax.set_xlabel('LV1')
            ax.set_ylabel('LV2')
            ax.set_title(f'PLS Scores  |  {l1_val}  |  {date_val}', fontsize=11)
            ax.legend(title=l3_col, fontsize=8)
            ax.grid(True, alpha=0.3)

            stem = _safe_stem(l1_val, date_val)
            _save(fig, output_dir / f'{stem}_pls_scores.png', dpi=dpi)


# ── 6. PCA scores scatter (PC1 vs PC2) ────────────────────────────────────────

def plot_pca_scores(
    results:    dict,
    output_dir: str | Path,
    cols:       dict,
    title:      str = 'analysis',
    dpi:        int = 150,
    figsize:    tuple = (10, 5),
    palette:    str = 'tab10',
) -> None:
    """
    Scatter plot of PC1 vs PC2 and PC1 vs PC3 for each level1 / date,
    with 95 % confidence ellipses.
    """
    output_dir = Path(output_dir)
    l3_col = cols['level3']
    palette_name = palette

    for l1_val, dates in results.items():
        for date_val, res in dates.items():
            scores_df = res.get('scores_df')
            if scores_df is None or scores_df.empty:
                log.warning('No scores_df for %s / %s — skipping.', l1_val, date_val)
                continue

            # Ensure the required PCs exist
            required = {'PC1', 'PC2', 'PC3'}
            missing = required - set(scores_df.columns)
            if missing:
                log.warning('Missing PCs %s for %s / %s — skipping.', missing, l1_val, date_val)
                continue

            # Setup figure with two side‑by‑side subplots
            fig, (ax12, ax13) = plt.subplots(1, 2, figsize=figsize)
            pair_info = [
                ('PC1', 'PC2', ax12, 'PC1 vs PC2'),
                ('PC1', 'PC3', ax13, 'PC1 vs PC3')
            ]

            classes = sorted(scores_df[l3_col].unique())
            colors = _palette(palette_name, len(classes))

            for x_col, y_col, ax, subtitle in pair_info:
                for col, cls in zip(colors, classes):
                    mask = scores_df[l3_col] == cls
                    x = scores_df.loc[mask, x_col].values
                    y = scores_df.loc[mask, y_col].values

                    if len(x) == 0:
                        continue

                    ax.scatter(
                        x, y,
                        label=str(cls),
                        color=col,
                        edgecolors='k',
                        alpha=0.75,
                        s=45,
                        linewidths=0.5,
                    )
                    # ---- draw the 95 % ellipse for this class ----
                    draw_ellipse(ax, x, y, confidence=0.95)

                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(subtitle, fontsize=10)
                ax.grid(True, alpha=0.3)

                # Legend – only once per axis
                handles, labels = ax.get_legend_handles_labels()
                if labels:
                    ax.legend(handles, labels, title=l3_col, fontsize=8)

            fig.suptitle(f'{title} | {l1_val} | {date_val}', fontsize=12)

            stem = _safe_stem(l1_val, date_val)
            _save(fig, output_dir / f'{stem}_pca_scores.png', dpi=dpi)


# --------------------------------------------------------------
# DAILY PCA – PC1 vs PC2  &  PC1 vs PC3 (plant‑averaged spectra)
# --------------------------------------------------------------
def plot_daily_pca(
    results:    dict,
    output_dir: str | Path,
    cols:       dict,
    title:      str = 'analysis',
    dpi:        int = 150,
    figsize:    tuple = (12, 5),
    palette:    str = 'tab10',
) -> None:
    """
    For every species (level1) and every collection day:
        * left panel  – PC1 vs PC2
        * right panel – PC1 vs PC3 (if PC3 exists)
    Points are the **average spectrum of each plant_id** for that day,
    coloured by treatment (level3).  The axis titles contain the %‑explained
    variance of the displayed components.
    """
    output_dir = Path(output_dir)
    l3_col = cols['level3']

    for l1_val, dates in results.items():
        for date_val, res in dates.items():
            scores_df = res.get('scores_df')
            if scores_df is None or scores_df.empty:
                continue

            # -----------------------------------------------------------------
            # Verify we have PC1/PC2 (always) and PC3 (optional)
            # -----------------------------------------------------------------
            have_pc2 = {'PC1', 'PC2'} <= set(scores_df.columns)
            have_pc3 = {'PC1', 'PC3'} <= set(scores_df.columns)

            if not have_pc2:
                log.warning('Daily PCA: missing PC1/PC2 for %s / %s – skipping.',
                            l1_val, date_val)
                continue

            # -----------------------------------------------------------------
            # Colour map for treatments
            # -----------------------------------------------------------------
            treatments = sorted(scores_df[l3_col].unique())
            colors = _palette(palette, len(treatments))
            colour_map = dict(zip(treatments, colors))

            # -----------------------------------------------------------------
            # Figure – two side‑by‑side sub‑plots
            # -----------------------------------------------------------------
            fig, (ax12, ax13) = plt.subplots(1, 2, figsize=figsize)

            # -------------------- PC1 vs PC2 --------------------
            for trt in treatments:
                mask = scores_df[l3_col] == trt
                ax12.scatter(
                    scores_df.loc[mask, 'PC1'],
                    scores_df.loc[mask, 'PC2'],
                    label=str(trt),
                    color=colour_map[trt],
                    edgecolors='k',
                    s=55,
                    alpha=0.8,
                )
            var1 = res['explained_variance'][0] * 100
            var2 = res['explained_variance'][1] * 100 if len(res['explained_variance']) > 1 else 0
            ax12.set_xlabel(f'PC1 ({var1:.1f} %)')
            ax12.set_ylabel(f'PC2 ({var2:.1f} %)')
            ax12.set_title('PC1 vs PC2')
            ax12.grid(True, alpha=0.3)

            # -------------------- PC1 vs PC3 (optional) --------------------
            if have_pc3:
                for trt in treatments:
                    mask = scores_df[l3_col] == trt
                    ax13.scatter(
                        scores_df.loc[mask, 'PC1'],
                        scores_df.loc[mask, 'PC3'],
                        label=str(trt),
                        color=colour_map[trt],
                        edgecolors='k',
                        s=55,
                        alpha=0.8,
                    )
                var3 = res['explained_variance'][2] * 100 if len(res['explained_variance']) > 2 else 0
                ax13.set_xlabel(f'PC1 ({var1:.1f} %)')
                ax13.set_ylabel(f'PC3 ({var3:.1f} %)')
                ax13.set_title('PC1 vs PC3')
                ax13.grid(True, alpha=0.3)
            else:
                # hide the empty panel to keep the layout clean
                ax13.set_visible(False)

            # -----------------------------------------------------------------
            # Shared legend (only once)
            # -----------------------------------------------------------------
            handles, labels = ax12.get_legend_handles_labels()
            if labels:
                ax12.legend(handles, labels, title=l3_col, fontsize=8)

            fig.suptitle(f'{title} | {l1_val} | {date_val}', fontsize=12)

            stem = _safe_stem(l1_val, date_val)
            _save(fig, output_dir / f'{stem}_daily_pca.png', dpi=dpi)

# ── 7. Metrics heatmap ─────────────────────────────────────────────────────────

def plot_metrics_heatmap(
    results:    dict,
    output_dir: str | Path,
    title:      str = 'analysis',
    dpi:        int = 150,
    figsize:    tuple = (10, 6),
    cmap:       str = 'YlGn',
    metrics:    list[str] | None = None,
) -> None:
    """
    Heatmap of macro‑averaged metrics (precision, recall, F1) across dates,
    one figure per level1 (species).

    The original implementation expected a column named ``class`` in the
    concatenated metrics DataFrames, which does not exist in the current
    ``run_plsda`` output (the DataFrames contain only macro scores).  This
    rewritten version works with the existing structure and will also
    gracefully handle a future ``class`` column if it appears.

    Parameters
    ----------
    results, output_dir, title, dpi, figsize, cmap : same as before
    metrics : list of metric column names to plot.
              If ``None`` the default list ``['precision','recall','f1']`` is used.
    """
    output_dir = Path(output_dir)
    metrics = metrics or ['precision', 'recall', 'f1']

    for l1_val, dates in results.items():
        # -----------------------------------------------------------------
        # 1️⃣ Gather per‑date metrics DataFrames
        # -----------------------------------------------------------------
        records = []                 # one row per fold
        for date_val, res in dates.items():
            mdf = res.get('metrics_df')
            if mdf is None or mdf.empty:
                continue

            # Keep only macro columns we care about
            sub = mdf[['precision', 'recall', 'f1']].copy()
            sub['date'] = date_val
            records.append(sub)

        if not records:
            log.warning('No metric data for %s — skipping heatmap.', l1_val)
            continue

        # -----------------------------------------------------------------
        # 2️⃣ Concatenate and compute the *mean* metric per date
        # -----------------------------------------------------------------
        all_metrics = pd.concat(records, ignore_index=True)

        # Pivot so rows = metric, columns = date
        heatmap_df = (
            all_metrics
            .groupby('date')[metrics]
            .mean()                     # macro average across folds
            .T                         # transpose → rows are metrics
        )
        # Ensure the dates appear in chronological order
        heatmap_df = heatmap_df[sorted(heatmap_df.columns)]

        # -----------------------------------------------------------------
        # 3️⃣ Plot the heat‑map
        # -----------------------------------------------------------------
        fig, ax = plt.subplots(figsize=figsize)
        cmap_obj = plt.get_cmap(cmap)

        im = ax.imshow(
            heatmap_df.values,
            aspect='auto',
            cmap=cmap_obj,
            vmin=0,
            vmax=1,
            interpolation='nearest',
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Tick labels
        ax.set_xticks(np.arange(len(heatmap_df.columns)))
        ax.set_xticklabels(
            [str(d) for d in heatmap_df.columns],
            rotation=45,
            ha='right',
            fontsize=9,
        )
        ax.set_yticks(np.arange(len(heatmap_df.index)))
        ax.set_yticklabels(heatmap_df.index, fontsize=9)

        ax.set_xlabel('Date')
        ax.set_ylabel('Metric')
        ax.set_title(f'{title} | {l1_val} | Macro‑Metric Heatmap', fontsize=12)

        # Annotate each cell with the numeric value
        for i in range(len(heatmap_df.index)):
            for j in range(len(heatmap_df.columns)):
                val = heatmap_df.iloc[i, j]
                txt_color = 'white' if val > 0.6 else 'black'
                ax.text(
                    j, i, f'{val:.2f}',
                    ha='center', va='center',
                    color=txt_color,
                    fontsize=8,
                )

        stem = _safe_stem(l1_val, '')
        _save(fig, output_dir / f'{stem}_metrics_heatmap.png', dpi=dpi)

        log.info('Saved metrics heatmap for %s → %s', l1_val, output_dir)

# ── 8. Trajectory (centroid path over time) ────────────────────────────────────

def plot_trajectory_centroids(
    results:    dict,
    output_dir: str | Path,
    cols:       dict,
    title:      str = 'analysis',
    dpi:        int = 150,
    figsize:    tuple = (7, 6),
    palette:    str = 'tab10',
    score_x:    str = 'PC1',
    score_y:    str = 'PC2',
) -> None:
    """
    Plot treatment‑level centroids in PC space over time.
    Works with both the *old* per‑date dict format and the *new* single‑dict
    format returned by ``compute_trajectories``.
    """
    output_dir = Path(output_dir)
    l3_col = cols['level3']

    # -----------------------------------------------------------------
    # Detect which structure we received
    # -----------------------------------------------------------------
    # *Old* format: results[l1][date] → dict with key 'scores_df'
    # *New* format: results[l1] → dict containing the key 'scores' (DataFrame)
    # -----------------------------------------------------------------
    for l1_val, content in results.items():
        # -----------------------------------------------------------------
        # 1️⃣ If `content` looks like a dict of dates (contains a dict)
        # -----------------------------------------------------------------
        if isinstance(content, dict) and any(isinstance(v, dict) for v in content.values()):
            # legacy path – iterate over dates
            records = []
            for date_val, res in content.items():
                df = res.get('scores_df')
                if df is None or df.empty:
                    continue
                if score_x not in df.columns or score_y not in df.columns:
                    continue
                for cls in df[l3_col].unique():
                    mask = df[l3_col] == cls
                    cx = df.loc[mask, score_x].mean()
                    cy = df.loc[mask, score_y].mean()
                    records.append(
                        {'date': date_val, 'class': cls, 'cx': cx, 'cy': cy}
                    )
        # -----------------------------------------------------------------
        # 2️⃣ New format – the DataFrame is directly under the key 'scores'
        # -----------------------------------------------------------------
        elif isinstance(content, dict) and 'scores' in content:
            df = content['scores']
            if df.empty or score_x not in df.columns or score_y not in df.columns:
                continue
            records = []
            for cls in df[l3_col].unique():
                mask = df[l3_col] == cls
                cx = df.loc[mask, score_x].mean()
                cy = df.loc[mask, score_y].mean()
                # Use the date column that already exists in the scores DF
                dates = df.loc[mask, cols['date']].unique()
                for d in dates:
                    records.append({'date': d, 'class': cls, 'cx': cx, 'cy': cy})
        else:
            # Unexpected format – skip this level
            log.warning('Trajectory centroids: unrecognised structure for %s — skipping.', l1_val)
            continue

        # -------------------------------------------------------------
        # Build the tidy DataFrame for plotting
        # -------------------------------------------------------------
        traj_df = pd.DataFrame(records)
        if traj_df.empty:
            continue
        traj_df = traj_df.sort_values('date')
        classes = sorted(traj_df['class'].unique())
        colors = _palette(palette, len(classes))

        # -------------------------------------------------------------
        # Plot
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=figsize)

        for color, cls in zip(colors, classes):
            sub = traj_df[traj_df['class'] == cls]
            ax.plot(sub['cx'].values, sub['cy'].values, color=color,
                    lw=1.5, alpha=0.8, label=str(cls))

            # annotate each point
            for x, y, d in zip(sub['cx'], sub['cy'], sub['date']):
                ax.scatter(x, y, color=color, edgecolors='k', s=55, zorder=5)
                ax.annotate(str(d), xy=(x, y), xytext=(5, 5),
                            textcoords='offset points', fontsize=7, color=color)

        # Legend (only if we have labels)
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(handles, labels, title=l3_col, fontsize=8)

        ax.set_xlabel(score_x)
        ax.set_ylabel(score_y)
        ax.set_title(f'Centroid Trajectories | {l1_val}', fontsize=11)
        ax.grid(True, alpha=0.3)

        stem = _safe_stem(l1_val, '')
        _save(fig, output_dir / f'{stem}_trajectory_centroids.png', dpi=dpi)

# ── 9. Full scatter trajectory (all points, faded by time) ────────────────────

def plot_trajectory(
    results:    dict,
    output_dir: str | Path,
    cols:       dict,
    title:      str = 'analysis',
    dpi:        int = 150,
    figsize:    tuple = (7, 6),
    palette:    str = 'tab10',
    score_x:    str = 'PC1',
    score_y:    str = 'PC2',
) -> None:
    """
    Scatter all individual observations in PC space, coloured by treatment,
    with a time‑fade effect. Works with both the legacy per‑date dict format
    and the new single‑dict format returned by ``compute_trajectories``.
    """
    output_dir = Path(output_dir)
    l3_col = cols['level3']

    for l1_val, content in results.items():
        # -----------------------------------------------------------------
        # Detect which result structure we have (legacy vs new)
        # -----------------------------------------------------------------
        if isinstance(content, dict) and any(isinstance(v, dict) for v in content.values()):
            # ----- LEGACY FORMAT: dict of dates each holding a 'scores_df' -----
            frames = []
            for date_val, res in content.items():
                df = res.get('scores_df')
                if df is None or df.empty:
                    continue
                if score_x not in df.columns or score_y not in df.columns:
                    continue
                tmp = df[[score_x, score_y, l3_col]].copy()
                tmp['date'] = date_val
                frames.append(tmp)

        elif isinstance(content, dict) and 'scores' in content:
            # ----- NEW FORMAT: single DataFrame already contains the date column -----
            df = content['scores']
            if df.empty or score_x not in df.columns or score_y not in df.columns:
                continue

            # Rename the original date column to a uniform name 'date'
            date_original = cols['date']
            tmp = df[[score_x, score_y, l3_col, date_original]].rename(
                columns={date_original: 'date'}
            )
            frames = [tmp]   # a single‑element list – same interface as the legacy case

        else:
            log.warning(
                'Trajectory plot: unrecognised structure for %s — skipping.',
                l1_val,
            )
            continue

        if not frames:
            continue

        # -------------------------------------------------------------
        # Combine all frames (whether from many dates or a single DF)
        # -------------------------------------------------------------
        combined = pd.concat(frames, ignore_index=True)

        # -------------------------------------------------------------
        # Build a colour map for the treatments (level3)
        # -------------------------------------------------------------
        classes = sorted(combined[l3_col].unique())
        colors = _palette(palette, len(classes))
        color_map = dict(zip(classes, colors))

        # -------------------------------------------------------------
        # Compute fade factor (earlier dates = more transparent)
        # -------------------------------------------------------------
        all_dates = sorted(combined['date'].unique())
        n_dates   = len(all_dates)
        date_rank = {d: i for i, d in enumerate(all_dates)}

        fig, ax = plt.subplots(figsize=figsize)

        for _, row in combined.iterrows():
            alpha = 0.15 + 0.75 * (date_rank[row['date']] / max(n_dates - 1, 1))
            ax.scatter(
                row[score_x],
                row[score_y],
                color   = color_map[row[l3_col]],
                alpha   = alpha,
                edgecolors = 'none',
                s       = 20,
            )

        # -------------------------------------------------------------
        # Legend (only if we actually have classes)
        # -------------------------------------------------------------
        import matplotlib.lines as mlines
        handles = [
            mlines.Line2D([], [], color=c, marker='o', linestyle='None',
                         markersize=6, label=str(cls))
            for cls, c in color_map.items()
        ]
        if handles:
            ax.legend(handles=handles, title=l3_col, fontsize=8)

        ax.set_xlabel(score_x)
        ax.set_ylabel(score_y)
        ax.set_title(f'Score Trajectory | {l1_val} (fade = time)', fontsize=11)
        ax.grid(True, alpha=0.3)

        stem = _safe_stem(l1_val, '')
        _save(fig, output_dir / f'{stem}_trajectory.png', dpi=dpi)