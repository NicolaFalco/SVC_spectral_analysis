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
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _spec_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith('w_')]


def _wavelengths(cols: list[str]) -> np.ndarray:
    return np.array([float(c.split('_')[1]) for c in cols])


def _safe_stem(l1: str, date: str) -> str:
    return f'{str(l1).replace(" ", "_")}_{str(date).replace("-", "")}'


def _palette(name: str, n: int) -> list:
    """Return n colours from the named matplotlib colormap."""
    cmap = plt.get_cmap(name)
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def _save(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    log.info('Saved: %s', path)


# ── 1. Mean spectra ────────────────────────────────────────────────────────────

def plot_mean_spectra(
    spectra_df:  pd.DataFrame,
    metadata_df: pd.DataFrame,
    cols:        dict,
    output_dir:  str | Path,
    title:       str = 'analysis',
    dpi:         int = 150,
    palette:     str = 'tab10',
    figsize:     tuple = (10, 4),
) -> None:
    """
    Plot mean ± std spectra grouped by level3 (treatment),
    one subplot per level1 (species/parent), one figure per date.
    """
    output_dir = Path(output_dir)
    scan_col   = cols['scan_id']
    l1_col     = cols['level1']
    l3_col     = cols['level3']
    date_col   = cols['date']

    merged  = spectra_df.merge(metadata_df, on=scan_col, how='inner')
    scols   = _spec_cols(merged)
    wl      = _wavelengths(scols)
    l1_vals = sorted(merged[l1_col].unique())
    dates   = sorted(merged[date_col].unique())

    for date in dates:
        dg    = merged[merged[date_col] == date]
        n_l1  = len(l1_vals)
        fig, axes = plt.subplots(
            1, n_l1,
            figsize=(figsize[0] * n_l1, figsize[1]),
            sharey=True,
        )
        if n_l1 == 1:
            axes = [axes]

        fig.suptitle(f'{title}  |  {date}', fontsize=13)

        for ax, l1 in zip(axes, l1_vals):
            sub    = dg[dg[l1_col] == l1]
            groups = sorted(sub[l3_col].unique())
            colors = _palette(palette, len(groups))

            for color, grp in zip(colors, groups):
                gdata = sub[sub[l3_col] == grp][scols].values.astype(float)
                mean  = gdata.mean(axis=0)
                std   = gdata.std(axis=0)
                ax.plot(wl, mean, label=grp, color=color, lw=1.5)
                ax.fill_between(wl, mean - std, mean + std,
                                alpha=0.2, color=color)

            ax.set_title(str(l1), fontsize=11)
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('Reflectance')
            ax.legend(title=l3_col, fontsize=8)
            ax.grid(True, alpha=0.3)

        safe_date = str(date).replace('-', '')
        _save(fig, output_dir / f'{title}_{safe_date}_mean_spectra.png', dpi=dpi)


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
    figsize:    tuple = (5, 4),
    palette:    str = 'tab10',
) -> None:
    """
    Scatter plot of the first two PCA score vectors (PC1 vs PC2),
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

            if 'PC1' not in scores_df.columns or 'PC2' not in scores_df.columns:
                log.warning('scores_df missing PC1/PC2 for %s / %s — skipping.',
                            l1_val, date_val)
                continue

            classes = sorted(scores_df[l3_col].unique())
            colors  = _palette(palette, len(classes))

            fig, ax = plt.subplots(figsize=figsize)

            for color, cls in zip(colors, classes):
                mask = scores_df[l3_col] == cls
                ax.scatter(
                    scores_df.loc[mask, 'PC1'],
                    scores_df.loc[mask, 'PC2'],
                    label      = str(cls),
                    color      = color,
                    alpha      = 0.75,
                    edgecolors = 'k',
                    linewidths = 0.4,
                    s          = 50,
                )

            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_title(f'PCA Scores  |  {l1_val}  |  {date_val}', fontsize=11)
            ax.legend(title=l3_col, fontsize=8)
            ax.grid(True, alpha=0.3)

            stem = _safe_stem(l1_val, date_val)
            _save(fig, output_dir / f'{stem}_pca_scores.png', dpi=dpi)


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
    Heatmap of per-class metrics (precision, recall, f1) across dates,
    one figure per level1.
    Rows = class labels, Columns = dates, faceted by metric.
    """
    output_dir = Path(output_dir)
    metrics    = metrics or ['precision', 'recall', 'f1']

    for l1_val, dates in results.items():
        # Collect all metrics_df rows
        records = []
        for date_val, res in dates.items():
            mdf = res.get('metrics_df')
            if mdf is None or mdf.empty:
                continue
            mdf = mdf.copy()
            mdf['date'] = date_val
            records.append(mdf)

        if not records:
            log.warning('No metrics data for %s — skipping heatmap.', l1_val)
            continue

        combined  = pd.concat(records, ignore_index=True)
        all_dates = sorted(combined['date'].unique())
        classes   = sorted(combined['class'].unique())

        n_metrics = len(metrics)
        fig, axes = plt.subplots(
            1, n_metrics,
            figsize=(figsize[0], figsize[1]),
            sharey=True,
        )
        if n_metrics == 1:
            axes = [axes]

        fig.suptitle(f'{title}  |  {l1_val}  |  Metrics Heatmap', fontsize=12)

        cmap_obj = plt.get_cmap(cmap)

        for ax, metric in zip(axes, metrics):
            # Build matrix: rows=classes, cols=dates
            matrix = np.full((len(classes), len(all_dates)), np.nan)
            for ci, cls in enumerate(classes):
                for di, date_val in enumerate(all_dates):
                    row = combined[
                        (combined['class'] == cls) &
                        (combined['date']  == date_val)
                    ]
                    if not row.empty and metric in row.columns:
                        matrix[ci, di] = row[metric].iloc[0]

            im = ax.imshow(matrix, aspect='auto', cmap=cmap_obj,
                           vmin=0, vmax=1, interpolation='nearest')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax.set_xticks(range(len(all_dates)))
            ax.set_xticklabels(all_dates, rotation=45, ha='right', fontsize=8)
            ax.set_yticks(range(len(classes)))
            ax.set_yticklabels(classes, fontsize=8)
            ax.set_title(metric.capitalize(), fontsize=10)

            # Annotate cells
            for ci in range(len(classes)):
                for di in range(len(all_dates)):
                    val = matrix[ci, di]
                    if np.isnan(val):
                        ax.text(di, ci, 'N/A', ha='center', va='center',
                                fontsize=7, color='grey')
                    else:
                        brightness = val  # 0–1 range
                        txt_color  = 'white' if brightness < 0.5 else 'black'
                        ax.text(di, ci, f'{val:.2f}', ha='center', va='center',
                                fontsize=7, color=txt_color)

        fig.tight_layout()
        _save(fig, output_dir / f'{str(l1_val).replace(" ", "_")}_metrics_heatmap.png', dpi=dpi)


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
    Plot centroid trajectories through score space over time,
    one line per treatment (level3), one figure per level1.
    Day-1 centroid is marked with a star; all days are annotated.
    """
    output_dir = Path(output_dir)
    l3_col     = cols['level3']

    for l1_val, dates in results.items():
        # Gather centroids
        records = []
        for date_val, res in dates.items():
            scores_df = res.get('scores_df')
            if scores_df is None or scores_df.empty:
                continue
            if score_x not in scores_df.columns or score_y not in scores_df.columns:
                continue
            for cls in scores_df[l3_col].unique():
                mask = scores_df[l3_col] == cls
                cx   = scores_df.loc[mask, score_x].mean()
                cy   = scores_df.loc[mask, score_y].mean()
                records.append({'date': date_val, 'class': cls, 'cx': cx, 'cy': cy})

        if not records:
            log.warning('No score data for trajectory plot: %s — skipping.', l1_val)
            continue

        traj      = pd.DataFrame(records).sort_values('date')
        classes   = sorted(traj['class'].unique())
        all_dates = sorted(traj['date'].unique())
        colors    = _palette(palette, len(classes))

        fig, ax = plt.subplots(figsize=figsize)

        for color, cls in zip(colors, classes):
            sub = traj[traj['class'] == cls].sort_values('date')
            xs  = sub['cx'].values
            ys  = sub['cy'].values
            ds  = sub['date'].values

            ax.plot(xs, ys, color=color, lw=1.5, alpha=0.8)

            for i, (x, y, d) in enumerate(zip(xs, ys, ds)):
                if i == 0:
                    # Day-1: star marker + annotation
                    ax.scatter(x, y, marker='*', s=220, color=color,
                               edgecolors='k', linewidths=0.5, zorder=5)
                else:
                    ax.scatter(x, y, marker='o', s=55, color=color,
                               edgecolors='k', linewidths=0.4, zorder=5)

                ax.annotate(
                    str(d),
                    xy       = (x, y),
                    xytext   = (5, 5),
                    textcoords = 'offset points',
                    fontsize = 7,
                    color    = color,
                )

        # Legend proxies
        import matplotlib.lines as mlines
        handles = [
            mlines.Line2D([], [], color=c, marker='o', markersize=6,
                          label=str(cls), linewidth=1.5)
            for c, cls in zip(colors, classes)
        ]
        ax.legend(handles=handles, title=l3_col, fontsize=8)

        ax.set_xlabel(score_x)
        ax.set_ylabel(score_y)
        ax.set_title(f'Centroid Trajectories  |  {l1_val}', fontsize=11)
        ax.grid(True, alpha=0.3)

        stem = str(l1_val).replace(' ', '_')
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
    Scatter all individual observations through score space,
    alpha-faded by time index (earlier = more transparent),
    coloured by treatment (level3), one figure per level1.
    """
    output_dir = Path(output_dir)
    l3_col     = cols['level3']

    for l1_val, dates in results.items():
        frames = []
        for date_val, res in dates.items():
            scores_df = res.get('scores_df')
            if scores_df is None or scores_df.empty:
                continue
            if score_x not in scores_df.columns or score_y not in scores_df.columns:
                continue
            df = scores_df[[score_x, score_y, l3_col]].copy()
            df['date'] = date_val
            frames.append(df)

        if not frames:
            log.warning('No score data for trajectory plot: %s — skipping.', l1_val)
            continue

        combined  = pd.concat(frames, ignore_index=True)
        all_dates = sorted(combined['date'].unique())
        n_dates   = len(all_dates)
        date_rank = {d: i for i, d in enumerate(all_dates)}

        classes = sorted(combined[l3_col].unique())
        colors  = _palette(palette, len(classes))
        color_map = dict(zip(classes, colors))

        fig, ax = plt.subplots(figsize=figsize)

        for _, row in combined.iterrows():
            alpha = 0.15 + 0.75 * (date_rank[row['date']] / max(n_dates - 1, 1))
            ax.scatter(
                row[score_x], row[score_y],
                color      = color_map[row[l3_col]],
                alpha      = alpha,
                edgecolors = 'none',
                s          = 20,
            )

        # Legend proxies
        import matplotlib.lines as mlines
        handles = [
            mlines.Line2D([], [], color=c, marker='o', markersize=6,
                          linestyle='None', label=str(cls))
            for cls, c in color_map.items()
        ]
        ax.legend(handles=handles, title=l3_col, fontsize=8)

        ax.set_xlabel(score_x)
        ax.set_ylabel(score_y)
        ax.set_title(f'Score Trajectory  |  {l1_val}  |  (fade = time)', fontsize=11)
        ax.grid(True, alpha=0.3)

        stem = str(l1_val).replace(' ', '_')
        _save(fig, output_dir / f'{stem}_trajectory.png', dpi=dpi)