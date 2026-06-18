"""
svc_analysis/__main__.py
========================
Full orchestrator for the hyperspectral analysis pipeline.
"""

from __future__ import annotations

import argparse
import importlib.resources as pkg_resources
import logging
from pathlib import Path

import pandas as pd
import yaml

# Import analytical modules
from .pca import run_pca
from .plsda import run_plsda
from .trajectory import compute_trajectories
from .diff_spectra import compute_diff_spectra

# Import plotting modules
from .plots import (
    plot_mean_spectra,
    plot_confusion_matrices,
    plot_vip_scores,
    plot_f1_over_dates,
    plot_metrics_heatmap,
    plot_pca_scores,
    plot_trajectory_centroids,
    plot_trajectory, 
    plot_daily_pca,
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = '%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt= '%H:%M:%S',
)
log = logging.getLogger(__name__)

# ── Config helpers ─────────────────────────────────────────────────────────────

def _default_config_path() -> Path:
    with pkg_resources.path('svc_analysis', 'config.yaml') as p:
        return Path(p)

def _load_config(path: str | Path | None = None) -> dict:
    if path is None:
        path = _default_config_path()
    with open(path) as f:
        cfg = yaml.safe_load(f)
    log.info('Config loaded from %s', path)
    return cfg

def _validate_config(cfg: dict) -> None:
    required = ['hierarchy.level1', 'hierarchy.level2', 'hierarchy.level3', 'scan.id_col', 'scan.date_col']
    for key in required:
        parts = key.split('.')
        node = cfg
        for p in parts:
            if not isinstance(node, dict) or p not in node:
                raise KeyError(f'Missing required config key: {key}')
            node = node[p]

def _get_cols(cfg: dict) -> dict:
    return {
        'level1':   cfg['hierarchy']['level1'],
        'level2':   cfg['hierarchy']['level2'],
        'level3':   cfg['hierarchy']['level3'],
        'scan_id':  cfg['scan']['id_col'],
        'date':     cfg['scan']['date_col'],
    }

# svc_analysis/__main__.py
# --------------------------------------------------------------
#  run_pipeline – the heart of the whole analysis
# --------------------------------------------------------------
def run_pipeline(cfg: dict) -> None:
    """
    Execute the full pipeline with plant‑level daily aggregation.
    All downstream analyses (PCA, PLS‑DA, trajectories, diff‑spectra,
    VIP, etc.) are performed on the **average spectrum of each plant
    per day**, guaranteeing a consistent data foundation.
    """
    # -----------------------------------------------------------------
    # 0️⃣ Prepare output folder, title and column‑mapping
    # -----------------------------------------------------------------
    output_dir = Path(cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    title = cfg.get('title', 'analysis')
    cols  = _get_cols(cfg)                     # level1, level2, level3, scan_id, date
    plot_cfg = cfg.get('plots', {})

    # -----------------------------------------------------------------
    # 1️⃣ Load raw CSVs (unchanged)
    # -----------------------------------------------------------------
    log.info('Loading raw spectra and metadata...')
    spectra_raw  = pd.read_csv(cfg['spectra_path'])
    metadata_raw = pd.read_csv(cfg['metadata_path'])

    # -----------------------------------------------------------------
    # 2️⃣ Aggregate → ONE averaged spectrum per plant per calendar day
    # -----------------------------------------------------------------
    log.info('Aggregating scans: 1 average spectrum per plant per day …')
    from .pca import aggregate_plant_day   # helper added in pca.py
    aggregated = aggregate_plant_day(spectra_raw, metadata_raw, cols)

    # -----------------------------------------------------------------
    # 3️⃣ Split the aggregated DataFrame back into the two objects the
    #    analysis functions expect (spectra + metadata)
    # -----------------------------------------------------------------
    spec_cols = [c for c in aggregated.columns if c.startswith('w_')]
    agg_spectra   = aggregated[[cols['scan_id']] + spec_cols]

    agg_metadata = aggregated[[cols['scan_id'],
                              cols['level1'],
                              cols['level2'],
                              cols['level3'],
                              cols['date']]]

    log.info('Aggregation complete → %d plant‑day rows', len(agg_spectra))

    # -----------------------------------------------------------------
    # 4️⃣ Run **all** analyses on the *aggregated* data
    # -----------------------------------------------------------------
    log.info('Running PCA analysis on aggregated plant spectra …')
    pca_results = run_pca(
        spectra_df   = agg_spectra,
        metadata_df  = agg_metadata,
        cols         = cols,
        output_dir   = output_dir,
        n_components = cfg.get('trajectory', {}).get('n_components', 10)
    )

    log.info('Running PLS‑DA classification on aggregated data …')
    pls_cfg = cfg.get('plsda', {})
    pls_results = run_plsda(
        spectra_df   = agg_spectra,
        metadata_df  = agg_metadata,
        cols         = cols,
        output_dir   = output_dir,
        cv           = pls_cfg.get('cv', 'stratified'),
        n_folds      = pls_cfg.get('n_folds', 5),
        n_iter_search = pls_cfg.get('n_iter_search', 20),
        random_state = pls_cfg.get('random_state', 42)
    )

    log.info('Computing trajectories …')
    traj_results = compute_trajectories(
        spectra_df   = agg_spectra,
        metadata_df  = agg_metadata,
        cols         = cols,
        output_dir   = output_dir,
        n_components = cfg.get('trajectory', {}).get('n_components', 10)
    )

    log.info('Computing difference spectra …')
    compute_diff_spectra(
        spectra_df   = agg_spectra,
        metadata_df  = agg_metadata,
        cols         = cols,
        output_dir   = output_dir
    )

    # -----------------------------------------------------------------
    # 5️⃣ Plotting – only pass arguments that each routine accepts
    # -----------------------------------------------------------------
    log.info('Generating final figures…')

    # -----------------------------------------------------------------
    # Re‑use the same three argument bundles throughout the block
    # -----------------------------------------------------------------
    common_args = {
        'dpi'      : plot_cfg.get('dpi', 150),
        'figsize'  : plot_cfg.get('figsize', (10, 4)),
    }

    palette_args = {
        'palette'  : plot_cfg.get('palette', 'tab10')
    }

    cmap_args = {
        'cmap'     : plot_cfg.get('cmap', 'Blues'),
        'dpi'      : plot_cfg.get('dpi', 150),
        'figsize' : plot_cfg.get('figsize', (10, 4)),
    }

    # -------------------------------------------------------------
    # 5.1 Mean spectra (raw data – shows variability)
    # -------------------------------------------------------------
    plot_mean_spectra(
        spectra_df   = agg_spectra,
        metadata_df = agg_metadata,
        cols         = cols,
        output_dir   = output_dir,
        title        = title,
        **common_args,
        **palette_args,
    )

    # -------------------------------------------------------------
    # 5.2 DAILY PCA (plant‑averaged spectra) – two panels per day
    # -------------------------------------------------------------
    plot_daily_pca(
        results    = pca_results,       # contains per‑day scores
        output_dir = output_dir,
        cols       = cols,
        title      = title,
        **common_args,
        **palette_args,
    )

    # -------------------------------------------------------------
    # 5.3 Global PCA (all days together) – optional overview
    # -------------------------------------------------------------
    plot_pca_scores(
        results    = pca_results,
        output_dir = output_dir,
        cols       = cols,
        title      = title,
        **common_args,
        **palette_args,
    )

    # -------------------------------------------------------------
    # 5.4 Confusion matrices (PLS‑DA)
    # -------------------------------------------------------------
    plot_confusion_matrices(
        results    = pls_results,
        output_dir = output_dir,
        title      = title,
        **cmap_args,
    )

    # -------------------------------------------------------------
    # 5.5 Metrics heatmap (macro metrics)
    # -------------------------------------------------------------
    plot_metrics_heatmap(
        results    = pls_results,
        output_dir = output_dir,
        title      = title,
        **cmap_args,
    )

    # -------------------------------------------------------------
    # 5.6 VIP scores (PLS‑DA) – no palette argument
    # -------------------------------------------------------------
    plot_vip_scores(
        results    = pls_results,
        output_dir = output_dir,
        title      = title,
        **common_args,          # palette deliberately omitted
    )

    # -------------------------------------------------------------
    # 5.7 F1‑over‑dates (PLS‑DA)
    # -------------------------------------------------------------
    plot_f1_over_dates(
        results    = pls_results,
        output_dir = output_dir,
        title      = title,
        **common_args,
        **palette_args,
    )

    # -------------------------------------------------------------
    # 5.8 Trajectory plots (centroids + full trajectories)
    # -------------------------------------------------------------
    plot_trajectory_centroids(
        results    = traj_results,
        output_dir = output_dir,
        cols       = cols,
        title      = title,
        **common_args,
        **palette_args,
    )

    plot_trajectory(
        results    = traj_results,
        output_dir = output_dir,
        cols       = cols,
        title      = title,
        **common_args,
        **palette_args,
    )

    # -----------------------------------------------------------------
    # 6️⃣ Final combined metrics CSV + console summary
    # -----------------------------------------------------------------
    all_metrics = []
    for level1, dates in pls_results.items():
        for date, res in dates.items():
            all_metrics.append(res['metrics_df'])

    if all_metrics:
        combined = pd.concat(all_metrics, ignore_index=True)
        combined.to_csv(output_dir / f'{title}_all_metrics.csv', index=False)

        summary = (
            combined
            .groupby(['level1', 'date'])[['f1', 'precision', 'recall']]
            .agg(['mean', 'std'])
            .round(3)
        )
        print('\n── PLS‑DA Summary (Aggregated Plants) ───────────────────────────────')
        print(summary.to_string())
        print()

    log.info('Pipeline complete. All outputs saved to: %s', output_dir)

# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Run the hyperspectral analysis pipeline.')
    parser.add_argument('--input-folder', '-i', required=True, help='Path to folder with CSVs')
    parser.add_argument('--output-dir', '-o', default=None, help='Output directory')
    parser.add_argument('--title', '-t', default=None, help='Base name for files')
    parser.add_argument('--config', '-c', default=None, help='Path to YAML config')
    return parser.parse_args(argv)

def main(argv=None):
    args = _parse_args(argv)
    cfg = _load_config(args.config)
    _validate_config(cfg)

    input_folder = Path(args.input_folder).resolve()
    cfg['spectra_path']  = input_folder / 'stacked_spectra.csv'
    cfg['metadata_path'] = input_folder / 'stacked_metadata.csv'
    cfg['output_dir'] = Path(args.output_dir).resolve() if args.output_dir else input_folder
    cfg['title'] = args.title or input_folder.name

    run_pipeline(cfg)

if __name__ == '__main__':
    main()