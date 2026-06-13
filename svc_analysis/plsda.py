"""
svc_analysis/plsda.py
=====================
PLS-DA classification for the hyperspectral analysis pipeline.

Runs per level1 (parent group) and per date, classifying level3
(most granular) within each level1.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    LeaveOneOut,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, StandardScaler

log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_spectral_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith('w_')]


def _wavelengths_from_cols(cols: list[str]) -> np.ndarray:
    return np.array([float(c.split('_')[1]) for c in cols])


def _vip_scores(model: PLSRegression, X: np.ndarray) -> np.ndarray:
    t     = model.x_scores_
    w     = model.x_weights_
    q     = model.y_loadings_
    n, p  = X.shape
    _, h  = t.shape
    vip   = np.zeros(p)
    s     = np.diag(t.T @ t @ q.T @ q)
    total = np.sum(s)
    for i in range(p):
        weight = np.array([
            (w[i, j] / np.linalg.norm(w[:, j])) ** 2
            for j in range(h)
        ])
        vip[i] = np.sqrt(p * (s @ weight) / total)
    return vip


# ── Main function ──────────────────────────────────────────────────────────────

def run_plsda(
    spectra_df:    pd.DataFrame,
    metadata_df:   pd.DataFrame,
    cols:          dict,
    output_dir:    str | Path,
    cv:            str = 'stratified',
    n_folds:       int = 5,
    n_iter_search: int = 20,
    random_state:  int = 42,
) -> dict:
    """
    Run PLS-DA for each level1 and date, classifying level3.

    Parameters
    ----------
    spectra_df : pd.DataFrame
    metadata_df : pd.DataFrame
    cols : dict
        Column mapping with keys: level1, level2, level3, scan_id, date.
    output_dir : str or Path
    cv : str
    n_folds : int
    n_iter_search : int
    random_state : int

    Returns
    -------
    results : dict
        results[level1][date] with keys:
            'metrics_df'       — per-fold metrics DataFrame
            'confusion_matrix' — aggregated confusion matrix
            'classes'          — class labels (level3 values)
            'vip_df'           — VIP scores DataFrame (wavelength, vip, mean_coef)
            'model'            — fitted PLSRegression (final full-data fit)
            'X'                — spectral matrix used
            'y'                — label array used
    """

    output_dir  = Path(output_dir)
    spec_cols   = _get_spectral_cols(spectra_df)
    wavelengths = _wavelengths_from_cols(spec_cols)

    l1_col   = cols['level1']
    l2_col   = cols['level2']
    l3_col   = cols['level3']
    scan_col = cols['scan_id']
    date_col = cols['date']

    # ── Merge spectra + metadata ───────────────────────────────────────────────
    merged = spectra_df.merge(metadata_df, on=scan_col, how='inner')
    log.info('Merged shape: %s', merged.shape)

    results: dict = {}

    for l1_val, l1_group in merged.groupby(l1_col):
        log.info('── Level1: %s ─────────────────────────────────', l1_val)
        results[l1_val] = {}

        for date_val, date_group in l1_group.groupby(date_col):
            log.info('   Date: %s  (n=%d)', date_val, len(date_group))

            X      = date_group[spec_cols].values.astype(float)
            y      = date_group[l3_col].values
            groups = date_group[l2_col].values

            classes = np.unique(y)
            n_cls   = len(classes)

            if n_cls < 2:
                log.warning(
                    '   Skipping %s / %s — only one class present (%s)',
                    l1_val, date_val, classes[0],
                )
                continue

            if len(X) < 4:
                log.warning(
                    '   Skipping %s / %s — too few samples (%d)',
                    l1_val, date_val, len(X),
                )
                continue

            # ── Cross-validation setup ─────────────────────────────────────────
            if cv == 'loo':
                splitter = LeaveOneOut()
                log.info('   CV: Leave-One-Out')
            else:
                actual_folds = min(n_folds, len(X))
                splitter     = StratifiedKFold(
                    n_splits     = actual_folds,
                    shuffle      = True,
                    random_state = random_state,
                )
                log.info('   CV: StratifiedKFold (k=%d)', actual_folds)

            # ── Fit binarizer once on full y ───────────────────────────────────
            lb = LabelBinarizer()
            lb.fit(y)                          # fix 1: fitted on full y, not per-fold

            # ── Hyperparameter search via inner CV ─────────────────────────────
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('pls',    PLSRegression(max_iter=500)),
            ])

            n_comp_max = min(X.shape[0] - 1, X.shape[1], 20)
            param_dist = {'pls__n_components': list(range(2, n_comp_max + 1))}

            # fix 2: guard inner_cv splits against smallest class count
            min_class_count = int(min(np.sum(y == c) for c in classes))
            inner_cv = StratifiedKFold(
                n_splits     = min(3, min_class_count),
                shuffle      = True,
                random_state = random_state,
            )

            search = RandomizedSearchCV(
                estimator           = pipe,
                param_distributions = param_dist,
                n_iter              = min(n_iter_search, n_comp_max - 1),
                cv                  = inner_cv,
                scoring             = 'f1_macro',
                n_jobs              = -1,
                random_state        = random_state,
            )

            try:
                search.fit(X, y)
                best_n = search.best_params_['pls__n_components']
                log.info('   Best n_components: %d', best_n)
            except Exception as e:
                log.warning('   Hyperparameter search failed (%s), using n=2', e)
                best_n = 2

            # ── Outer CV loop ──────────────────────────────────────────────────
            fold_records = []
            cm_accum     = np.zeros((n_cls, n_cls), dtype=int)
            scaler       = StandardScaler()
            pls_model    = PLSRegression(n_components=best_n, max_iter=500)

            for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
                X_tr, X_te = X[train_idx], X[test_idx]
                y_tr, y_te = y[train_idx], y[test_idx]

                try:
                    X_tr_s = scaler.fit_transform(X_tr)
                    X_te_s = scaler.transform(X_te)

                    Y_tr_b = lb.transform(y_tr)   # consistent with full-y binarizer
                    pls_model.fit(X_tr_s, Y_tr_b)

                    Y_pred_b = pls_model.predict(X_te_s)
                    y_pred   = lb.classes_[np.argmax(Y_pred_b, axis=1)]

                    fold_cm   = confusion_matrix(y_te, y_pred, labels=classes)
                    cm_accum += fold_cm

                    fold_records.append({
                        'level1':    l1_val,
                        'date':      date_val,
                        'fold':      fold_idx,
                        'f1':        f1_score(y_te, y_pred, average='macro', zero_division=0),
                        'precision': precision_score(y_te, y_pred, average='macro', zero_division=0),
                        'recall':    recall_score(y_te, y_pred, average='macro', zero_division=0),
                        'n_train':   len(train_idx),
                        'n_test':    len(test_idx),
                    })

                except Exception as e:
                    log.warning('   Fold %d failed: %s', fold_idx, e)
                    continue

            if not fold_records:
                log.warning(
                    '   No valid folds for %s / %s — skipping.',
                    l1_val, date_val,
                )
                continue

            metrics_df = pd.DataFrame(fold_records)

            # ── Final full-data model for VIP scores ───────────────────────────
            try:
                X_all_s     = scaler.fit_transform(X)
                Y_all_b     = lb.transform(y)
                final_model = PLSRegression(n_components=best_n, max_iter=500)
                final_model.fit(X_all_s, Y_all_b)

                vip   = _vip_scores(final_model, X_all_s)
                coefs = (
                    np.abs(final_model.coef_).mean(axis=1)
                    if final_model.coef_.ndim > 1
                    else np.abs(final_model.coef_).ravel()
                )

                vip_df = pd.DataFrame({
                    'wavelength': wavelengths,
                    'vip':        vip,
                    'mean_coef':  coefs[:len(wavelengths)],
                })

            except Exception as e:
                log.warning('   Final model fit failed (%s); VIP unavailable.', e)
                final_model = None
                vip_df      = pd.DataFrame(columns=['wavelength', 'vip', 'mean_coef'])

            # ── Save per-level1/date artifacts ─────────────────────────────────
            safe_l1   = str(l1_val).replace(' ', '_').replace('/', '_').replace('\\', '_')
            safe_date = str(date_val).replace('-', '').replace('/', '_').replace('\\', '_')
            stem      = f'{safe_l1}_{safe_date}'

            metrics_df.to_csv(output_dir / f'{stem}_metrics.csv', index=False)
            vip_df.to_csv(output_dir / f'{stem}_vip.csv', index=False)
            np.savetxt(
                output_dir / f'{stem}_confusion.csv',
                cm_accum,
                delimiter = ',',
                fmt       = '%d',
                header    = ','.join(classes),
            )

            log.info(
                '   F1 mean=%.3f  std=%.3f',
                metrics_df['f1'].mean(),
                metrics_df['f1'].std(),
            )

            results[l1_val][date_val] = {
                'metrics_df':       metrics_df,
                'confusion_matrix': cm_accum,
                'classes':          classes,
                'vip_df':           vip_df,
                'model':            final_model,
                'scaler':           scaler,
                'label_binarizer':  lb,
                'X':                X,
                'y':                y,
                'groups':           groups,
                'n_components':     best_n,
            }

    return results