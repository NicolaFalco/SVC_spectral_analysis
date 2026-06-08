"""
svc_reader

A Python package for reading SVC HR-1024i/HR-512i spectroradiometer
.sig files and exporting reflectance spectra to CSV.

Modules
-------
importsvc   — parse raw .sig files into structured Python dicts
export      — build reflectance matrices, smooth, detect outliers, save CSVs
vi          — narrow-band vegetation index library (optional)
"""

from .importsvc import importsvc
from .export import (
    build_reflectance_matrix,
    smooth_reflectance,
    build_metadata_table,
    build_reflectance_table,
    detect_outliers_mean,
    remove_outliers,
    remove_nan_inf,
    export_tables,
)
from .vi import narrowVI

__all__ = [
    "importsvc",
    "build_reflectance_matrix",
    "smooth_reflectance",
    "build_metadata_table",
    "build_reflectance_table",
    "detect_outliers_mean",
    "remove_outliers",
    "remove_nan_inf",
    "export_tables",
    "narrowVI",
]