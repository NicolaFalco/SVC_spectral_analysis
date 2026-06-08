"""
importsvc.py

Import SVC HR-1024 .sig files and return target and reference
spectrum dictionaries.
"""

import os
import re
import glob
import numpy as np

from svc_reader.utils import fileparts
from svc_reader.header_parsers import parse_header_hr1024


def importsvc(file_names):
    """
    Import SVC HR-1024 .sig files into Python data structures.

    Parameters
    ----------
    file_names : str
        Path to a single file or wildcard pattern (e.g. '/data/*moc.sig').

    Returns
    -------
    target : list of dict
        Each dict contains: name, datetime, header, wavelength, data, pair.
    reference : list of dict
        Each dict contains: name, datetime, header, wavelength, data.
    """
    folder, name_part, ext_part = fileparts(file_names)
    pattern = (os.path.join(folder, name_part + ext_part)
               if folder else name_part + ext_part)
    matching = sorted(glob.glob(pattern))

    list_of_files = [f for f in matching if os.path.isfile(f)]

    if not list_of_files:
        raise FileNotFoundError(
            f"No files were found which match:\n\t{file_names}")

    target = []
    reference = []

    for file_path in list_of_files:
        file_name = os.path.basename(file_path)
        file_name_without_ext = os.path.splitext(file_name)[0]

        print(f"  Importing: {file_name}")

        with open(file_path, 'r', errors='replace') as f:
            file_content = f.read().replace('\r', '')

        # Only HR-1024 supported
        if re.search(r'HR-1024', file_content):
            T, R = parse_header_hr1024(file_content)
            # HR-1024 column order: wavelength, reference, target, (4th ignored)
            wavelength, reference_data, target_data = _parse_data_columns(
                file_content, col_order='wrt')
        else:
            print(f"  SKIPPING {file_name}: not recognised as HR-1024 format.")
            continue

        spectrum_name = file_name_without_ext

        reference.append({
            'name':       spectrum_name + '_reference',
            'datetime':   R.get('DateTime', 'Unknown'),
            'header':     R,
            'wavelength': wavelength,
            'data':       reference_data,
        })

        target.append({
            'name':       spectrum_name,
            'datetime':   T.get('DateTime', 'Unknown'),
            'header':     T,
            'pair':       spectrum_name + '_reference',
            'wavelength': wavelength,
            'data':       target_data,
        })

    return target, reference


def _parse_data_columns(file_content, col_order='wrt'):
    """
    Parse numeric spectral data columns from a .sig file.

    The data section contains 4 columns. For HR-1024:
        col 0: wavelength
        col 1: reference radiance
        col 2: target radiance
        col 3: reflectance (ignored — we compute it ourselves)

    Parameters
    ----------
    file_content : str
    col_order : str
        'wrt' = wavelength, reference, target
        'wtr' = wavelength, target, reference

    Returns
    -------
    wavelength : np.ndarray
    reference_data : np.ndarray
    target_data : np.ndarray
    """
    data_pattern = re.compile(
        r'^\s*([\.\d]+)\s+([-\.\d]+)\s+([-\.\d]+)\s+([-\.\d]+)',
        re.MULTILINE
    )
    matches = data_pattern.findall(file_content)

    if not matches:
        raise ValueError("No spectral data found in file.")

    col0 = np.array([float(m[0]) for m in matches])
    col1 = np.array([float(m[1]) for m in matches])
    col2 = np.array([float(m[2]) for m in matches])

    wavelength = col0
    if col_order == 'wrt':
        reference_data = col1
        target_data    = col2
    else:
        target_data    = col1
        reference_data = col2

    return wavelength, reference_data, target_data