"""
__main__.py

Entry point for the SVC sig file reader package.

Usage:
    python -m svc_reader
    python -m svc_reader --input /path/to/sig/files
    python -m svc_reader --input /path/to/sig/files --output /path/to/output --name my_site
    python -m svc_reader --input /path/to/sig/files --sm
    python -m svc_reader --input /path/to/sig/files --vi
    python -m svc_reader --input /path/to/sig/files --sm --vi

Output files:
    SVC_<name>_metadata.csv        always — Spec_scan, Datetime, Latitude, Longitude
    SVC_<name>.csv                 always — Spec_scan + reflectance, raw
    SVC_<name>_noOut.csv           always — raw, outliers removed
    SVC_<name>_noOut_cl.csv        only if NaN/Inf found — raw, outliers + NaN/Inf removed
    SVC_<name>_vi.csv              only with --vi — Spec_scan + VI, raw
    SVC_<name>_sm.csv              only with --sm — smoothed reflectance
    SVC_<name>_sm_noOut.csv        only with --sm — smoothed, outliers removed
    SVC_<name>_sm_noOut_cl.csv     only with --sm + NaN/Inf found
    SVC_<name>_sm_vi.csv           only with --sm and --vi — Spec_scan + VI, smoothed
"""

import os
import sys
import glob
import argparse

from svc_reader.importsvc import importsvc
from svc_reader.export import export_tables
from svc_reader.vi import narrowVI


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Read SVC HR-1024 .sig files and export reflectance spectra to CSV.\n"
            "Searches recursively for all files matching --pattern "
            "under the input directory."
        )
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=None,
        help="Root folder to search for .sig files. Defaults to current directory."
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help=(
            "Output directory for CSV files. "
            "Defaults to 'svc_output' inside the input directory."
        )
    )
    parser.add_argument(
        '--name', '-n',
        type=str,
        default='output',
        help="Base name for output CSV files (default: 'output')."
    )
    parser.add_argument(
        '--pattern', '-p',
        type=str,
        default='*.sig',
        help="Glob pattern for .sig files (default: '*.sig')."
    )
    parser.add_argument(
        '--sm',
        action='store_true',
        default=False,
        help='Apply Savitzky-Golay smoothing and export smoothed outputs.'
    )
    parser.add_argument(
        '--vi',
        action='store_true',
        default=False,
        help='Compute and export vegetation indices (SVC_<name>_vi.csv).'
    )
    return parser.parse_args()


def find_sig_folders(root_dir, pattern):
    """
    Find all unique folders containing files matching pattern.

    Parameters
    ----------
    root_dir : str
    pattern : str

    Returns
    -------
    folders : list of str
    all_files : list of str
    """
    all_files = glob.glob(
        os.path.join(root_dir, '**', pattern), recursive=True)
    folders = sorted(set(os.path.dirname(f) for f in all_files))
    return folders, all_files


def main():
    args = parse_args()

    # Resolve directories
    input_dir  = os.path.abspath(args.input if args.input else os.getcwd())
    output_dir = os.path.abspath(
        args.output if args.output
        else os.path.join(input_dir, 'svc_output')
    )

    if not os.path.isdir(input_dir):
        print(f"ERROR: Input directory not found:\n  {input_dir}")
        sys.exit(1)

    vi_func   = narrowVI if args.vi else None
    pattern   = args.pattern
    base_name = args.name

    print("=" * 60)
    print("SVC Sig File Reader — HR-1024")
    print("=" * 60)
    print(f"  Input directory   : {input_dir}")
    print(f"  Output directory  : {output_dir}")
    print(f"  File pattern      : {pattern}")
    print(f"  Output name       : {base_name}")
    print(f"  Smoothing         : {'Yes' if args.sm else 'No'}")
    print(f"  Vegetation indices: {'Yes' if args.vi else 'No'}")
    print("=" * 60)

    # Find files
    folders, all_files = find_sig_folders(input_dir, pattern)

    if not all_files:
        print(f"\nNo files matching '{pattern}' found under:\n  {input_dir}")
        sys.exit(1)

    print(f"\nFound {len(all_files)} file(s) in {len(folders)} folder(s).\n")

    # Load all files folder by folder
    all_targets    = []
    all_references = []

    for folder in folders:
        file_pattern = os.path.join(folder, pattern)
        print(f"Folder: {folder}")
        try:
            targets, references = importsvc(file_pattern)
            all_targets.extend(targets)
            all_references.extend(references)
            print(f"  Loaded {len(targets)} spectrum/spectra.\n")
        except FileNotFoundError as e:
            print(f"  WARNING: {e}\n")
        except ValueError as e:
            print(f"  WARNING: {e}\n")

    if not all_targets:
        print("No spectra were successfully loaded. Exiting.")
        sys.exit(1)

    print(f"Total spectra loaded: {len(all_targets)}\n")

    # Export
    export_tables(
        target=all_targets,
        reference=all_references,
        output_dir=output_dir,
        base_name=base_name,
        smooth=args.sm,
        vi_func=vi_func,
    )


if __name__ == '__main__':
    main()