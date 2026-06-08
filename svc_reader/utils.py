"""
Utility functions for SVC sig file reader.
"""

import os
import re


def regex_str(pattern, text, default='Unknown'):
    """Extract a single string match or return default."""
    m = re.search(pattern, text, re.MULTILINE)
    return m.group(1).strip() if m else default


def regex_float(pattern, text, default=float('nan')):
    """Extract a single float match or return default."""
    m = re.search(pattern, text, re.MULTILINE)
    try:
        return float(m.group(1)) if m else default
    except (ValueError, AttributeError):
        return default


def regex_int(pattern, text, default=0):
    """Extract a single int match or return default."""
    m = re.search(pattern, text, re.MULTILINE)
    try:
        return int(m.group(1)) if m else default
    except (ValueError, AttributeError):
        return default


def fileparts(path):
    """
    Replicate MATLAB fileparts().
    Returns (folder, name_without_ext, extension).
    Handles wildcard paths like /some/folder/*.sig
    """
    folder = os.path.dirname(path)
    base   = os.path.basename(path)
    name, ext = os.path.splitext(base)
    return folder, name, ext