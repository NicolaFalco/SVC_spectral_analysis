"""
GPS parsing logic for SVC .sig files.
"""

import re
import warnings


def parse_gps(file_content, T, R):
    """
    Parse GPS coordinates from a .sig file.

    Parameters
    ----------
    file_content : str
    T : dict  — target header (modified in place)
    R : dict  — reference header (modified in place)

    Returns
    -------
    T, R : dict
    """
    # GPS inactive: longitude field contains only whitespace
    if re.search(r'^longitude=\s+,\s+$', file_content, re.MULTILINE):
        for d in [T, R]:
            d['GPSActive'] = 'No'
            d['GPSCoordinates'] = [float('nan'), float('nan')]
        _parse_gps_time(file_content, T, R)
        return T, R

    # Reference GPS (first value on each line)
    if re.search(r'^longitude=.*?\d+.*?$', file_content, re.MULTILINE):
        R['GPSActive'] = 'Yes'
        ref_lon_deg = _gps_float(r'^longitude=\s*(\d{3})\d{2}\.\d+', file_content)
        ref_lon_min = _gps_float(r'^longitude=\s*\d{3}(\d{2}\.\d+)', file_content)
        ref_lon_dir = _gps_str(r'^longitude=\s*[\d\.]+,([EW]),', file_content)
        ref_lat_deg = _gps_float(r'^latitude=\s*(\d{2})\d{2}\.\d+', file_content)
        ref_lat_min = _gps_float(r'^latitude=\s*\d{2}(\d{2}\.\d+)', file_content)
        ref_lat_dir = _gps_str(r'^latitude=\s*[\d\.]+,([NS]),', file_content)

        if any(v is None for v in [ref_lon_deg, ref_lon_min,
                                    ref_lat_deg, ref_lat_min]):
            warnings.warn("Could not parse reference GPS coordinates.")
            R['GPSCoordinates'] = [float('nan'), float('nan')]
        else:
            lon_sign = 1 if ref_lon_dir == 'E' else -1
            lat_sign = 1 if ref_lat_dir == 'N' else -1
            R['GPSCoordinates'] = [
                lat_sign * (ref_lat_deg + ref_lat_min / 60.0),
                lon_sign * (ref_lon_deg + ref_lon_min / 60.0)
            ]
    else:
        R['GPSActive'] = 'No'
        R['GPSCoordinates'] = [float('nan'), float('nan')]

    # Target GPS (second value on each line, after comma separator)
    if re.search(r'^longitude=.*?,\s*\d+', file_content, re.MULTILINE):
        T['GPSActive'] = 'Yes'
        tar_lon_deg = _gps_float(
            r'^longitude=\s*[\d\.]+[EW]\s*,\s*(\d{3})\d{2}\.\d+', file_content)
        tar_lon_min = _gps_float(
            r'^longitude=\s*[\d\.]+[EW]\s*,\s*\d{3}(\d{2}\.\d+)', file_content)
        tar_lon_dir = _gps_str(
            r'^longitude=\s*[\d\.]+[EW]\s*,\s*[\d\.]+([EW])', file_content)
        tar_lat_deg = _gps_float(
            r'^latitude=\s*[\d\.]+[NS]\s*,\s*(\d{2})\d{2}\.\d+', file_content)
        tar_lat_min = _gps_float(
            r'^latitude=\s*[\d\.]+[NS]\s*,\s*\d{2}(\d{2}\.\d+)', file_content)
        tar_lat_dir = _gps_str(
            r'^latitude=\s*[\d\.]+[NS]\s*,\s*[\d\.]+([NS])', file_content)

        if any(v is None for v in [tar_lon_deg, tar_lon_min,
                                    tar_lat_deg, tar_lat_min]):
            warnings.warn("Could not parse target GPS coordinates.")
            T['GPSCoordinates'] = [float('nan'), float('nan')]
        else:
            lon_sign = 1 if tar_lon_dir == 'E' else -1
            lat_sign = 1 if tar_lat_dir == 'N' else -1
            T['GPSCoordinates'] = [
                lat_sign * (tar_lat_deg + tar_lat_min / 60.0),
                lon_sign * (tar_lon_deg + tar_lon_min / 60.0)
            ]
    else:
        T['GPSActive'] = 'No'
        T['GPSCoordinates'] = [float('nan'), float('nan')]

    _parse_gps_time(file_content, T, R)
    return T, R


def _parse_gps_time(file_content, T, R):
    """Parse GPS time fields (HHMMSS format)."""
    ref_m = re.search(
        r'^gpstime=\s*(\d{2})(\d{2})([\d\.]+)',
        file_content, re.MULTILINE)
    R['GPSTime'] = (f"{ref_m.group(1)}:{ref_m.group(2)}:{ref_m.group(3)}Z"
                    if ref_m else 'Unknown')

    tar_m = re.search(
        r'^gpstime=\s*[\d\.]+\s*,\s*(\d{2})(\d{2})([\d\.]+)',
        file_content, re.MULTILINE)
    T['GPSTime'] = (f"{tar_m.group(1)}:{tar_m.group(2)}:{tar_m.group(3)}Z"
                    if tar_m else 'Unknown')


def _gps_float(pattern, text):
    """Extract GPS float, returning None if not found."""
    m = re.search(pattern, text, re.MULTILINE)
    try:
        return float(m.group(1)) if m else None
    except (ValueError, AttributeError):
        return None


def _gps_str(pattern, text):
    """Extract GPS direction string, returning empty string if not found."""
    m = re.search(pattern, text, re.MULTILINE)
    return m.group(1).strip() if m else ''