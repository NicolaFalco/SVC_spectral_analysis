"""
Header parsers for SVC instrument types.
Currently supports: HR-1024
"""

import re
import warnings
from datetime import datetime

from .utils import regex_str, regex_float, regex_int
from .gps_parser import parse_gps


def parse_header_hr1024(file_content):
    """Parse header from an HR-1024 .sig file."""
    T = {}
    R = {}

    # Name and type
    T['Name'] = regex_str(r'^name= (.*?)$', file_content)
    R['Name'] = T['Name']
    T['Type'] = 'Target'
    R['Type'] = 'Reference'

    # Instrument info
    T['InstrumentModel'] = 'HR-1024'
    R['InstrumentModel'] = 'HR-1024'
    T['InstrumentManufacturer'] = 'Spectra Vista Corporation'
    R['InstrumentManufacturer'] = 'Spectra Vista Corporation'
    serial = regex_str(r'^instrument=\s*?HR:\s*?(\d+)\s*?$', file_content)
    T['InstrumentSerialNumber'] = serial
    R['InstrumentSerialNumber'] = serial

    # Date and time
    T, R = _parse_datetime_hr1024(file_content, T, R)
    T['DateTimeSource'] = 'Unknown'
    R['DateTimeSource'] = 'Unknown'

    # Memory slots
    T['MemorySlot'] = regex_int(r'^memory slot=.*?,\s*?(\d+).*?$', file_content)
    R['MemorySlot'] = regex_int(r'^memory slot=\s*?(\d+).*?,', file_content)

    if T['MemorySlot'] == 0 and R['MemorySlot'] == 0:
        T['AcquisitionDevice'] = 'Computer or PDA'
        R['AcquisitionDevice'] = 'Computer or PDA'
    else:
        T['AcquisitionDevice'] = 'Spectrometer memory slot'
        R['AcquisitionDevice'] = 'Spectrometer memory slot'

    # Scan time
    T['ScanTime'] = regex_float(r'^scan time= .*?,\s?([\.\d]+)', file_content)
    R['ScanTime'] = regex_float(r'^scan time=\s*([\.\d]+),', file_content)
    T['ScanTimeUnits'] = 's'
    R['ScanTimeUnits'] = 's'

    # Integration times: R_VNIR, R_SWIR1, R_SWIR2, T_VNIR, T_SWIR1, T_SWIR2
    int_pat = (r'^integration= ([\.\d]+), ([\.\d]+), ([\.\d]+),'
               r' ([\.\d]+), ([\.\d]+), ([\.\d]+)\s*?$')
    int_m = re.search(int_pat, file_content, re.MULTILINE)
    if int_m:
        g = int_m.groups()
        R['IntegrationTimeVNIR']  = float(g[0])
        R['IntegrationTimeSWIR1'] = float(g[1])
        R['IntegrationTimeSWIR2'] = float(g[2])
        T['IntegrationTimeVNIR']  = float(g[3])
        T['IntegrationTimeSWIR1'] = float(g[4])
        T['IntegrationTimeSWIR2'] = float(g[5])
    else:
        for key in ['IntegrationTimeVNIR', 'IntegrationTimeSWIR1',
                    'IntegrationTimeSWIR2']:
            T[key] = float('nan')
            R[key] = float('nan')
    T['IntegrationTimeUnits'] = 'ms'
    R['IntegrationTimeUnits'] = 'ms'

    # Scan settings: format= ref_dark, ref_int, tar_dark, tar_int
    dark_type_map = {'AD': 'auto', 'SD': 'scaled', 'UD': 'unknown'}
    int_type_map  = {'AI': 'auto', 'FI': 'fixed',  'UI': 'unknown'}

    ss_m = re.search(
        r'scan settings= (\w+), (\w+), (\w+), (\w+)\s*?$',
        file_content, re.MULTILINE)
    if ss_m:
        r_dark, r_int, t_dark, t_int = ss_m.groups()
        R['DarkType']        = dark_type_map.get(r_dark, 'unknown')
        T['DarkType']        = dark_type_map.get(t_dark, 'unknown')
        R['IntegrationType'] = int_type_map.get(r_int, 'unknown')
        T['IntegrationType'] = int_type_map.get(t_int, 'unknown')
    else:
        for d in [T, R]:
            d['DarkType'] = 'unknown'
            d['IntegrationType'] = 'unknown'

    # ------------------------------------------------------------------ #
    # External data sets                                                   #
    # Your .sig files use comma-separated values (not space-separated),   #
    # and the values are all zeros stored as 32 entries per set.          #
    # We split on commas and store R = first half, T = second half.       #
    # ------------------------------------------------------------------ #
    for ds_key, ds_pat in [
        ('ExternalDataSet1', r'^external data set1= (.*?)$'),
        ('ExternalDataSet2', r'^external data set2= (.*?)$'),
    ]:
        ds_m = re.search(ds_pat, file_content, re.MULTILINE)
        if ds_m:
            raw = ds_m.group(1).strip()
            try:
                # handle both comma-separated and space-separated formats
                if ',' in raw:
                    vals = [float(x) for x in raw.split(',')]
                else:
                    vals = [float(x) for x in raw.split()]
                half = len(vals) // 2
                R[ds_key] = vals[:half]
                T[ds_key] = vals[half:]
            except ValueError:
                R[ds_key] = []
                T[ds_key] = []
        else:
            R[ds_key] = []
            T[ds_key] = []

    # External dark and mask (also comma-separated in your files)
    for ext_key, ext_pat in [
        ('ExternalDataDark', r'^external data dark= (.*?)$'),
        ('ExternalDataMask', r'^external data mask= (.*?)$'),
    ]:
        ext_m = re.search(ext_pat, file_content, re.MULTILINE)
        if ext_m:
            raw = ext_m.group(1).strip()
            try:
                if ',' in raw:
                    vals = [float(x) for x in raw.split(',')]
                else:
                    vals = [float(x) for x in raw.split()]
                R[ext_key] = vals
                T[ext_key] = vals
            except ValueError:
                R[ext_key] = []
                T[ext_key] = []
        else:
            R[ext_key] = 'Data missing'
            T[ext_key] = 'Data missing'

    # Fore optic
    R['ForeOptic'] = regex_str(r'^optic=\s*?(\S.*?),', file_content)
    T['ForeOptic'] = regex_str(r'^optic=.*?,\s*?(\S.*?)$', file_content)

    # Detector temperatures: R_VNIR, R_SWIR1, R_SWIR2, T_VNIR, T_SWIR1, T_SWIR2
    temp_m = re.search(r'^temp=(.*?)$', file_content, re.MULTILINE)
    if temp_m:
        try:
            temps = [float(x) for x in temp_m.group(1).split(',')]
            if len(temps) >= 6:
                R['TemperatureVNIRDetector']  = temps[0]
                R['TemperatureSWIR1Detector'] = temps[1]
                R['TemperatureSWIR2Detector'] = temps[2]
                T['TemperatureVNIRDetector']  = temps[3]
                T['TemperatureSWIR1Detector'] = temps[4]
                T['TemperatureSWIR2Detector'] = temps[5]
            else:
                for d in [T, R]:
                    d['TemperatureVNIRDetector']  = float('nan')
                    d['TemperatureSWIR1Detector'] = float('nan')
                    d['TemperatureSWIR2Detector'] = float('nan')
        except ValueError:
            for d in [T, R]:
                d['TemperatureVNIRDetector']  = float('nan')
                d['TemperatureSWIR1Detector'] = float('nan')
                d['TemperatureSWIR2Detector'] = float('nan')
    else:
        for d in [T, R]:
            d['TemperatureVNIRDetector']  = float('nan')
            d['TemperatureSWIR1Detector'] = float('nan')
            d['TemperatureSWIR2Detector'] = float('nan')

    # Battery voltages
    R['BatteryVoltage'] = regex_float(r'^battery=\s*?([\.\+\d]+),', file_content)
    T['BatteryVoltage'] = regex_float(r'^battery=.*?,\s*?([\.\+\d]+)', file_content)

    # Error codes
    R['ErrorCode'] = regex_int(r'^error=\s*(\d+),', file_content)
    T['ErrorCode'] = regex_int(r'^error=.*?,\s*?(\d+)\s*?$', file_content)
    if R['ErrorCode'] != 0 or T['ErrorCode'] != 0:
        warnings.warn(
            f"Spectrometer reported an error whilst recording '{T['Name']}'.")

    # Overlap handling
    T, R = _parse_overlap(file_content, T, R)

    # GPS
    T, R = parse_gps(file_content, T, R)

    # Comments
    comm = regex_str(r'comm=(.*?)$', file_content, default='')
    T['Comments'] = comm
    R['Comments'] = comm

    return T, R


def _parse_datetime_hr1024(file_content, T, R):
    """Parse datetime for HR-1024 files (American or British format)."""

    # ------------------------------------------------------------------ #
    # American format with AM/PM                                           #
    # Your files look like: 5/27/2026 3:34:07 AM                         #
    # The original code used %y (2-digit year) — fixed to %Y (4-digit)   #
    # ------------------------------------------------------------------ #
    if re.search(
            r'^time= \d+/\d+/\d+ \d+:\d+:\d+ (?:AM|PM)',
            file_content, re.MULTILINE):
        r_dt_m = re.search(
            r'^time= (\d+/\d+/\d+ \d+:\d+:\d+ .M),',
            file_content, re.MULTILINE)
        t_dt_m = re.search(
            r'^time= .*?, (\d+/\d+/\d+ \d+:\d+:\d+ .M)',
            file_content, re.MULTILINE)
        try:
            R['DateTime'] = datetime.strptime(
                r_dt_m.group(1).strip(), "%m/%d/%Y %I:%M:%S %p"  # %Y not %y
            ).strftime("%d-%b-%Y %H:%M:%S")
            T['DateTime'] = datetime.strptime(
                t_dt_m.group(1).strip(), "%m/%d/%Y %I:%M:%S %p"  # %Y not %y
            ).strftime("%d-%b-%Y %H:%M:%S")
        except (AttributeError, ValueError):
            warnings.warn("Could not parse datetime (American format).")
            T['DateTime'] = 'Unknown'
            R['DateTime'] = 'Unknown'

    # British format (DD/MM/YYYY HH:MM:SS, no AM/PM)
    elif re.search(
            r'^time= \d+/\d+/\d+ \d+:\d+:\d+',
            file_content, re.MULTILINE):
        r_dt_m = re.search(
            r'^time= (\d+/\d+/\d+ \d+:\d+:\d+),',
            file_content, re.MULTILINE)
        t_dt_m = re.search(
            r'^time= .*?, (\d+/\d+/\d+ \d+:\d+:\d+)',
            file_content, re.MULTILINE)
        try:
            R['DateTime'] = datetime.strptime(
                r_dt_m.group(1).strip(), "%d/%m/%Y %H:%M:%S"
            ).strftime("%d-%b-%Y %H:%M:%S")
            T['DateTime'] = datetime.strptime(
                t_dt_m.group(1).strip(), "%d/%m/%Y %H:%M:%S"
            ).strftime("%d-%b-%Y %H:%M:%S")
        except (AttributeError, ValueError):
            warnings.warn("Could not parse datetime (British format).")
            T['DateTime'] = 'Unknown'
            R['DateTime'] = 'Unknown'
    else:
        warnings.warn("Date and time format was not recognised.")
        T['DateTime'] = 'Unknown'
        R['DateTime'] = 'Unknown'

    return T, R


def _parse_overlap(file_content, T, R):
    """Parse overlap handling fields for HR-1024 files."""

    if re.search(r'[Oo]verlap: ?[Rr]emove', file_content):
        R['OverlapDataHandling'] = 'Removed'
        T['OverlapDataHandling'] = 'Removed'
        trans_wave = regex_float(
            r'[Oo]verlap: ?[Rr]emove ?@ ?(\d+)', file_content)
        R['OverlapTransitionWavelength'] = trans_wave
        T['OverlapTransitionWavelength'] = trans_wave
    elif re.search(r'[Oo]verlap: ?[Pp]reserve', file_content):
        R['OverlapDataHandling'] = 'Preserved'
        T['OverlapDataHandling'] = 'Preserved'
        R['OverlapTransitionWavelength'] = 'Not applicable'
        T['OverlapTransitionWavelength'] = 'Not applicable'
    else:
        R['OverlapDataHandling'] = 'Unknown'
        T['OverlapDataHandling'] = 'Unknown'
        R['OverlapTransitionWavelength'] = 'Unknown'
        T['OverlapTransitionWavelength'] = 'Unknown'

    # Matching type
    mt_m = re.search(r'Matching Type: ?(\w+)', file_content)
    if mt_m:
        matching_type = mt_m.group(1)
        R['OverlapMatchingType'] = matching_type
        T['OverlapMatchingType'] = matching_type
    else:
        R['OverlapMatchingType'] = 'Unknown'
        T['OverlapMatchingType'] = 'Unknown'
        matching_type = 'Unknown'

    # Matching region
    if matching_type in ('Reflectance', 'Radiance'):
        region_start = regex_float(
            r'Matching Type: \w+ @ (\d+)', file_content)
        region_end = regex_float(
            r'Matching Type: \w+ @ \d+ - (\d+)', file_content)
        R['OverlapMatchingRegionWavelengthRange'] = [region_start, region_end]
        T['OverlapMatchingRegionWavelengthRange'] = [region_start, region_end]
    else:
        R['OverlapMatchingRegionWavelengthRange'] = 'Not applicable'
        T['OverlapMatchingRegionWavelengthRange'] = 'Not applicable'

    # NIR-SWIR algorithm
    if re.search(r'NIR-SWIR [Oo]ff', file_content):
        R['OverlapNIRSWIRAlgorithmEnabled'] = 'No'
        T['OverlapNIRSWIRAlgorithmEnabled'] = 'No'
    elif re.search(r'NIR-SWIR [Oo]n', file_content):
        R['OverlapNIRSWIRAlgorithmEnabled'] = 'Yes'
        T['OverlapNIRSWIRAlgorithmEnabled'] = 'Yes'
    else:
        R['OverlapNIRSWIRAlgorithmEnabled'] = 'Unknown'
        T['OverlapNIRSWIRAlgorithmEnabled'] = 'Unknown'

    # Matching factors
    ref_mf_m = re.search(
        r'factors= ([-\+\.\d]+), [-\+\.\d]+, [-\+\.\d]+', file_content)
    tar_mf_m = re.search(
        r'factors= [-\+\.\d]+, ([-\+\.\d]+), [-\+\.\d]+', file_content)
    R['OverlapMatchingFactor'] = (
        float(ref_mf_m.group(1)) if ref_mf_m else 'Not applicable')
    T['OverlapMatchingFactor'] = (
        float(tar_mf_m.group(1)) if tar_mf_m else 'Not applicable')

    return T, R