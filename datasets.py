"""
Loaders for SpO2 testing.

Only dataset_primer_1 carries the dual-wavelength PPG (PPG_Red + PPG_IR) and a
ground-truth SpO2 that this algorithm needs. BIDMC has a single PLETH channel
(one wavelength), so the ratio-of-ratios SpO2 cannot be computed from it.
"""
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
PRIMER_DIR = os.path.join(_ROOT, "dataset_primer_1")
PHYSIONET_DIR = os.path.join(
    _ROOT, "physionet.org", "files", "pulse-transit-time-ppg", "1.1.0")


@dataclass
class PpgRecord:
    name: str
    red: np.ndarray
    ir: np.ndarray
    fs: float
    time: np.ndarray
    spo2_ref: Optional[float]      # ground-truth SpO2 (%) from metadata

    @property
    def duration(self) -> float:
        return float(self.time[-1] - self.time[0])


def _fs_from_time(time: np.ndarray) -> float:
    time = np.asarray(time, dtype=float)
    return float((time.size - 1) / (time[-1] - time[0]))


def _to_float(value) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_primer_record(ppg_path: str) -> PpgRecord:
    name = os.path.basename(ppg_path).replace("_ppg.csv", "")
    df = pd.read_csv(ppg_path)
    df.columns = df.columns.str.strip()
    time = df["Time (s)"].to_numpy(dtype=float)
    red = df["PPG_Red"].to_numpy(dtype=float)
    ir = df["PPG_IR"].to_numpy(dtype=float)

    fs = _fs_from_time(time)
    spo2 = None
    meta_path = ppg_path.replace("_ppg.csv", "_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            meta = json.load(fh)
        spo2 = _to_float(meta.get("ground_truth", {}).get("oxygen_saturation_percent"))
        specs = meta.get("signal_specs", {}).get("ppg", {})
        if specs.get("sampling_rate_hz"):
            fs = float(specs["sampling_rate_hz"])

    return PpgRecord(name, red, ir, fs, time, spo2)


def load_primer(limit: Optional[int] = None) -> List[PpgRecord]:
    paths = sorted(glob.glob(os.path.join(PRIMER_DIR, "*_ppg.csv")))
    if limit:
        paths = paths[:limit]
    return [load_primer_record(p) for p in paths]


# --------------------------------------------------------------------------- #
# PhysioNet "Pulse Transit Time PPG" dataset (WFDB)
#   pleth_1 = RED (660 nm), pleth_2 = IR (880 nm), distal finger, fs = 500 Hz.
#   Ground-truth SpO2 is the mean of <spo2_start> and <spo2_end> in the header.
# --------------------------------------------------------------------------- #
def load_physionet_record(record_base: str) -> PpgRecord:
    """Load one WFDB record by its base path (no extension)."""
    import wfdb  # imported lazily so the primer path has no wfdb dependency

    name = os.path.basename(record_base)
    rec = wfdb.rdrecord(record_base, channel_names=["pleth_1", "pleth_2"])
    fs = float(rec.fs)
    red = rec.p_signal[:, rec.sig_name.index("pleth_1")].astype(float)
    ir = rec.p_signal[:, rec.sig_name.index("pleth_2")].astype(float)
    time = np.arange(red.size) / fs

    comment = " ".join(rec.comments)
    vals = [int(m) for m in re.findall(r"<spo2_(?:start|end)>:\s*(\d+)", comment)]
    spo2 = float(np.mean(vals)) if vals else None

    return PpgRecord(name, red, ir, fs, time, spo2)


def load_physionet(limit: Optional[int] = None) -> List[PpgRecord]:
    records_file = os.path.join(PHYSIONET_DIR, "RECORDS")
    if os.path.exists(records_file):
        with open(records_file) as fh:
            names = [ln.strip() for ln in fh if ln.strip()]
    else:
        names = sorted(os.path.basename(p)[:-4]
                       for p in glob.glob(os.path.join(PHYSIONET_DIR, "*.hea")))
    if limit:
        names = names[:limit]
    return [load_physionet_record(os.path.join(PHYSIONET_DIR, n)) for n in names]
