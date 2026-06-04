#!/usr/bin/env python3
"""
SpO2 estimation from a saved dual-wavelength PPG file, using the windowed
double-FFT algorithm in spo2.py with the calibration fitted on
dataset_primer_1 (the same MAX30102 rig).

    SpO2 = A*R^2 + B*R + C        (A, B, C below; fitted on 122 primer records)

This is a *library*: import it and call the functions. You handle acquisition
(your ESP32 framework writes the file); this turns that file into an SpO2.

Input file format
-----------------
The file your framework writes via::

    np.savetxt(path, np.column_stack((red, ir)), fmt="%.6f",
               header=f"PPG Data - Samples: {N}, Duration: {D}s, Rate: {FS}Hz\\nRED\\tIR",
               comments="# ")

i.e. two whitespace-separated columns (RED, IR) under a ``#`` header whose first
line carries ``Rate: <fs>Hz``. The sample rate is read from that header (pass
``fs=`` to override). The stream is resampled to 125 Hz — the rate the
calibration was fitted on — so R stays on the same scale.

Library use
-----------
    from pi_spo2 import estimate_from_file, estimate_spo2, load_ppg

    r = estimate_from_file("ppg_signals.txt")
    if r.valid:
        print(r.spo2, r.hr_bpm, r.R)

    # or if you already hold the arrays:
    r = estimate_spo2(red, ir, fs_native=200.0)

CLI (optional convenience)
--------------------------
    python3 pi_spo2.py ppg_signals.txt
    python3 pi_spo2.py --selftest
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from spo2 import spo2_from_ppg, ratio_of_ratios

# ============================================================================
# Calibration — fitted on dataset_primer_1 (MAX30102, 125 Hz, n=122 records).
# Re-fit any time with:  python evaluate.py --dataset primer
# and paste the printed "fitted calibration" coefficients here.
# ============================================================================
CALIB = (-31.324658381464815, 21.86129953204696, 94.45240103301107)  # A, B, C

# Range of R seen in the primer fit. Outside this the quadratic extrapolates
# and the reading is flagged unreliable.
R_MIN, R_MAX = 0.176, 0.534

# Analysis settings — MUST match how the calibration was produced.
FS_HZ = 125.0               # calibration rate; the input is resampled to this
WINDOW_S = 60.0             # one-minute analysis window
METHOD = "wfft_harmonic"    # paper's Test3 (recommended)
MIN_VALIDITY = 0.5          # reject a reading if fewer than this fraction of
#                             windows had agreeing RED/IR cardiac peaks


@dataclass
class Reading:
    """Result of one SpO2 estimate."""
    spo2: float                         # estimated SpO2 (%)
    R: float                            # median ratio-of-ratios
    hr_bpm: float                       # heart rate (beats/min)
    valid_fraction: float               # fraction of windows RED/IR agreed
    valid: bool                         # overall: safe to trust this reading?
    fs_native: float                    # input sample rate before resampling
    n_samples: int                      # input sample count
    flags: List[str] = field(default_factory=list)   # why a reading is suspect


# ============================================================================
# Loading
# ============================================================================
def _read_rate_from_header(path) -> Optional[float]:
    """Parse ``Rate: <fs>Hz`` from the ``#`` header, if present."""
    with open(path, "r") as fh:
        for line in fh:
            if not line.lstrip().startswith("#"):
                break                       # header is over
            m = re.search(r"Rate:\s*([\d.]+)\s*Hz", line)
            if m:
                return float(m.group(1))
    return None


def load_ppg(path, fs: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """Load a two-column (RED, IR) PPG file. Returns ``(red, ir, fs_native)``.

    ``fs`` overrides the rate; otherwise it is read from the header. Raises if
    neither is available.
    """
    if fs is None:
        fs = _read_rate_from_header(path)
    if fs is None:
        raise ValueError(
            f"No 'Rate: <fs>Hz' in the header of {path}; pass fs= explicitly.")
    data = np.loadtxt(path)                  # '#' header lines skipped by default
    data = np.atleast_2d(data)
    if data.shape[1] < 2:
        raise ValueError(f"Expected 2 columns (RED, IR) in {path}, "
                         f"got shape {data.shape}.")
    red = np.asarray(data[:, 0], float)
    ir = np.asarray(data[:, 1], float)
    return red, ir, float(fs)


# ============================================================================
# Pipeline
# ============================================================================
def resample_to(x, fs_in, fs_out=FS_HZ):
    """Linear-resample a uniformly-sampled signal from ``fs_in`` to ``fs_out``.

    This keeps R comparable to the 125 Hz dataset whatever the ESP32 rate is.
    """
    x = np.asarray(x, float)
    if abs(fs_in - fs_out) < 1e-9 or x.size < 2:
        return x
    dur = x.size / fs_in
    n_out = int(round(dur * fs_out))
    if n_out < 2:
        return x
    t_in = np.arange(x.size) / fs_in
    t_out = np.arange(n_out) / fs_out
    return np.interp(t_out, t_in, x)


def estimate_spo2(red, ir, fs_native) -> Reading:
    """Estimate SpO2 from RED/IR arrays sampled at ``fs_native`` Hz.

    Resamples to 125 Hz, runs the windowed double-FFT estimator, and applies
    the primer calibration. Returns a :class:`Reading` (check ``.valid``).
    """
    red = np.asarray(red, float)
    ir = np.asarray(ir, float)
    n_in = min(red.size, ir.size)
    red, ir = red[:n_in], ir[:n_in]

    r = resample_to(red, fs_native, FS_HZ)
    i = resample_to(ir, fs_native, FS_HZ)

    res = spo2_from_ppg(r, i, FS_HZ, method=METHOD, window=WINDOW_S)
    _, f0_hz, _ = ratio_of_ratios(r, i, FS_HZ, method=METHOD)   # for heart rate

    A, B, C = CALIB
    spo2 = float(np.clip(A * res.R ** 2 + B * res.R + C, 0, 100))
    hr = float(f0_hz * 60.0)

    flags: List[str] = []
    if not np.isfinite(res.R):
        flags.append("no valid pulse (check finger contact / signal)")
    else:
        if res.valid_fraction < MIN_VALIDITY:
            flags.append(f"low validity {res.valid_fraction:.0%}")
        if not (R_MIN <= res.R <= R_MAX):
            flags.append(f"R={res.R:.3f} outside calibration range "
                         f"[{R_MIN}, {R_MAX}] (extrapolating)")
    valid = np.isfinite(res.R) and res.valid_fraction >= MIN_VALIDITY

    return Reading(spo2=spo2, R=float(res.R), hr_bpm=hr,
                   valid_fraction=float(res.valid_fraction), valid=bool(valid),
                   fs_native=float(fs_native), n_samples=int(n_in), flags=flags)


def estimate_from_file(path="ppg_signals.txt", fs: Optional[float] = None) -> Reading:
    """Load a PPG file and return its :class:`Reading`."""
    red, ir, fs_native = load_ppg(path, fs=fs)
    return estimate_spo2(red, ir, fs_native)


def format_reading(r: Reading) -> str:
    """One-line human-readable summary of a :class:`Reading`."""
    status = "OK" if r.valid else "UNRELIABLE"
    extra = f"  [{'; '.join(r.flags)}]" if r.flags else ""
    if not np.isfinite(r.R):
        return f"reading rejected: {'; '.join(r.flags) or 'no pulse'}"
    return (f"SpO2 = {r.spo2:4.1f} %   HR = {r.hr_bpm:5.1f} bpm   R = {r.R:.3f}   "
            f"validity = {r.valid_fraction:.0%}   ({status}){extra}")


# ============================================================================
# Self-test: validate the pipeline with no file / no sensor.
# ============================================================================
def selftest():
    """Feed a synthetic two-tone PPG through the pipeline."""
    fs = 200.0                               # pretend the ESP32 ran at 200 Hz
    n = int(WINDOW_S * fs)
    t = np.arange(n) / fs
    hr = 1.2                                 # 72 bpm
    ir = 10000 + 200 * np.sin(2*np.pi*hr*t) + 30 * np.sin(2*np.pi*2*hr*t)
    red = 8000 + 70 * np.sin(2*np.pi*hr*t) + 10 * np.sin(2*np.pi*2*hr*t)
    r = estimate_spo2(red, ir, fs)
    print("Self-test (synthetic PPG @ 200 Hz -> resampled to 125 Hz):")
    print("  " + format_reading(r))
    print("  (synthetic R may fall outside the primer range — expected here.)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default="ppg_signals.txt",
                    help="PPG file to estimate (default: ppg_signals.txt)")
    ap.add_argument("--fs", type=float, default=None,
                    help="override sample rate (Hz); default reads it from header")
    ap.add_argument("--selftest", action="store_true",
                    help="run on a synthetic signal (no file)")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    r = estimate_from_file(args.path, fs=args.fs)
    print(f"{args.path}: {r.n_samples} samples @ {r.fs_native:.0f} Hz")
    print("  " + format_reading(r))


if __name__ == "__main__":
    main()
