"""
Evaluate the windowed double-FFT SpO2 algorithm on both datasets:

  * dataset_primer_1                 (MAX30102 PPG, PPG_Red / PPG_IR, 125 Hz)
  * PhysioNet Pulse-Transit-Time PPG (MAX30101 PPG, pleth_1=RED/pleth_2=IR, 500 Hz)

For each dataset it reports:
  1. Validity rate (RED/IR cardiac frequencies agree) for the paper's Test1/2/3.
  2. SpO2 accuracy of the recommended method: a device-specific calibration
     SpO2 = A*R^2 + B*R + C is fitted, then MAE is reported in-sample, by
     leave-one-out cross-validation, and against the trivial predict-the-mean
     baseline (calibration must be per-device because the raw R scale differs).

Run:
    python evaluate.py                 # both datasets
    python evaluate.py --dataset physionet --limit 9
"""
from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd

import datasets as ds
from spo2 import spo2_from_ppg, fit_calibration

warnings.filterwarnings("ignore")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
METHODS = ["fft_peak", "wfft_peak", "wfft_harmonic"]
METHOD_LABEL = {"fft_peak": "Test1 FFT+peak",
                "wfft_peak": "Test2 windowed double-FFT+peak",
                "wfft_harmonic": "Test3 windowed double-FFT+harmonic"}


def _loo_mae(R, S, degree=2):
    """Leave-one-out MAE of the polynomial calibration R -> SpO2."""
    R, S = np.asarray(R), np.asarray(S)
    if R.size < 4:
        return float("nan")
    errs = []
    for i in range(R.size):
        mask = np.arange(R.size) != i
        coeffs = np.polyfit(R[mask], S[mask], degree)
        errs.append(abs(np.polyval(coeffs, R[i]) - S[i]))
    return float(np.mean(errs))


def _collect(records, method, window):
    rows = []
    for rec in records:
        if rec.spo2_ref is None:
            continue
        res = spo2_from_ppg(rec.red, rec.ir, rec.fs, method=method, window=window)
        if np.isfinite(res.R):
            rows.append(dict(record=rec.name, R=res.R,
                             valid_fraction=res.valid_fraction,
                             spo2_ref=rec.spo2_ref))
    return pd.DataFrame(rows)


def evaluate_dataset(title, records, window):
    print(f"\n{'='*72}\n{title}  ({len(records)} records)\n{'='*72}")

    print("Validity rate (RED/IR cardiac frequencies agree) - paper Test1/2/3")
    print("-" * 72)
    df_harm = None
    for method in METHODS:
        df = _collect(records, method, window)
        if method == "wfft_harmonic":
            df_harm = df
        if len(df):
            print(f"  {METHOD_LABEL[method]:42s}: {100*df['valid_fraction'].mean():5.1f}%"
                  f"  ({len(df)} records)")

    df = df_harm
    R, S = df["R"].to_numpy(), df["spo2_ref"].to_numpy()
    A, B, C = fit_calibration(R, S, degree=2)
    df["spo2_est"] = np.clip(A * R**2 + B * R + C, 0, 100)
    insample = float(np.mean(np.abs(df["spo2_est"] - S)))
    loo = _loo_mae(R, S)
    baseline = float(np.mean(np.abs(S - S.mean())))
    corr = float(np.corrcoef(R, S)[0, 1]) if R.size > 1 else float("nan")

    print("\nSpO2 accuracy (method = wfft_harmonic)")
    print("-" * 72)
    print(f"  SpO2 ground-truth   : {S.min():.1f}-{S.max():.1f}%  "
          f"(mean {S.mean():.1f}, std {S.std():.2f})")
    print(f"  fitted calibration  : SpO2 = {A:.3f}*R^2 + {B:.3f}*R + {C:.3f}")
    print(f"  corr(R, SpO2)       : {corr:+.3f}")
    print(f"  MAE in-sample fit   : {insample:.2f} %")
    print(f"  MAE leave-one-out   : {loo:.2f} %")
    print(f"  MAE predict-the-mean: {baseline:.2f} %  (trivial baseline)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = title.split()[0].lower()
    df.to_csv(os.path.join(RESULTS_DIR, f"{tag}_spo2.csv"), index=False)
    return dict(dataset=title, n=int(R.size), corr=corr,
                mae_fit=insample, mae_loo=loo, mae_baseline=baseline)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["primer", "physionet", "both"],
                    default="both")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--window", type=float, default=60.0,
                    help="analysis window in seconds")
    args = ap.parse_args()

    summary = []
    if args.dataset in ("primer", "both"):
        recs = ds.load_primer(limit=args.limit)
        summary.append(evaluate_dataset("primer (MAX30102, 125 Hz)", recs, args.window))
    if args.dataset in ("physionet", "both"):
        recs = ds.load_physionet(limit=args.limit)
        summary.append(evaluate_dataset(
            "PhysioNet PTT-PPG (MAX30101, 500 Hz)", recs, args.window))

    if len(summary) > 1:
        print(f"\n{'='*72}\nSpO2 MAE SUMMARY\n{'='*72}")
        print(f"  {'dataset':40s} {'n':>4} {'corr':>7} {'MAE(LOO)':>9} {'baseline':>9}")
        for s in summary:
            print(f"  {s['dataset']:40s} {s['n']:>4} {s['corr']:>+7.3f} "
                  f"{s['mae_loo']:>8.2f}% {s['mae_baseline']:>8.2f}%")


if __name__ == "__main__":
    main()
