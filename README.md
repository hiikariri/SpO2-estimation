# SpO2-estimation
# SpO2 Library (Windowed Double-FFT)

Estimate blood-oxygen saturation (SpO2) from a **dual-wavelength PPG** (a RED
and an IR photoplethysmogram), following:

> R. Jin et al., "Windowed Double-FFT Algorithm for SpO2 Measurement in Areas of
> Low Vascular Density," *IEEE Sensors Journal*, 25(1), 2025.

## Idea

SpO2 comes from the **ratio-of-ratios** of the two PPG colours:

```
R = (AC_red / DC_red) / (AC_ir / DC_ir)
SpO2 = A·R² + B·R + C        (empirical calibration)
```

The hard part is measuring AC (the pulsatile amplitude) and DC (the mean light
level) cleanly. This algorithm does it in the frequency domain:

```
moving-average filter (3-pt, 8-pt)
  → DC = 0-frequency bin of the plain FFT (≈ mean)
  → remove DC, remove baseline drift (piecewise linear fit)
  → Hann window → FFT                      ← the second "double" FFT
  → AC = amplitude at the cardiac fundamental, found by the harmonic rule
```

The **window** (Hann) cuts spectral leakage; the **double FFT** keeps that
windowing from corrupting the DC term; the **harmonic rule** picks the true
heart-rate peak (only a peak whose 2×/3× harmonics also exist), which rejects
noise so the RED and IR frequencies agree.

## How to use

```python
from spo2 import spo2_from_ppg

res = spo2_from_ppg(red, ir, fs)     # red, ir = PPG samples; fs in Hz
res.spo2            # estimated SpO2 (%)
res.R               # ratio-of-ratios
res.valid_fraction  # fraction of windows where RED & IR frequencies agreed
```

`method="wfft_harmonic"` (default, paper's Test3) is recommended; `"wfft_peak"`
(Test2) and `"fft_peak"` (Test1) are the simpler baselines. Pass your own
`calibration=(A, B, C)` — fit it for your device with `fit_calibration(R, spo2)`.

## Files

| File | What it is |
|------|------------|
| `spo2.py` | The library |
| `datasets.py` | Loads dataset_primer_1 and PhysioNet PTT-PPG (RED+IR) + SpO2 |
| `evaluate.py` | Validity + accuracy evaluation on both datasets |
| `spo2_gui.py` | GUI: visualises the pipeline (time, double-FFT spectrum, R→SpO2) |

## GUI

```bash
python spo2_gui.py
```

Pick a dataset folder and record to see the algorithm step by step:

1. **Time domain** — the RED and IR PPG of the analysed segment.
2. **Frequency domain** — the windowed double-FFT spectrum of each channel, with
   the cardiac fundamental (the **AC location**) marked, the IR harmonics shown
   (the harmonic rule), and whether the RED/IR fundamentals agree (validity).
3. **Per-window R** across the whole record, the median R, and the SpO2 it maps
   to vs the ground truth.

"Evaluate all" runs the whole folder, fits the SpO2 calibration, reports MAE,
and applies that calibration to subsequent single-record estimates. The
"Segment start"/"Window" controls move the analysed slice (time + FFT panels)
live. Works with both the primer (`*_ppg.csv`) and PhysioNet (`*.hea`) layouts.

## Testing

```bash
python evaluate.py                      # both datasets
python evaluate.py --dataset physionet  # one dataset
```

Two datasets carry the dual-wavelength PPG this algorithm needs:

* **dataset_primer_1** — MAX30102, `PPG_Red`/`PPG_IR`, 125 Hz, 122 records.
* **PhysioNet Pulse-Transit-Time PPG** — MAX30101, `pleth_1`=RED / `pleth_2`=IR,
  500 Hz, 66 records (22 subjects × sit/walk/run). Needs the `wfdb` package.

### Validity (the paper's primary metric)

Fraction of windows where the RED and IR cardiac frequencies agree (a mismatch
makes the reading invalid). The Test1→Test2→Test3 trend reproduces the paper —
windowing gives the big jump:

| Method | primer | PhysioNet |
|--------|-------:|----------:|
| Test1 — FFT + peak | 86.6% | 86.8% |
| Test2 — windowed double-FFT + peak | 94.7% | 94.0% |
| **Test3 — windowed double-FFT + harmonic** | **94.9%** | **94.0%** |

(Absolute validity is higher than the paper's because both are strong fingertip
PPGs, not the low-vascular-density forearm/sacral signals the paper targets;
PhysioNet's walk/run records do show validity dropping on heavy motion.)

### SpO2 accuracy (MAE)

A device-specific calibration `SpO2 = A·R² + B·R + C` is fitted per dataset
(the raw R scale differs by sensor). MAE is reported by leave-one-out CV against
the trivial "predict the mean" baseline:

| Dataset | n | corr(R, SpO2) | MAE (LOO) | baseline |
|---------|--:|--------------:|----------:|---------:|
| primer (MAX30102) | 122 | −0.03 | 0.86% | 0.85% |
| PhysioNet (MAX30101) | 66 | −0.24 | 0.69% | 0.66% |

Both MAEs are sub-1%, **but that is mostly because SpO2 barely varies** in these
healthy subjects (primer 95–100%, PhysioNet 94.5–98.5%). Neither fitted
calibration beats predicting the mean in cross-validation, so the MAE reflects
the narrow range, not validated accuracy. The encouraging sign is that on
PhysioNet R shows the **physiologically correct negative correlation** with SpO2
(−0.24, vs −0.03 on primer) — evidence the ratio-of-ratios captures real
information; it just needs subjects with genuine desaturation to be validated.

## Why BIDMC is not tested

The ratio-of-ratios needs **two** wavelengths. The BIDMC dataset has a single
`PLETH` channel (one wavelength), so R cannot be formed. BIDMC stores the
monitor's SpO2 *number*, but not the raw RED/IR signals this algorithm needs.
