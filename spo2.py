"""
SpO2 estimation from dual-wavelength PPG via a windowed double-FFT algorithm
with harmonic-based feature extraction.

Implements the method of:
    R. Jin et al., "Windowed Double-FFT Algorithm for SpO2 Measurement in Areas
    of Low Vascular Density," IEEE Sensors Journal, 25(1), 2025.

Pipeline (per RED and IR PPG channel)
-------------------------------------
    moving-average filter (3-pt then 8-pt)
      -> DC component  = 0-frequency bin of the (un-windowed) FFT  ≈ mean
      -> remove DC, remove baseline drift (piecewise linear fit)
      -> Hann window -> FFT                                 (the "double FFT")
      -> AC component = amplitude at the cardiac fundamental, located by
         harmonic-based feature extraction (a peak is the heart rate only if
         its integer-multiple harmonics are also present, which rejects noise)

Then the ratio-of-ratios and the empirical calibration:
    R = (AC_red / DC_red) / (AC_ir / DC_ir)
    SpO2 = A * R**2 + B * R + C

Three method variants mirror the paper's Test1/2/3:
    "fft_peak"      Test1: plain FFT + peak pick           (baseline)
    "wfft_peak"     Test2: windowed double-FFT + peak pick
    "wfft_harmonic" Test3: windowed double-FFT + harmonic pick   (recommended)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "moving_average_filter",
    "piecewise_linear_detrend",
    "spectrum",
    "ac_dc_components",
    "ratio_of_ratios",
    "spo2_from_ppg",
    "fit_calibration",
    "pipeline_stages",
    "PipelineStages",
    "SpO2Result",
    "DEFAULT_CALIBRATION",
    "HR_BAND",
]

# Heart-rate search band (Hz). 0.7-3.5 Hz ~= 42-210 bpm.
HR_BAND = (0.7, 3.5)

# Default empirical calibration SpO2 = A*R^2 + B*R + C. These are the textbook
# reflectance coefficients; for a specific device use :func:`fit_calibration`.
DEFAULT_CALIBRATION = (-31.324658381464815, 21.86129953204696, 94.45240103301107)


# =============================================================================
# Preprocessing
# =============================================================================
def moving_average_filter(x, sizes=(3, 8)):
    """Two-step moving-average smoother (default 3-point then 8-point)."""
    x = np.asarray(x, dtype=float)
    for n in sizes:
        if n > 1:
            x = np.convolve(x, np.ones(n) / n, mode="same")
    return x


def piecewise_linear_detrend(x, seg_len, alpha=4.0):
    """Remove baseline drift by subtracting a per-segment least-squares line.

    The signal is split into ``seg_len``-sample segments; each gets its own
    linear fit subtracted. ``alpha`` (>=2) softens the discontinuity at segment
    joins by re-aligning each segment to the running output by a fraction
    ``1/alpha`` of the jump (the correction factor from the paper).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if seg_len < 2 or seg_len >= n:
        # single linear detrend over the whole signal
        t = np.arange(n)
        a, b = np.polyfit(t, x, 1)
        return x - (a * t + b)

    out = np.empty(n)
    prev_end = None
    for start in range(0, n, seg_len):
        end = min(start + seg_len, n)
        t = np.arange(end - start)
        seg = x[start:end]
        if t.size >= 2:
            a, b = np.polyfit(t, seg, 1)
            detr = seg - (a * t + b)
        else:
            detr = seg - seg.mean()
        if prev_end is not None and alpha >= 2:
            detr = detr + (prev_end - detr[0]) / alpha   # soften the join
        out[start:end] = detr
        prev_end = detr[-1]
    return out


# =============================================================================
# AC / DC extraction (the core of the double-FFT method)
# =============================================================================
def _local_maxima(mag):
    """Indices of interior local maxima of ``mag``."""
    if mag.size < 3:
        return np.array([], dtype=int)
    interior = np.where((mag[1:-1] > mag[:-2]) & (mag[1:-1] >= mag[2:]))[0] + 1
    return interior


def _harmonic_fundamental(mag, freqs, band, n_harmonics=3, m_required=1, tol=0.18):
    """Pick the cardiac fundamental: the strongest in-band peak whose integer
    harmonics are also present. Falls back to the strongest in-band peak."""
    in_band = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
    if in_band.size == 0:
        return int(np.argmax(mag))

    maxima = set(_local_maxima(mag).tolist())
    cands = [i for i in in_band if i in maxima] or list(in_band)
    cands.sort(key=lambda i: mag[i], reverse=True)

    fmax = freqs[-1]
    for c in cands:
        f0 = freqs[c]
        if f0 <= 0:
            continue
        harmonics = 0
        for h in range(2, n_harmonics + 1):
            target = h * f0
            if target > fmax:
                break
            near = np.where(np.abs(freqs - target) <= tol * f0)[0]
            # a harmonic counts if there is a clear local bump near h*f0
            if near.size and mag[near].max() > 1.2 * np.median(mag):
                harmonics += 1
        if harmonics >= m_required:
            return c
    return cands[0]


def _preprocess(channel, fs, method):
    """Run the per-channel preprocessing and return every intermediate stage.

    This is the single source of truth shared by the spectrum/AC-DC estimator
    and the step-by-step visualiser. Returns
    ``(filtered, dc, detrended, window_fn, windowed)``:

      * ``filtered``  - two-step moving-average smooth (3-pt then 8-pt).
      * ``dc``        - DC component = mean of ``filtered`` (0-frequency bin).
      * ``detrended`` - ``filtered`` with DC removed and, for the windowed
                        methods, the piecewise-linear baseline drift removed.
      * ``window_fn`` - the Hann window (all-ones for ``fft_peak``).
      * ``windowed``  - ``detrended * window_fn``, the input to the second FFT.
    """
    filtered = moving_average_filter(np.asarray(channel, dtype=float))
    n = filtered.size
    dc = float(np.mean(filtered))               # 0-frequency component
    detrended = filtered - dc
    if method in ("wfft_peak", "wfft_harmonic"):
        seg = max(2, int(round(fs)))            # ~1 s baseline segments
        detrended = piecewise_linear_detrend(detrended, seg_len=seg)
        window_fn = np.hanning(n)
    else:
        window_fn = np.ones(n)
    windowed = detrended * window_fn
    return filtered, dc, detrended, window_fn, windowed


def _channel_spectrum(channel, fs, method):
    """Preprocess one channel per ``method`` and return ``(freqs, mag, dc, n)``.

    The "second FFT" arm: subtract DC, remove baseline drift, apply a Hann
    window, then FFT.
    """
    _, dc, _, _, windowed = _preprocess(channel, fs, method)
    n = windowed.size
    mag = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, mag, dc, n


def _pick_fundamental(mag, freqs, method, band):
    """Bin index of the cardiac fundamental for the given method."""
    if method == "wfft_harmonic":
        return _harmonic_fundamental(mag, freqs, band)
    in_band = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
    return int(in_band[np.argmax(mag[in_band])]) if in_band.size else int(np.argmax(mag))


def spectrum(channel, fs, *, method="wfft_harmonic", band=HR_BAND):
    """Analysis spectrum of one PPG channel, for visualization.

    Returns ``(freqs_hz, magnitude, f0_bin, dc)`` — the same spectrum and
    cardiac-fundamental bin the estimator uses internally.
    """
    freqs, mag, dc, _ = _channel_spectrum(channel, fs, method)
    f0_bin = _pick_fundamental(mag, freqs, method, band)
    return freqs, mag, int(f0_bin), abs(dc)


@dataclass
class PipelineStages:
    """Every intermediate stage of one channel's double-FFT pipeline.

    Captured for the step-by-step visualiser so the GUI plots exactly what the
    estimator computes (no re-derivation).
    """
    fs: float
    method: str
    raw: np.ndarray             # the input segment
    filtered: np.ndarray        # after the two-step moving-average filter
    dc: float                   # DC component (mean of ``filtered`` = 0-Hz bin)
    detrended: np.ndarray       # DC + baseline removed (input to the window)
    window_fn: np.ndarray       # Hann window (all-ones for ``fft_peak``)
    windowed: np.ndarray        # ``detrended * window_fn`` (input to 2nd FFT)
    freqs: np.ndarray           # rFFT frequencies (Hz) shared by both spectra
    mag_dc: np.ndarray          # |FFT| of ``filtered`` — first FFT; DC at bin 0
    mag_ac: np.ndarray          # |FFT| of ``windowed`` — second FFT; AC arm
    f0_bin: int                 # cardiac-fundamental bin in the second FFT
    ac: float                   # AC amplitude read at ``f0_bin``


def pipeline_stages(
    channel,
    fs,
    *,
    method: str = "wfft_harmonic",
    band: Tuple[float, float] = HR_BAND,
    f0_bin: Optional[int] = None,
) -> PipelineStages:
    """Run the full per-channel pipeline and return all intermediate stages.

    The "double FFT": ``mag_dc`` is the un-windowed FFT of the filtered signal
    (its 0-Hz bin is the DC), and ``mag_ac`` is the FFT of the detrended,
    Hann-windowed signal (its peak at the cardiac fundamental is the AC).

    If ``f0_bin`` is given, the AC is read at that exact bin instead of the one
    picked here (used to lock the RED channel to the IR channel's fundamental).
    """
    raw = np.asarray(channel, dtype=float)
    filtered, dc, detrended, window_fn, windowed = _preprocess(raw, fs, method)
    n = filtered.size
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag_dc = np.abs(np.fft.rfft(filtered))      # first FFT: DC lives at bin 0
    mag_ac = np.abs(np.fft.rfft(windowed))      # second FFT: AC at fundamental
    if f0_bin is None:
        f0_bin = _pick_fundamental(mag_ac, freqs, method, band)
    ac = float(mag_ac[f0_bin]) * 2.0 / n
    return PipelineStages(
        fs=float(fs), method=method, raw=raw, filtered=filtered, dc=dc,
        detrended=detrended, window_fn=window_fn, windowed=windowed,
        freqs=freqs, mag_dc=mag_dc, mag_ac=mag_ac, f0_bin=int(f0_bin), ac=ac,
    )


def ac_dc_components(
    channel,
    fs,
    *,
    method: str = "wfft_harmonic",
    band: Tuple[float, float] = HR_BAND,
    f0_bin: Optional[int] = None,
):
    """Return ``(ac, dc, f0_bin)`` for one PPG channel.

    ``method``:
      * ``"fft_peak"``      - plain FFT, AC = tallest in-band peak (no window).
      * ``"wfft_peak"``     - DC + baseline removal, Hann window, FFT,
                              AC = tallest in-band peak.
      * ``"wfft_harmonic"`` - as above but the fundamental is chosen by the
                              harmonic rule.
    If ``f0_bin`` is given, the AC amplitude is read at that exact rFFT bin
    (used to lock the RED channel to the IR channel's fundamental).
    """
    freqs, mag, dc, n = _channel_spectrum(channel, fs, method)
    if f0_bin is None:
        f0_bin = _pick_fundamental(mag, freqs, method, band)
    ac = float(mag[f0_bin]) * 2.0 / n
    return ac, abs(dc), int(f0_bin)


def ratio_of_ratios(red, ir, fs, *, method="wfft_harmonic", band=HR_BAND):
    """Compute R = (AC/DC)_red / (AC/DC)_ir for one window.

    The cardiac fundamental is located on the IR channel (higher SNR) and the
    RED amplitude is read at the same frequency bin, so both ratios refer to the
    same heartbeat. Returns ``(R, freq_hz, valid)`` where ``valid`` is whether
    RED's own fundamental matches IR's.
    """
    red = np.asarray(red, dtype=float)
    ir = np.asarray(ir, dtype=float)

    ac_ir, dc_ir, k_ir = ac_dc_components(ir, fs, method=method, band=band)
    ac_red, dc_red, _ = ac_dc_components(red, fs, method=method, band=band,
                                         f0_bin=k_ir)
    _, _, k_red = ac_dc_components(red, fs, method=method, band=band)

    freqs = np.fft.rfftfreq(ir.size, d=1.0 / fs)
    f0 = float(freqs[k_ir])
    valid = abs(freqs[k_red] - f0) <= max(2.0 * (freqs[1] - freqs[0]), 0.1)

    if dc_red <= 0 or dc_ir <= 0 or ac_ir <= 0:
        return float("nan"), f0, False
    R = (ac_red / dc_red) / (ac_ir / dc_ir)
    return float(R), f0, bool(valid)


# =============================================================================
# Full SpO2 estimate
# =============================================================================
@dataclass
class SpO2Result:
    spo2: float                 # estimated SpO2 (%)
    R: float                    # median ratio-of-ratios
    fs: float
    method: str
    n_windows: int              # windows that produced a valid R
    valid_fraction: float       # fraction of windows whose RED/IR freqs matched
    window_R: np.ndarray        # per-window R values


def spo2_from_ppg(
    red,
    ir,
    fs,
    *,
    method: str = "wfft_harmonic",
    calibration=DEFAULT_CALIBRATION,
    window: float = 60.0,
    hop: Optional[float] = None,
    band: Tuple[float, float] = HR_BAND,
) -> SpO2Result:
    """Estimate SpO2 from RED and IR PPG.

    The signal is split into ``window``-second segments (hop defaults to
    ``window/2``); R is computed per window and the median R is mapped to SpO2
    by the quadratic ``calibration`` (A, B, C). Using the median makes the
    estimate robust to occasional bad windows.
    """
    red = np.asarray(red, dtype=float)
    ir = np.asarray(ir, dtype=float)
    n = min(red.size, ir.size)
    red, ir = red[:n], ir[:n]

    win = max(8, int(round(window * fs)))
    if win >= n:
        starts = [0]
        win = n
    else:
        step = int(round((hop if hop else window / 2) * fs))
        step = max(1, step)
        starts = list(range(0, n - win + 1, step))

    Rs, n_valid = [], 0
    for s in starts:
        R, _, valid = ratio_of_ratios(red[s:s + win], ir[s:s + win], fs,
                                      method=method, band=band)
        if np.isfinite(R):
            Rs.append(R)
            n_valid += int(valid)
    Rs = np.asarray(Rs, dtype=float)

    if Rs.size == 0:
        return SpO2Result(float("nan"), float("nan"), float(fs), method,
                          0, 0.0, Rs)

    R_med = float(np.median(Rs))
    A, B, C = calibration
    spo2 = A * R_med ** 2 + B * R_med + C
    return SpO2Result(
        spo2=float(np.clip(spo2, 0, 100)),
        R=R_med, fs=float(fs), method=method,
        n_windows=int(Rs.size),
        valid_fraction=float(n_valid / Rs.size),
        window_R=Rs,
    )


def fit_calibration(R_values, spo2_true, degree=2):
    """Least-squares fit of SpO2 = poly(R). Returns (A, B, C) for degree 2
    (or padded with leading zeros for lower degrees)."""
    R_values = np.asarray(R_values, dtype=float)
    spo2_true = np.asarray(spo2_true, dtype=float)
    ok = np.isfinite(R_values) & np.isfinite(spo2_true)
    coeffs = np.polyfit(R_values[ok], spo2_true[ok], degree)
    if coeffs.size < 3:
        coeffs = np.concatenate([np.zeros(3 - coeffs.size), coeffs])
    return tuple(float(c) for c in coeffs)
