"""
GUI for the windowed double-FFT SpO2 algorithm.

Load a dual-wavelength PPG record and watch the algorithm work, step by step:
  1. Time domain  - the RED and IR PPG of the analysed segment.
  2. Frequency domain - the windowed double-FFT spectrum of each channel, with
     the cardiac fundamental (the AC location), its harmonics, and whether the
     RED and IR fundamentals agree (the paper's "validity").
  3. Per-window R across the whole record, the median R, and the SpO2 it maps to
     vs the ground-truth SpO2.

Supports both dual-wavelength layouts:
  * dataset_primer_1                 -> ``*_ppg.csv``  (PPG_Red / PPG_IR)
  * PhysioNet Pulse-Transit-Time PPG -> ``*.hea``      (pleth_1=RED / pleth_2=IR)

Run:
    python spo2_gui.py
"""
import re
import sys
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import datasets as ds
from spo2 import (spo2_from_ppg, ratio_of_ratios, spectrum, fit_calibration,
                  DEFAULT_CALIBRATION, HR_BAND)

DEFAULT_DATASET = ds.PHYSIONET_DIR if Path(ds.PHYSIONET_DIR).exists() else ds.PRIMER_DIR


# --------------------------------------------------------------------------- #
# Data helpers (bridge the two dataset layouts to one interface)
# --------------------------------------------------------------------------- #
def discover_records(folder):
    folder = Path(folder)
    recs = []
    for f in sorted(folder.glob("*_ppg.csv")):
        recs.append(("primer", str(f), f.name[: -len("_ppg.csv")]))
    for f in sorted(folder.glob("*.hea")):
        recs.append(("physionet", str(f)[:-4], f.stem))   # base path, no ext
    return recs


def load_record(kind, path):
    if kind == "primer":
        return ds.load_primer_record(path)
    return ds.load_physionet_record(path)


def short_label(name):
    s = re.sub(r"_\d{8}_\d{6}$", "", name)
    return s[len("data_"):] if s.startswith("data_") else s


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #
class SpO2Gui(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Windowed Double-FFT SpO2")
        self.resize(1300, 950)
        self.dataset_dir = Path(DEFAULT_DATASET)
        self.records = []
        self.calibration = DEFAULT_CALIBRATION
        self._cal_note = "default"

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        controls = QtWidgets.QGridLayout()
        root.addLayout(controls)

        # Dataset folder
        controls.addWidget(QtWidgets.QLabel("Dataset folder"), 0, 0)
        self.dataset_edit = QtWidgets.QLineEdit(str(self.dataset_dir))
        controls.addWidget(self.dataset_edit, 0, 1, 1, 5)
        browse = QtWidgets.QPushButton("Browse")
        browse.clicked.connect(self.browse_dataset)
        controls.addWidget(browse, 0, 6)

        # Record + navigation
        controls.addWidget(QtWidgets.QLabel("Record"), 1, 0)
        self.record_combo = QtWidgets.QComboBox()
        controls.addWidget(self.record_combo, 1, 1)
        self.prev_btn = QtWidgets.QPushButton("◀ Prev")
        self.prev_btn.clicked.connect(lambda: self.step_record(-1))
        controls.addWidget(self.prev_btn, 1, 2)
        self.next_btn = QtWidgets.QPushButton("Next ▶")
        self.next_btn.clicked.connect(lambda: self.step_record(1))
        controls.addWidget(self.next_btn, 1, 3)
        self.record_combo.currentIndexChanged.connect(self._update_nav_buttons)
        refresh = QtWidgets.QPushButton("Refresh")
        refresh.clicked.connect(self.refresh_records)
        controls.addWidget(refresh, 1, 4)

        # Method
        controls.addWidget(QtWidgets.QLabel("Method"), 2, 0)
        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.addItems(["wfft_harmonic", "wfft_peak", "fft_peak"])
        controls.addWidget(self.method_combo, 2, 1)

        # Segment (drives the time + FFT panels)
        controls.addWidget(QtWidgets.QLabel("Segment start (s)"), 2, 2)
        self.start_spin = QtWidgets.QDoubleSpinBox()
        self.start_spin.setRange(0.0, 1_000_000.0)
        self.start_spin.setSingleStep(5.0)
        controls.addWidget(self.start_spin, 2, 3)
        controls.addWidget(QtWidgets.QLabel("Window (s)"), 2, 4)
        self.window_spin = QtWidgets.QDoubleSpinBox()
        self.window_spin.setRange(4.0, 120.0)
        self.window_spin.setSingleStep(1.0)
        self.window_spin.setValue(60.0)
        controls.addWidget(self.window_spin, 2, 5)
        self.start_spin.valueChanged.connect(self._replot_segment)
        self.window_spin.valueChanged.connect(self._replot_segment)

        # Buttons
        self.plot_btn = QtWidgets.QPushButton("Estimate && Plot")
        self.plot_btn.clicked.connect(self.estimate_and_plot)
        controls.addWidget(self.plot_btn, 1, 5, 1, 1)
        self.batch_btn = QtWidgets.QPushButton("Evaluate all (fit + MAE)")
        self.batch_btn.clicked.connect(self.evaluate_all)
        controls.addWidget(self.batch_btn, 1, 6, 1, 1)

        # Metrics
        self.metrics_label = QtWidgets.QLabel("Select a record and click Estimate && Plot.")
        self.metrics_label.setStyleSheet("font-weight: bold;")
        root.addWidget(self.metrics_label)

        # Figure
        self.figure = Figure(figsize=(12, 8.5), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        root.addWidget(self.toolbar)
        root.addWidget(self.canvas)

        # cached current record/result for cheap segment redraws
        self._cur = None        # (rec, res, name)
        self.refresh_records()

    # ----------------------------------------------------------------- #
    def browse_dataset(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select dataset folder", str(self.dataset_dir))
        if path:
            self.dataset_edit.setText(path)
            self.refresh_records()

    def refresh_records(self):
        self.dataset_dir = Path(self.dataset_edit.text().strip())
        self.record_combo.clear()
        if not self.dataset_dir.exists():
            self.metrics_label.setText(f"Folder not found: {self.dataset_dir}")
            self.records = []
            return
        self.records = discover_records(self.dataset_dir)
        self.record_combo.addItems([short_label(n) for _, _, n in self.records])
        # reset per-dataset calibration (raw R scale is device-specific)
        self.calibration, self._cal_note = DEFAULT_CALIBRATION, "default"
        if self.records:
            self.metrics_label.setText(
                f"{len(self.records)} records. Pick one and Estimate && Plot "
                f"(or Evaluate all to fit the SpO2 calibration).")
        else:
            self.metrics_label.setText(
                f"No '*_ppg.csv' or '*.hea' records in {self.dataset_dir}")
        self._update_nav_buttons()

    def step_record(self, delta):
        n = self.record_combo.count()
        if n == 0:
            return
        i = min(n - 1, max(0, self.record_combo.currentIndex() + delta))
        if i != self.record_combo.currentIndex():
            self.record_combo.setCurrentIndex(i)
            self.estimate_and_plot()
        self._update_nav_buttons()

    def _update_nav_buttons(self):
        n, i = self.record_combo.count(), self.record_combo.currentIndex()
        self.prev_btn.setEnabled(n > 0 and i > 0)
        self.next_btn.setEnabled(n > 0 and i < n - 1)

    # ----------------------------------------------------------------- #
    def estimate_and_plot(self):
        i = self.record_combo.currentIndex()
        if i < 0 or i >= len(self.records):
            QtWidgets.QMessageBox.warning(self, "No record", "No record selected.")
            return
        kind, path, name = self.records[i]
        method = self.method_combo.currentText()
        window = self.window_spin.value()
        try:
            rec = load_record(kind, path)
            res = spo2_from_ppg(rec.red, rec.ir, rec.fs, method=method, window=window)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Failed", str(exc))
            return
        self._cur = (rec, res, name)
        self.start_spin.setMaximum(max(0.0, rec.duration - window))
        self._draw()

    def _replot_segment(self):
        # segment controls changed: cheap redraw using the cached result
        if self._cur is not None:
            self._draw()

    def _draw(self):
        rec, res, name = self._cur
        method = self.method_combo.currentText()
        window = self.window_spin.value()
        fs = rec.fs

        # --- analysed segment ---
        s0 = int(self.start_spin.value() * fs)
        wlen = int(window * fs)
        s0 = max(0, min(s0, rec.red.size - wlen))
        red_seg = np.asarray(rec.red[s0:s0 + wlen], float)
        ir_seg = np.asarray(rec.ir[s0:s0 + wlen], float)
        t_seg = rec.time[s0:s0 + wlen]

        R_seg, f0_seg, valid_seg = ratio_of_ratios(red_seg, ir_seg, fs, method=method)
        A, B, C = self.calibration
        spo2_est = float(np.clip(A * res.R ** 2 + B * res.R + C, 0, 100))
        ref = rec.spo2_ref if rec.spo2_ref is not None else float("nan")
        err = abs(spo2_est - ref) if np.isfinite(ref) else float("nan")

        self.metrics_label.setText(
            f"{name} | fs {fs:.0f} Hz | R(median) {res.R:.3f} | "
            f"SpO2 est {spo2_est:.1f}%  ref {ref:.1f}%  |err| {err:.2f}%  | "
            f"validity {100*res.valid_fraction:.0f}%  | calib: {self._cal_note}"
        )

        self.figure.clear()
        ax_t = self.figure.add_subplot(3, 1, 1)
        ax_f = self.figure.add_subplot(3, 1, 2)
        ax_r = self.figure.add_subplot(3, 1, 3)

        # 1) time domain (z-scored so RED & IR overlay)
        def z(x):
            return (x - x.mean()) / (x.std() or 1.0)
        ax_t.plot(t_seg, z(red_seg), color="tab:red", lw=0.8, label="RED (z)")
        ax_t.plot(t_seg, z(ir_seg), color="tab:purple", lw=0.8, label="IR (z)")
        pi_red = (red_seg.max() - red_seg.min()) / (abs(red_seg.mean()) or 1) * 100
        pi_ir = (ir_seg.max() - ir_seg.min()) / (abs(ir_seg.mean()) or 1) * 100
        ax_t.set_title(f"{name} — analysed segment  "
                       f"(perfusion idx RED≈{pi_red:.1f}%, IR≈{pi_ir:.1f}%)")
        ax_t.set_xlabel("Time (s)")
        ax_t.set_ylabel("PPG (z-score)")
        ax_t.grid(alpha=0.3)
        ax_t.legend(loc="upper right", fontsize=8)

        # 2) frequency domain — the windowed double-FFT (the key feature)
        fr, mag_r, k_r, _ = spectrum(red_seg, fs, method=method)
        fi, mag_i, k_i, _ = spectrum(ir_seg, fs, method=method)
        lo, hi = HR_BAND[0] * 60, HR_BAND[1] * 60
        ax_f.axvspan(lo, hi, color="0.6", alpha=0.15,
                     label=f"HR band ({lo:.0f}–{hi:.0f} bpm)")
        # normalise each spectrum to its own peak so both are visible together
        ax_f.plot(fr * 60, mag_r / (mag_r.max() or 1), color="tab:red",
                  lw=1.0, label="RED spectrum")
        ax_f.plot(fi * 60, mag_i / (mag_i.max() or 1), color="tab:purple",
                  lw=1.0, label="IR spectrum")
        f_red, f_ir = fr[k_r] * 60, fi[k_i] * 60
        ax_f.scatter([f_red], [1.0], c="tab:red", s=45, zorder=5,
                     label=f"RED AC @ {f_red:.0f} bpm")
        ax_f.scatter([f_ir], [1.0], c="tab:purple", s=45, marker="D", zorder=5,
                     label=f"IR AC @ {f_ir:.0f} bpm")
        for h in (2, 3):                       # IR harmonics (harmonic rule)
            ax_f.axvline(h * f_ir, color="tab:purple", ls=":", alpha=0.5,
                         label="IR harmonics" if h == 2 else None)
        agree = "MATCH → valid" if valid_seg else "MISMATCH → invalid"
        ax_f.set_title(f"Windowed double-FFT spectra  |  segment R = {R_seg:.3f}  |  "
                       f"RED/IR fundamentals {agree}")
        ax_f.set_xlabel("Frequency (bpm)")
        ax_f.set_ylabel("Norm. magnitude")
        ax_f.set_xlim(0, min(fr[-1] * 60, max(300, 4 * f_ir)))
        ax_f.grid(alpha=0.3)
        ax_f.legend(loc="upper right", fontsize=8, ncol=2)

        # 3) per-window R across the record + median, mapped to SpO2
        wr = res.window_R
        ax_r.plot(np.arange(wr.size), wr, "o-", ms=3, color="tab:green",
                  lw=0.8, label="per-window R")
        ax_r.axhline(res.R, ls="--", color="tab:blue", label=f"median R = {res.R:.3f}")
        ax_r.set_title(f"Ratio-of-ratios per {window:.0f}s window  →  "
                       f"SpO2 est {spo2_est:.1f}%  (ref {ref:.1f}%, |err| {err:.2f}%)")
        ax_r.set_xlabel("window #")
        ax_r.set_ylabel("R")
        ax_r.grid(alpha=0.3)
        ax_r.legend(loc="upper right", fontsize=8)

        self.canvas.draw_idle()

    # ----------------------------------------------------------------- #
    def evaluate_all(self):
        """Compute R for every record, fit the SpO2 calibration, report MAE."""
        if not self.records:
            QtWidgets.QMessageBox.warning(self, "No records", "No records found.")
            return
        method = self.method_combo.currentText()
        window = self.window_spin.value()

        progress = QtWidgets.QProgressDialog(
            "Estimating R for all records...", "Cancel", 0, len(self.records), self)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setWindowTitle("Evaluate all")
        progress.show()

        names, R, S, vf = [], [], [], []
        for k, (kind, path, name) in enumerate(self.records):
            if progress.wasCanceled():
                break
            progress.setValue(k)
            QtWidgets.QApplication.processEvents()
            try:
                rec = load_record(kind, path)
                res = spo2_from_ppg(rec.red, rec.ir, rec.fs, method=method, window=window)
                if rec.spo2_ref is None or not np.isfinite(res.R):
                    continue
                names.append(short_label(name)); R.append(res.R)
                S.append(rec.spo2_ref); vf.append(res.valid_fraction)
            except Exception:
                continue
        progress.close()

        R, S = np.asarray(R), np.asarray(S)
        if R.size < 3:
            QtWidgets.QMessageBox.warning(self, "Too few", "Not enough valid records.")
            return

        A, B, C = fit_calibration(R, S, degree=2)
        self.calibration, self._cal_note = (A, B, C), f"fitted (n={R.size})"
        est = np.clip(A * R ** 2 + B * R + C, 0, 100)
        mae = float(np.mean(np.abs(est - S)))
        baseline = float(np.mean(np.abs(S - S.mean())))
        corr = float(np.corrcoef(R, S)[0, 1]) if R.std() else float("nan")

        self.metrics_label.setText(
            f"[{method}] records {R.size} | mean validity {100*np.mean(vf):.0f}% | "
            f"corr(R,SpO2) {corr:+.3f} | SpO2 MAE {mae:.2f}% "
            f"(predict-mean baseline {baseline:.2f}%) | calibration fitted & applied"
        )

        # plots: R vs SpO2 (with fitted curve) and per-record abs error
        self.figure.clear()
        ax1 = self.figure.add_subplot(2, 1, 1)
        ax2 = self.figure.add_subplot(2, 1, 2)

        ax1.scatter(R, S, s=26, alpha=0.75, color="tab:green", label="records")
        xs = np.linspace(R.min(), R.max(), 100)
        ax1.plot(xs, A * xs ** 2 + B * xs + C, "--", color="tab:red",
                 label="fitted SpO2 = A·R²+B·R+C")
        ax1.set_xlabel("R (ratio-of-ratios)")
        ax1.set_ylabel("Ground-truth SpO2 (%)")
        ax1.set_title(f"R vs SpO2 — all records ({method}),  corr {corr:+.3f}")
        ax1.legend(loc="best")
        ax1.grid(alpha=0.3)

        idx = np.arange(R.size)
        ax2.bar(idx, np.abs(est - S), color="tab:orange", alpha=0.85)
        ax2.axhline(mae, ls="--", color="tab:red", label=f"MAE = {mae:.2f}%")
        ax2.set_xticks(idx)
        ax2.set_xticklabels(names, rotation=90, fontsize=6)
        ax2.set_ylabel("|SpO2 error| (%)")
        ax2.set_title("Per-record absolute SpO2 error")
        ax2.legend(loc="best")
        ax2.grid(alpha=0.3, axis="y")

        self.canvas.draw_idle()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = SpO2Gui()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
