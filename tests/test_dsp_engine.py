"""Pruebas unitarias para el motor DSP y generador de señales RF."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.dsp_engine import apply_fir_filter, compute_spectrum, design_lowpass_fir
from src.report_exporter import export_html_report
from src.signal_generator import add_awgn, generate_am, generate_multitone


class TestDSPEngine(unittest.TestCase):
    def test_pure_tone_peak_frequency(self) -> None:
        fs = 16384.0
        n = 1024
        sig = generate_multitone(fs, n, tones=[(1000.0, 0.8)])
        spec = compute_spectrum(sig, fs, window_type="hann")
        self.assertAlmostEqual(spec.peak_freq_hz, 1000.0, delta=5.0)
        self.assertGreater(spec.estimated_snr_db, 40.0)

    def test_am_sidebands_presence(self) -> None:
        fs = 16384.0
        n = 1024
        sig = generate_am(fs, n, fc_hz=3000.0, fm_hz=500.0, mod_index=0.8)
        spec = compute_spectrum(sig, fs, window_type="hann")
        self.assertAlmostEqual(spec.peak_freq_hz, 3000.0, delta=5.0)

        # Verificar energía en banda lateral superior (3500 Hz) y portadora (3000 Hz)
        bin_usb = min(range(len(spec.freqs_hz)), key=lambda i: abs(spec.freqs_hz[i] - 3500.0))
        bin_lsb = min(range(len(spec.freqs_hz)), key=lambda i: abs(spec.freqs_hz[i] - 2500.0))
        self.assertGreater(spec.psd_dbfs[bin_usb], -25.0)
        self.assertGreater(spec.psd_dbfs[bin_lsb], -25.0)

    def test_awgn_reduces_snr(self) -> None:
        fs = 8192.0
        n = 512
        clean = generate_multitone(fs, n, tones=[(1024.0, 0.8)])
        noisy = add_awgn(clean, snr_db=10.0, seed=7)
        spec_clean = compute_spectrum(clean, fs)
        spec_noisy = compute_spectrum(noisy, fs)
        self.assertGreater(spec_clean.estimated_snr_db, spec_noisy.estimated_snr_db)

    def test_fir_lowpass_attenuates_high_tone(self) -> None:
        fs = 16384.0
        n = 1024
        sig = generate_multitone(fs, n, tones=[(600.0, 0.7), (5500.0, 0.7)])
        coeffs = design_lowpass_fir(cutoff_hz=2000.0, fs_hz=fs, num_taps=51)
        filtered = apply_fir_filter(sig, coeffs)
        spec = compute_spectrum(filtered, fs)
        self.assertAlmostEqual(spec.peak_freq_hz, 600.0, delta=10.0)

        bin_high = min(range(len(spec.freqs_hz)), key=lambda i: abs(spec.freqs_hz[i] - 5500.0))
        self.assertLess(spec.psd_dbfs[bin_high], -35.0)

    def test_html_report_export(self) -> None:
        fs = 8192.0
        sig = generate_multitone(fs, 256, tones=[(800.0, 0.75)])
        spec = compute_spectrum(sig, fs)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            export_html_report(out, "Test Report", sig, fs, spec)
            text = out.read_text(encoding="utf-8")
            self.assertIn("<svg", text)
            self.assertIn("Test Report", text)


if __name__ == "__main__":
    unittest.main()
