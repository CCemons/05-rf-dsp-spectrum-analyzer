"""
Demostración de síntesis y análisis espectral de señales AM, FM y Multitono con ruido AWGN.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dsp_engine import apply_fir_filter, compute_spectrum, design_lowpass_fir
from src.report_exporter import export_html_report
from src.signal_generator import add_awgn, generate_am, generate_fm, generate_multitone


def main() -> None:
    fs_hz = 16384.0
    num_samples = 1024

    # 1. Señal AM con portadora en 2.5 kHz y moduladora en 400 Hz
    am_clean = generate_am(fs_hz, num_samples, fc_hz=2500.0, fm_hz=400.0, mod_index=0.7)
    am_noisy = add_awgn(am_clean, snr_db=28.0)
    am_spec = compute_spectrum(am_noisy, fs_hz, window_type="hann")
    print(
        f"[AM] Peak={am_spec.peak_freq_hz} Hz ({am_spec.peak_power_dbfs} dBFS) | "
        f"SNR={am_spec.estimated_snr_db} dB | THD={am_spec.thd_pct}%"
    )

    # 2. Señal FM en 3.2 kHz
    fm_sig = add_awgn(generate_fm(fs_hz, num_samples, fc_hz=3200.0, fm_hz=250.0, freq_dev_hz=600.0), snr_db=30.0)
    fm_spec = compute_spectrum(fm_sig, fs_hz, window_type="blackman")
    print(
        f"[FM] Peak={fm_spec.peak_freq_hz} Hz ({fm_spec.peak_power_dbfs} dBFS) | "
        f"NoiseFloor={fm_spec.noise_floor_dbfs} dBFS"
    )

    # 3. Multitono + Filtrado Pasa-Bajos FIR
    multi = generate_multitone(fs_hz, num_samples, tones=[(800.0, 0.7), (5200.0, 0.5)])
    lpf_coeffs = design_lowpass_fir(cutoff_hz=2000.0, fs_hz=fs_hz, num_taps=51)
    filtered = apply_fir_filter(multi, lpf_coeffs)
    filt_spec = compute_spectrum(filtered, fs_hz, window_type="hann")
    print(
        f"[LPF] Tono restante Peak={filt_spec.peak_freq_hz} Hz | "
        f"SNR={filt_spec.estimated_snr_db} dB"
    )

    out_html = Path(__file__).resolve().parent.parent / "spectrum_report.html"
    export_html_report(out_html, "Análisis Espectral AM + AWGN (2.5 kHz)", am_noisy, fs_hz, am_spec)
    print(f"Reporte exportado en: {out_html}")


if __name__ == "__main__":
    main()
