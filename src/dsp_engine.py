"""
Motor de Procesamiento Digital de Señales (DSP):
- Ventanas espectrales (Rect, Hann, Hamming, Blackman)
- FFT Radix-2 iterativa (Cooley-Tukey) con fallback puro Python / aceleración NumPy
- Densidad espectral de potencia en dBFS e interpolación parabólica de picos
- Estimación de SNR, THD (%) y diseño de filtro FIR pasa-bajos por ventana Sinc
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from statistics import median


@dataclass
class SpectrumResult:
    freqs_hz: list[float]
    magnitudes_linear: list[float]
    psd_dbfs: list[float]
    peak_freq_hz: float
    peak_power_dbfs: float
    noise_floor_dbfs: float
    estimated_snr_db: float
    thd_pct: float
    window_type: str


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def next_power_of_two(n: int) -> int:
    """Devuelve la potencia de 2 mayor o igual a n."""
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


def window_coefficients(num_samples: int, window_type: str = "hann") -> list[float]:
    """Calcula los coeficientes de la ventana espectral solicitada."""
    if num_samples <= 1:
        return [1.0] * max(0, num_samples)

    kind = window_type.lower()
    denom = num_samples - 1
    if kind in ("rect", "rectangular"):
        return [1.0] * num_samples
    if kind == "hann":
        return [0.5 * (1.0 - math.cos((2.0 * math.pi * n) / denom)) for n in range(num_samples)]
    if kind == "hamming":
        return [0.54 - 0.46 * math.cos((2.0 * math.pi * n) / denom) for n in range(num_samples)]
    if kind == "blackman":
        return [
            0.42
            - 0.5 * math.cos((2.0 * math.pi * n) / denom)
            + 0.08 * math.cos((4.0 * math.pi * n) / denom)
            for n in range(num_samples)
        ]
    raise ValueError(f"Ventana no soportada: {window_type}")


def fft_radix2(samples: list[complex | float]) -> list[complex]:
    """
    Transformada Rápida de Fourier (FFT) Cooley-Tukey iterativa en orden bit-reverso.
    Si la longitud no es potencia de 2, aplica zero-padding automático.
    """
    n = len(samples)
    if n == 0:
        return []

    target_n = n if _is_power_of_two(n) else next_power_of_two(n)
    buf: list[complex] = [complex(x) for x in samples] + [0j] * (target_n - n)

    # Permutación bit-reverso
    j = 0
    for i in range(1, target_n):
        bit = target_n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            buf[i], buf[j] = buf[j], buf[i]

    # Mariposas iterativas
    size = 2
    while size <= target_n:
        half = size // 2
        angle_step = -2.0 * math.pi / size
        w_step = cmath.exp(1j * angle_step)
        for start in range(0, target_n, size):
            w = 1.0 + 0j
            for k in range(half):
                u = buf[start + k]
                t = w * buf[start + k + half]
                buf[start + k] = u + t
                buf[start + k + half] = u - t
                w *= w_step
        size <<= 1

    return buf


def compute_spectrum(
    signal: list[float],
    fs_hz: float,
    window_type: str = "hann",
) -> SpectrumResult:
    """Calcula el espectro unilateral en dBFS y extrae métricas RF/DSP clave."""
    if len(signal) < 8 or fs_hz <= 0:
        raise ValueError("La señal debe tener al menos 8 muestras y fs_hz > 0.")

    n_orig = len(signal)
    n_fft = n_orig if _is_power_of_two(n_orig) else next_power_of_two(n_orig)

    win = window_coefficients(n_orig, window_type)
    coherent_gain = sum(win) / n_orig
    windowed = [x * w for x, w in zip(signal, win)]

    try:
        import numpy as np  # type: ignore

        padded = np.zeros(n_fft, dtype=float)
        padded[:n_orig] = windowed
        fft_vals = np.fft.rfft(padded)
        half_n = len(fft_vals) - 1
        raw_mags = np.abs(fft_vals).tolist()
    except ImportError:
        fft_vals = fft_radix2(windowed + [0.0] * (n_fft - n_orig))
        half_n = n_fft // 2
        raw_mags = [abs(fft_vals[k]) for k in range(half_n + 1)]

    norm_factor = coherent_gain * n_orig
    mags: list[float] = []
    psd_dbfs: list[float] = []
    freqs_hz: list[float] = []

    for k in range(half_n + 1):
        scale = 1.0 if (k == 0 or k == half_n) else 2.0
        amp = (raw_mags[k] * scale) / norm_factor
        mags.append(amp)
        psd_dbfs.append(20.0 * math.log10(max(amp, 1e-12)))
        freqs_hz.append((k * fs_hz) / n_fft)

    # Localizar pico fundamental (ignorando DC bin 0)
    peak_idx = max(range(1, len(psd_dbfs)), key=lambda idx: psd_dbfs[idx])
    peak_db = psd_dbfs[peak_idx]

    # Interpolación parabólica para resolución sub-bin
    if 1 < peak_idx < len(psd_dbfs) - 1:
        alpha = psd_dbfs[peak_idx - 1]
        beta = psd_dbfs[peak_idx]
        gamma = psd_dbfs[peak_idx + 1]
        denom = alpha - 2.0 * beta + gamma
        delta = 0.5 * (alpha - gamma) / denom if abs(denom) > 1e-12 else 0.0
    else:
        delta = 0.0

    bin_res = fs_hz / n_fft
    peak_freq_hz = (peak_idx + delta) * bin_res

    # Piso de ruido estimado por mediana espectral (robusto frente a picos discretos)
    noise_floor_db = float(median(psd_dbfs[1:]))

    # Relación señal a ruido en el dominio espectral (excluyendo lóbulos cercanos al pico)
    signal_power = sum(mags[idx] ** 2 for idx in range(max(1, peak_idx - 2), min(len(mags), peak_idx + 3)))
    noise_bins = [
        mags[idx] ** 2
        for idx in range(1, len(mags))
        if abs(idx - peak_idx) > 3
    ]
    noise_power = sum(noise_bins) if noise_bins else 1e-15
    snr_db = 10.0 * math.log10(max(signal_power, 1e-15) / max(noise_power, 1e-15))

    # Distorsión Armónica Total (THD) sobre armónicos 2..5 dentro de Nyquist
    fundamental_amp = max(mags[peak_idx], 1e-12)
    harmonic_sq_sum = 0.0
    for h in range(2, 6):
        h_freq = h * peak_freq_hz
        if h_freq >= fs_hz * 0.49:
            break
        h_bin = round(h_freq / bin_res)
        if 1 <= h_bin < len(mags):
            local_peak = max(mags[max(1, h_bin - 1) : min(len(mags), h_bin + 2)])
            harmonic_sq_sum += local_peak ** 2

    thd_pct = (math.sqrt(harmonic_sq_sum) / fundamental_amp) * 100.0

    return SpectrumResult(
        freqs_hz=freqs_hz,
        magnitudes_linear=mags,
        psd_dbfs=psd_dbfs,
        peak_freq_hz=round(peak_freq_hz, 2),
        peak_power_dbfs=round(peak_db, 2),
        noise_floor_dbfs=round(noise_floor_db, 2),
        estimated_snr_db=round(snr_db, 2),
        thd_pct=round(thd_pct, 3),
        window_type=window_type,
    )


def design_lowpass_fir(cutoff_hz: float, fs_hz: float, num_taps: int = 51) -> list[float]:
    """Diseña un filtro digital FIR pasa-bajos de fase lineal por ventana Sinc."""
    if num_taps < 3 or num_taps % 2 == 0:
        raise ValueError("num_taps debe ser un entero impar >= 3.")
    if not (0.0 < cutoff_hz < fs_hz * 0.5):
        raise ValueError("cutoff_hz debe estar entre 0 y fs_hz / 2.")

    fc_norm = cutoff_hz / fs_hz
    mid = (num_taps - 1) / 2.0
    hamming = window_coefficients(num_taps, "hamming")
    coeffs: list[float] = []

    for n in range(num_taps):
        offset = n - mid
        if abs(offset) < 1e-12:
            sinc_val = 2.0 * fc_norm
        else:
            sinc_val = math.sin(2.0 * math.pi * fc_norm * offset) / (math.pi * offset)
        coeffs.append(sinc_val * hamming[n])

    gain = sum(coeffs)
    return [c / gain for c in coeffs]


def apply_fir_filter(signal: list[float], coeffs: list[float]) -> list[float]:
    """Aplica convolución causal directa con los coeficientes FIR dados."""
    num_taps = len(coeffs)
    out: list[float] = []
    for n in range(len(signal)):
        acc = 0.0
        for k in range(num_taps):
            if n - k >= 0:
                acc += coeffs[k] * signal[n - k]
        out.append(acc)
    return out
