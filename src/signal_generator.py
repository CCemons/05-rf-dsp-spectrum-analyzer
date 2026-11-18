"""
Generador de señales de prueba para análisis espectral RF y DSP.
Soporta multitono, modulación AM, modulación FM, barrido Chirp y ruido AWGN.
"""

from __future__ import annotations

import math
import random


def time_axis(fs_hz: float, num_samples: int) -> list[float]:
    """Construye el vector de tiempo en segundos para fs_hz y num_samples."""
    if fs_hz <= 0 or num_samples <= 0:
        raise ValueError("fs_hz y num_samples deben ser positivos.")
    return [n / fs_hz for n in range(num_samples)]


def generate_multitone(
    fs_hz: float,
    num_samples: int,
    tones: list[tuple[float, float]],
) -> list[float]:
    """Genera una suma de sinusoides [(frecuencia_hz, amplitud), ...]."""
    t_vec = time_axis(fs_hz, num_samples)
    signal = [0.0] * num_samples
    for freq_hz, amplitude in tones:
        omega = 2.0 * math.pi * freq_hz
        for idx, t in enumerate(t_vec):
            signal[idx] += amplitude * math.sin(omega * t)
    return signal


def generate_am(
    fs_hz: float,
    num_samples: int,
    fc_hz: float,
    fm_hz: float,
    mod_index: float = 0.65,
    amplitude: float = 0.6,
) -> list[float]:
    """Sintetiza una señal modulada en amplitud (AM DSB-LC)."""
    t_vec = time_axis(fs_hz, num_samples)
    wc = 2.0 * math.pi * fc_hz
    wm = 2.0 * math.pi * fm_hz
    return [
        amplitude * (1.0 + mod_index * math.cos(wm * t)) * math.cos(wc * t)
        for t in t_vec
    ]


def generate_fm(
    fs_hz: float,
    num_samples: int,
    fc_hz: float,
    fm_hz: float,
    freq_dev_hz: float = 1200.0,
    amplitude: float = 0.9,
) -> list[float]:
    """Sintetiza una señal modulada en frecuencia (FM) mediante índice beta."""
    if fm_hz <= 0:
        raise ValueError("fm_hz debe ser mayor que cero.")
    t_vec = time_axis(fs_hz, num_samples)
    wc = 2.0 * math.pi * fc_hz
    wm = 2.0 * math.pi * fm_hz
    beta = freq_dev_hz / fm_hz
    return [amplitude * math.cos(wc * t + beta * math.sin(wm * t)) for t in t_vec]


def generate_chirp(
    fs_hz: float,
    num_samples: int,
    f0_hz: float,
    f1_hz: float,
    amplitude: float = 0.9,
) -> list[float]:
    """Genera un barrido lineal de frecuencia (Chirp) entre f0_hz y f1_hz."""
    t_vec = time_axis(fs_hz, num_samples)
    duration = num_samples / fs_hz
    slope = (f1_hz - f0_hz) / duration
    return [
        amplitude * math.sin(2.0 * math.pi * (f0_hz * t + 0.5 * slope * t * t))
        for t in t_vec
    ]


def add_awgn(signal: list[float], snr_db: float, seed: int = 42) -> list[float]:
    """Añade Ruido Blanco Gaussiano Aditivo (AWGN) calibrado a snr_db."""
    if not signal:
        return []
    sig_power = sum(x * x for x in signal) / len(signal)
    if sig_power <= 0.0:
        sig_power = 1.0

    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise_std = math.sqrt(noise_power)
    rng = random.Random(seed)
    return [x + rng.gauss(0.0, noise_std) for x in signal]
