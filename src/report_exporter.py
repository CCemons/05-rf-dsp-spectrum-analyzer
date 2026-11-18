"""
Exportador de reportes espectrales autocontenidos en HTML5 + SVG.
Genera gráficas vectoriales nítidas del dominio del tiempo y espectro en dBFS.
"""

from __future__ import annotations

import html
from pathlib import Path

from .dsp_engine import SpectrumResult


def _polyline_points(
    xs: list[float],
    ys: list[float],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    width: int,
    height: int,
    pad: int = 36,
) -> str:
    plot_w = width - 2 * pad
    plot_h = height - 2 * pad
    x_span = (x_max - x_min) or 1.0
    y_span = (y_max - y_min) or 1.0

    pts: list[str] = []
    for x_val, y_val in zip(xs, ys):
        px = pad + ((x_val - x_min) / x_span) * plot_w
        py = pad + plot_h - ((y_val - y_min) / y_span) * plot_h
        px = max(pad, min(width - pad, px))
        py = max(pad, min(height - pad, py))
        pts.append(f"{px:.1f},{py:.1f}")
    return " ".join(pts)


def export_html_report(
    output_path: Path | str,
    title: str,
    signal: list[float],
    fs_hz: float,
    spectrum: SpectrumResult,
) -> Path:
    """Crea un reporte visual HTML+SVG con la señal temporal, espectro FFT y métricas."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    width, height = 840, 260
    time_limit = min(len(signal), 256)
    t_ms = [(i / fs_hz) * 1000.0 for i in range(time_limit)]
    sig_slice = signal[:time_limit]
    max_abs = max((abs(v) for v in sig_slice), default=1.0) or 1.0

    time_pts = _polyline_points(
        t_ms, sig_slice, t_ms[0], t_ms[-1], -max_abs * 1.1, max_abs * 1.1, width, height
    )

    spec_pts = _polyline_points(
        spectrum.freqs_hz,
        spectrum.psd_dbfs,
        0.0,
        fs_hz / 2.0,
        -110.0,
        5.0,
        width,
        height,
    )

    safe_title = html.escape(title)
    document = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <title>{safe_title}</title>
  <style>
    body {{ font-family: "Segoe UI", system-ui, sans-serif; background: #0b0f19; color: #f8fafc; margin: 0; padding: 24px; }}
    .wrap {{ max-width: 920px; margin: 0 auto; display: grid; gap: 18px; }}
    .card {{ background: #111827; border: 1px solid rgba(148,163,184,.2); border-radius: 14px; padding: 18px; }}
    h1, h2 {{ margin: 0 0 10px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }}
    .m-box {{ background: #070b14; padding: 12px; border-radius: 10px; border: 1px solid rgba(148,163,184,.16); }}
    .m-box span {{ display: block; font-size: 12px; color: #94a3b8; }}
    .m-box strong {{ font-size: 18px; color: #22d3ee; font-family: Consolas, monospace; }}
    svg {{ width: 100%; height: auto; background: #070b14; border-radius: 10px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>{safe_title}</h1>
      <p style="color:#94a3b8;margin:0;">Fs = {fs_hz:.0f} Hz · Ventana = {html.escape(spectrum.window_type)} · Muestras = {len(signal)}</p>
    </div>

    <div class="metrics">
      <div class="m-box"><span>Frecuencia Pico</span><strong>{spectrum.peak_freq_hz:.2f} Hz</strong></div>
      <div class="m-box"><span>Potencia Pico</span><strong>{spectrum.peak_power_dbfs:.2f} dBFS</strong></div>
      <div class="m-box"><span>SNR Estimado</span><strong>{spectrum.estimated_snr_db:.2f} dB</strong></div>
      <div class="m-box"><span>Piso de Ruido</span><strong>{spectrum.noise_floor_dbfs:.2f} dBFS</strong></div>
      <div class="m-box"><span>THD</span><strong>{spectrum.thd_pct:.3f} %</strong></div>
    </div>

    <div class="card">
      <h2>Dominio del Tiempo (Primeras {time_limit} muestras)</h2>
      <svg viewBox="0 0 {width} {height}">
        <line x1="36" y1="{height/2}" x2="{width-36}" y2="{height/2}" stroke="#334155" stroke-dasharray="4" />
        <polyline fill="none" stroke="#10b981" stroke-width="2" points="{time_pts}" />
      </svg>
    </div>

    <div class="card">
      <h2>Espectro de Potencia Unilateral (0 — {fs_hz/2:.0f} Hz)</h2>
      <svg viewBox="0 0 {width} {height}">
        <line x1="36" y1="36" x2="{width-36}" y2="36" stroke="#1e293b" />
        <line x1="36" y1="{height/2}" x2="{width-36}" y2="{height/2}" stroke="#1e293b" stroke-dasharray="4" />
        <polyline fill="none" stroke="#06b6d4" stroke-width="2" points="{spec_pts}" />
      </svg>
    </div>
  </div>
</body>
</html>
"""
    out_file.write_text(document, encoding="utf-8")
    return out_file
