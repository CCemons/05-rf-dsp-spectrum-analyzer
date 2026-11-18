# Analizador de Espectro RF & Procesamiento Digital de Señales (DSP)

Suite modular en **Python** enfocada en ingeniería electrónica, telecomunicaciones y radio definida por software (**SDR**). Implementa síntesis de formas de onda RF/banda base, cálculo de transformada rápida de Fourier (**FFT Radix-2**), medición de métricas espectrales (**SNR**, **THD**, piso de ruido) y diseño de filtros digitales **FIR** con exportación automática de reportes visuales en **HTML + SVG**.

---

## Características Técnicas

- **Motor FFT Híbrido**: Implementación nativa en Python puro del algoritmo iterativo **Cooley-Tukey Radix-2** (cero dependencias obligatorias) con aceleración automática por `numpy.fft.rfft` cuando está instalado.
- **Ventanas Espectrales Calibradas**: Soporte para `Rect`, `Hann`, `Hamming` y `Blackman` con normalización por ganancia coherente.
- **Métricas Espectrales Precisas**:
  - Interpolación parabólica sub-bin para estimar la frecuencia exacta del pico.
  - Estimación robusta del piso de ruido mediante mediana espectral.
  - Cálculo de relación señal a ruido (**SNR en dB**) y Distorsión Armónica Total (**THD %**).
- **Filtro Digital FIR Pasa-Bajos**: Diseño por ventana Sinc + Hamming y convolución causal directa.
- **Reportes Vectoriales**: Generación de documentos HTML autocontenidos con gráficas SVG en dominio del tiempo y frecuencia.

---

## Estructura del Repositorio

```text
05-rf-dsp-spectrum-analyzer/
├── src/
│   ├── signal_generator.py  # AM, FM, Chirp, Multitono y AWGN
│   ├── dsp_engine.py        # FFT Radix-2, PSD dBFS, SNR, THD y filtros FIR
│   └── report_exporter.py   # Exportador HTML + SVG autocontenido
├── examples/
│   └── demo_am_fm_spectrum.py
├── tests/
│   └── test_dsp_engine.py
├── requirements.txt
└── README.md
```

---

## Ejecución Rápida

```bash
# Ejecutar demostración AM/FM/FIR y generar reporte spectrum_report.html
python examples/demo_am_fm_spectrum.py

# Ejecutar suite de pruebas unitarias
python -m unittest discover -s tests -v
```

---

## Autor

**Camilo Castañeda** · Proyecto Open Source bajo licencia **MIT**.
