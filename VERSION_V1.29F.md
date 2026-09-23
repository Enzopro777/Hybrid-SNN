# IA 1.6.29f

## Readout Stability & Bridge Credit Control

1.6.29f es una intervención experimental sobre el cuello de botella H0/Latent → readout físico. Parte de 1.6.29e y deja sin cambios Cortex, TemporalCore, Event Queue, contrato de 60 frames, gaze reward, REM-like y la arquitectura de radio.

### Cambios principales

- `drive_ema` queda fuera de la decisión; sólo observabilidad histórica.
- Homeostasis unilateral: sólo eleva el umbral de pools claramente hiperactivos; nunca baja el umbral por baja actividad.
- Estado homeostático nuevo, sin offsets negativos heredados.
- LTP target gobernado por déficit de margen real, no por `1-score`.
- LTD restringida al rival que realmente compite con el target.
- Normalización simétrica de columnas y escalado sináptico lento desactivados durante el experimento.
- Replay físico del bridge desactivado por defecto; replay NREM del workspace latente se conserva.
- Nuevas auditorías de homeostasis, margen objetivo, saturación de threshold y estado legado.

### Arranque experimental

`CLEAN_EXPERIMENT_DEFAULT = True`. Para el experimento recomendado no se debe copiar ningún `simulation_state_*`, journal ni checkpoint de versiones anteriores. Los pesos nacen de la configuración base y la primera sesión empieza sin estado homeostático aprendido.
