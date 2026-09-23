# Changelog IA 1.6.29f

## Readout stability
- Se elimina el uso decisional de `drive_ema`.
- Se reemplaza la homeostasis simétrica 1.6.27 por una corrección unilateral contra hiperactividad.
- Se invalida el estado de homeostasis/drive de checkpoints anteriores a 29f.

## Plasticidad
- El target sólo aprende si el margen frente al rival superior está por debajo del margen deseado.
- La depresión rival se limita al rival superior y a una competición real.
- Replay físico del bridge OFF por defecto.

## Experimento
- Normalización de columnas y escalado sináptico heredados no participan en el circuito de 29f.
- `CLEAN_EXPERIMENT_DEFAULT=True`.
