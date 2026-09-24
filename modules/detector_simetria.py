# modules/detector_simetria.py
#
# =====================================================================
# BLOQUE DETECTOR DE SIMETRÍA — Arquitectura de Bloques v0.1
# =====================================================================
#
# Posición en el pipeline:
#   Transductor → ... → [SimetriaDetectorBlock] → z=82
#
# Qué hace:
#   Opera sobre la grilla NxN (mismo input que BordesV/H).
#   Calcula DOS scores de simetría por frame:
#
#   Simetría Vertical (espejo izquierda↔derecha):
#     diff_v[i,j] = |grid[i,j] - grid[i, N-1-j]|
#     sym_v = 1 - mean(diff_v) / (max_val + eps)
#     → Alta para letras simétricas: X, O, A, T
#     → Baja para letras asimétricas: E
#
#   Simetría Horizontal (espejo arriba↔abajo):
#     diff_h[i,j] = |grid[i,j] - grid[N-1-i, j]|
#     sym_h = 1 - mean(diff_h) / (max_val + eps)
#     → Alta para: X, O
#     → Media para: T (barra arriba, vertical abajo → no simétrica H)
#     → Baja para: A, E
#
#   Emite activaciones en sym_v_z y sym_h_z separadas para que el
#   LetraDetector pueda usarlas como features ortogonales.
#
# Valor discriminante por letra:
#   X  — sym_v ALTA, sym_h ALTA  (perfectamente simétrica en ambos ejes)
#   O  — sym_v ALTA, sym_h ALTA
#   A  — sym_v ALTA, sym_h BAJA  (punta arriba, base abajo)
#   T  — sym_v ALTA, sym_h BAJA  (barra arriba, pie abajo)
#   E  — sym_v BAJA, sym_h BAJA  (barras solo a la derecha)
#
# Emite en dos z distintos para máxima información:
#   sym_v_z = 82.0   (simetría vertical/bilateral)
#   sym_h_z = 84.0   (simetría horizontal/radial)
#
# =====================================================================

import time
import numpy as np
from typing import Tuple

try:
    from modules.monitor_block import BaseBlock
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from modules.monitor_block import BaseBlock


class SimetriaDetectorBlock(BaseBlock):
    """
    Detector de simetría bilateral y radial — Bloque 2e de la arquitectura.

    Detecta cuán simétrica es la distribución de intensidad de la grilla
    respecto al eje vertical (bilateral) y horizontal (arriba/abajo).

    Emite activaciones uniformemente distribuidas en la grilla cuando
    se detecta alta simetría — la FUERZA de la señal codifica el score.
    """

    BLOCK_TYPE = "detector_simetria"

    def __init__(
        self,
        zone_z: Tuple[float, float] = (120.0, 130.0),
        grid_res: int = 20,
        sym_v_z: float = 82.0,
        sym_h_z: float = 84.0,
        sym_threshold: float = 0.55,
        output_strength_scale: float = 8.0,
        output_strength_min: float = 2.0,
        output_strength_max: float = 8.0,
        parvo_weight: float = 1.0,
        magno_weight: float = 0.3,
        min_pts: int = 15,
        n_emit_points: int = 8,
        log_every: int = 25,
    ):
        super().__init__(
            block_id   = self.BLOCK_TYPE,
            zone_z     = zone_z,
            n_immortal = 0,
            log_every  = log_every,
        )

        self.grid_res              = grid_res
        self.sym_v_z               = sym_v_z
        self.sym_h_z               = sym_h_z
        self.sym_threshold         = sym_threshold
        self.output_strength_scale = output_strength_scale
        self.output_strength_min   = output_strength_min
        self.output_strength_max   = output_strength_max
        self.parvo_weight          = parvo_weight
        self.magno_weight          = magno_weight
        self.min_pts               = min_pts
        self.n_emit_points         = n_emit_points

        self._grid = np.zeros((grid_res, grid_res), dtype=np.float32)

        # Posiciones de emisión: cuadrícula uniforme de n_emit_points puntos
        side = int(np.ceil(np.sqrt(n_emit_points)))
        xs = np.linspace(0.1, 0.9, side)
        ys = np.linspace(0.1, 0.9, side)
        self._emit_pos = [(x, y) for y in ys for x in xs][:n_emit_points]

        self.health.update({
            "frames_procesados":  0,
            "frames_descartados": 0,
            "puntos_entrada":     0,
            "sym_v":              0.0,
            "sym_h":              0.0,
            "local_loss":         1.0,
        })

        self._acc_loss     = 0.0
        self._last_log_time = 0.0
        self._injected = True

    # ------------------------------------------------------------------
    # MÉTODO PRINCIPAL
    # ------------------------------------------------------------------

    def process(self, activations: list) -> list:
        """
        Recibe output del Transductor (z=30,45).
        Produce activaciones de simetría en sym_v_z y sym_h_z.
        """
        if not activations:
            return []

        self._update_count += 1
        n_entrada = len(activations)
        self.health["puntos_entrada"] = n_entrada

        if n_entrada < self.min_pts:
            self.health["frames_descartados"] += 1
            self._acc_loss += 1.0
            return []

        self.health["frames_procesados"] += 1

        # 1. Construir grilla
        self._build_grid(activations)

        grid_max = float(self._grid.max())
        if grid_max < 1e-6:
            return []

        grid_norm = self._grid / grid_max

        # 2. Calcular scores de simetría
        sym_v = self._compute_sym_vertical(grid_norm)
        sym_h = self._compute_sym_horizontal(grid_norm)

        self.health["sym_v"] = sym_v
        self.health["sym_h"] = sym_h

        # 3. Emitir activaciones si se supera el umbral
        result = []
        if sym_v > self.sym_threshold:
            strength_v = float(np.clip(
                sym_v * self.output_strength_scale,
                self.output_strength_min,
                self.output_strength_max
            ))
            for px, py in self._emit_pos:
                result.append((px, py, self.sym_v_z, strength_v))

        if sym_h > self.sym_threshold:
            strength_h = float(np.clip(
                sym_h * self.output_strength_scale,
                self.output_strength_min,
                self.output_strength_max
            ))
            for px, py in self._emit_pos:
                result.append((px, py, self.sym_h_z, strength_h))

        # 4. Loss: queremos que al menos uno de los dos se detecte
        best_sym = max(sym_v, sym_h)
        ideal_lo, ideal_hi = 0.55, 1.0
        if ideal_lo <= best_sym <= ideal_hi:
            local_loss = 0.0
        else:
            local_loss = min(1.0, (ideal_lo - best_sym) / ideal_lo)

        self._acc_loss += local_loss
        self.health["local_loss"] = local_loss

        _now = time.time()
        if (self._update_count % self.log_every == 0 or
                (_now - self._last_log_time) > 45.0):
            self._emit_log()
            self._last_log_time = _now
            self._acc_loss = 0.0

        return result

    # ------------------------------------------------------------------
    # SUB-FUNCIONES
    # ------------------------------------------------------------------

    def _build_grid(self, activations: list) -> None:
        self._grid.fill(0.0)
        for act in activations:
            try:
                x, y, z, strength = (
                    float(act[0]), float(act[1]),
                    float(act[2]), float(act[3])
                )
            except (IndexError, TypeError, ValueError):
                continue
            col = int(min(x, 0.9999) * self.grid_res)
            row = int(min(y, 0.9999) * self.grid_res)
            w   = self.parvo_weight if abs(z - 45.0) < 5.0 else self.magno_weight
            self._grid[row, col] += strength * w

    def _compute_sym_vertical(self, grid_norm: np.ndarray) -> float:
        """
        Simetría bilateral (izq ↔ der).
        1.0 = perfectamente simétrica, 0.0 = totalmente asimétrica.
        """
        flipped = grid_norm[:, ::-1]
        diff    = np.abs(grid_norm - flipped)
        return float(1.0 - diff.mean())

    def _compute_sym_horizontal(self, grid_norm: np.ndarray) -> float:
        """
        Simetría arriba ↔ abajo.
        1.0 = perfectamente simétrica, 0.0 = totalmente asimétrica.
        """
        flipped = grid_norm[::-1, :]
        diff    = np.abs(grid_norm - flipped)
        return float(1.0 - diff.mean())

    # ------------------------------------------------------------------
    # LOG
    # ------------------------------------------------------------------

    def _emit_log(self):
        h        = self.health
        frames_d = min(self._update_count, self.log_every) if self._update_count > 0 else 1
        avg_loss = self._acc_loss / max(1, frames_d)
        sym_v_str = "✅" if h["sym_v"] > self.sym_threshold else "—"
        sym_h_str = "✅" if h["sym_h"] > self.sym_threshold else "—"
        print(
            f"🪞 [Simetría] "
            f"frames={h['frames_procesados']} desc={h['frames_descartados']} | "
            f"sym_v={h['sym_v']:.3f}{sym_v_str} sym_h={h['sym_h']:.3f}{sym_h_str} | "
            f"threshold={self.sym_threshold} | "
            f"loss={avg_loss:.3f} | "
            f"z_v={self.sym_v_z} z_h={self.sym_h_z}",
            flush=True
        )

    # ------------------------------------------------------------------
    # REGISTRO Y SERIALIZACIÓN
    # ------------------------------------------------------------------

    def register(self, net):
        net.modules[self.block_id] = self
        print(
            f"🪞 [Simetría] Registrado | "
            f"grid={self.grid_res}x{self.grid_res} | "
            f"sym_v_z={self.sym_v_z} sym_h_z={self.sym_h_z} | "
            f"threshold={self.sym_threshold} | "
            f"log_every={self.log_every}",
            flush=True
        )

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "grid_res":              self.grid_res,
            "sym_v_z":               self.sym_v_z,
            "sym_h_z":               self.sym_h_z,
            "sym_threshold":         self.sym_threshold,
            "output_strength_scale": self.output_strength_scale,
            "output_strength_min":   self.output_strength_min,
            "output_strength_max":   self.output_strength_max,
            "parvo_weight":          self.parvo_weight,
            "magno_weight":          self.magno_weight,
            "min_pts":               self.min_pts,
            "n_emit_points":         self.n_emit_points,
        })
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SimetriaDetectorBlock":
        b = cls(
            zone_z                = tuple(d.get("zone_z",                (120.0, 130.0))),
            grid_res              = d.get("grid_res",                    20),
            sym_v_z               = d.get("sym_v_z",                    82.0),
            sym_h_z               = d.get("sym_h_z",                    84.0),
            sym_threshold         = d.get("sym_threshold",              0.55),
            output_strength_scale = d.get("output_strength_scale",       8.0),
            output_strength_min   = d.get("output_strength_min",         2.0),
            output_strength_max   = d.get("output_strength_max",         8.0),
            parvo_weight          = d.get("parvo_weight",                1.0),
            magno_weight          = d.get("magno_weight",                0.3),
            min_pts               = d.get("min_pts",                     15),
            n_emit_points         = d.get("n_emit_points",               8),
        )
        b.health    = d.get("health", b.health)
        b._injected = True
        return b
