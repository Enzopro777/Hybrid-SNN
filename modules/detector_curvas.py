# modules/detector_curvas.py
#
# =====================================================================
# BLOQUE DETECTOR DE CURVAS/REDONDEZ — Arquitectura de Bloques v0.1
# =====================================================================
#
# Posición en el pipeline:
#   Transductor → ... → [CurvaDetectorBlock] → z=78
#   → LetraDetector (usa curv_z=78 como feature adicional)
#
# Qué hace:
#   Opera sobre la grilla NxN del Transductor (z=30 magno, z=45 parvo).
#   Detecta regiones con DISTRIBUCIÓN CIRCULAR de intensidad usando la
#   diferencia entre el gradiente radial y el tangencial.
#
#   Método: Estructura Tensor simplificado
#     Para cada celda (i,j), calcula Gx y Gy (gradiente 2D).
#     El "curvature score" local = |Gx*Gy| / (Gx²+Gy²+eps)
#     → Cerca de 0 cuando el gradiente es puramente horizontal o vertical
#       (→ bordes rectos)
#     → Cerca de 0.5 cuando el gradiente tiene componente en 45° (→ curvas)
#
#   Valor discriminante por letra:
#     O  — curv ALTA (contorno cerrado y curvo)
#     X  — curv MEDIA (brazos diagonales tienen componente 45°)
#     A  — curv BAJA-MEDIA
#     T  — curv BAJA (solo rectas)
#     E  — curv BAJA (solo rectas)
#
# Parámetros:
#   curv_threshold  — score mínimo para emitir punto de curva [0,1]
#   output_z        — z donde se inyectan las curvas (default: 78.0)
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


class CurvaDetectorBlock(BaseBlock):
    """
    Detector de curvas/redondez — Bloque 2d de la arquitectura.

    Detecta regiones con distribución circular de gradiente (indicador
    de contornos curvos versus bordes rectos).

    Emite activaciones en output_z=78.
    El LetraDetector puede usar curv_z=78 como feature discriminante
    (O vs T/E) sin modificación de su grilla principal.
    """

    BLOCK_TYPE = "detector_curvas"

    def __init__(
        self,
        zone_z: Tuple[float, float] = (110.0, 120.0),
        grid_res: int = 20,
        curv_threshold: float = 0.08,
        output_z: float = 78.0,
        output_strength_scale: float = 7.0,
        output_strength_min: float = 3.0,
        output_strength_max: float = 8.0,
        parvo_weight: float = 1.0,
        magno_weight: float = 0.3,
        min_pts: int = 10,
        log_every: int = 25,
    ):
        super().__init__(
            block_id   = self.BLOCK_TYPE,
            zone_z     = zone_z,
            n_immortal = 0,
            log_every  = log_every,
        )

        self.grid_res              = grid_res
        self.curv_threshold        = curv_threshold
        self.output_z              = output_z
        self.output_strength_scale = output_strength_scale
        self.output_strength_min   = output_strength_min
        self.output_strength_max   = output_strength_max
        self.parvo_weight          = parvo_weight
        self.magno_weight          = magno_weight
        self.min_pts               = min_pts

        self._grid = np.zeros((grid_res, grid_res), dtype=np.float32)

        self.health.update({
            "frames_procesados":  0,
            "frames_descartados": 0,
            "puntos_entrada":     0,
            "curvas_emitidas":    0,
            "curv_mean":          0.0,
            "local_loss":         1.0,
        })

        self._acc_entrada  = 0
        self._acc_curvas   = 0
        self._acc_loss     = 0.0
        self._last_log_time = 0.0
        self._injected = True

    # ------------------------------------------------------------------
    # MÉTODO PRINCIPAL
    # ------------------------------------------------------------------

    def process(self, activations: list) -> list:
        """
        Recibe output del Transductor (z=30,45).
        Produce activaciones de curvas/redondez en output_z=78.
        """
        if not activations:
            return []

        self._update_count += 1
        n_entrada = len(activations)
        self.health["puntos_entrada"] = n_entrada
        self._acc_entrada += n_entrada

        if n_entrada < self.min_pts:
            self.health["frames_descartados"] += 1
            self._acc_loss += 1.0
            return []

        self.health["frames_procesados"] += 1

        # 1. Construir grilla
        self._build_grid(activations)

        # 2. Normalizar
        grid_max = float(self._grid.max())
        if grid_max < 1e-6:
            return []
        grid_norm = self._grid / grid_max

        # 3. Calcular mapa de curvatura
        curv_map = self._compute_curvature(grid_norm)

        # 4. Emitir
        curv_acts = self._emit_curves(curv_map)

        # 5. Métricas
        n_curvas = len(curv_acts)
        self._acc_curvas += n_curvas
        self.health["curvas_emitidas"] = n_curvas

        active = curv_map[curv_map > self.curv_threshold]
        self.health["curv_mean"] = float(active.mean()) if active.size > 0 else 0.0

        ratio = n_curvas / max(1, n_entrada)
        ideal_lo, ideal_hi = 0.05, 0.30
        if ideal_lo <= ratio <= ideal_hi:
            local_loss = 0.0
        elif ratio < ideal_lo:
            local_loss = min(1.0, (ideal_lo - ratio) / ideal_lo)
        else:
            local_loss = min(1.0, (ratio - ideal_hi) / ideal_hi)

        self._acc_loss += local_loss
        self.health["local_loss"] = local_loss

        _now = time.time()
        if (self._update_count % self.log_every == 0 or
                (_now - self._last_log_time) > 45.0):
            self._emit_log()
            self._last_log_time = _now
            self._acc_entrada = 0
            self._acc_curvas  = 0
            self._acc_loss    = 0.0

        return curv_acts

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

    def _compute_curvature(self, grid_norm: np.ndarray) -> np.ndarray:
        """
        Calcula curvatura local usando el producto cruzado normalizado
        de los gradientes Gx y Gy.
        curv[i,j] = |Gx*Gy| / (Gx²+Gy²+eps)
        → 0 en bordes puramente ortogonales, máximo en ángulos diagonales (curvas).
        """
        Gy = np.gradient(grid_norm, axis=0)  # gradiente vertical
        Gx = np.gradient(grid_norm, axis=1)  # gradiente horizontal
        eps = 1e-8
        curv = np.abs(Gx * Gy) / (Gx**2 + Gy**2 + eps)
        return curv.astype(np.float32)

    def _emit_curves(self, curv_map: np.ndarray) -> list:
        rows, cols = np.where(curv_map > self.curv_threshold)
        if len(rows) == 0:
            return []
        result   = []
        grid_res = self.grid_res
        for r, c in zip(rows, cols):
            strength = float(np.clip(
                float(curv_map[r, c]) * self.output_strength_scale,
                self.output_strength_min,
                self.output_strength_max
            ))
            x = (c + 0.5) / grid_res
            y = (r + 0.5) / grid_res
            result.append((x, y, self.output_z, strength))
        return result

    # ------------------------------------------------------------------
    # LOG
    # ------------------------------------------------------------------

    def _emit_log(self):
        h   = self.health
        frames_d  = min(self._update_count, self.log_every) if self._update_count > 0 else 1
        avg_ent   = self._acc_entrada / max(1, frames_d)
        avg_curv  = self._acc_curvas  / max(1, frames_d)
        avg_loss  = self._acc_loss    / max(1, frames_d)
        avg_ratio = avg_curv / max(1, avg_ent)
        print(
            f"〇 [Curvas] "
            f"frames={h['frames_procesados']} desc={h['frames_descartados']} | "
            f"entrada={avg_ent:.0f}pts | "
            f"curvas={avg_curv:.0f} ({avg_ratio*100:.0f}%) | "
            f"curv_mean={h['curv_mean']:.3f} | "
            f"loss={avg_loss:.3f} | "
            f"output_z={self.output_z}",
            flush=True
        )

    # ------------------------------------------------------------------
    # REGISTRO Y SERIALIZACIÓN
    # ------------------------------------------------------------------

    def register(self, net):
        net.modules[self.block_id] = self
        print(
            f"〇 [Curvas] Registrado | "
            f"grid={self.grid_res}x{self.grid_res} | "
            f"threshold={self.curv_threshold} | "
            f"output_z={self.output_z} | "
            f"log_every={self.log_every}",
            flush=True
        )

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "grid_res":              self.grid_res,
            "curv_threshold":        self.curv_threshold,
            "output_z":              self.output_z,
            "output_strength_scale": self.output_strength_scale,
            "output_strength_min":   self.output_strength_min,
            "output_strength_max":   self.output_strength_max,
            "parvo_weight":          self.parvo_weight,
            "magno_weight":          self.magno_weight,
            "min_pts":               self.min_pts,
        })
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CurvaDetectorBlock":
        b = cls(
            zone_z                = tuple(d.get("zone_z",                (110.0, 120.0))),
            grid_res              = d.get("grid_res",                    20),
            curv_threshold        = d.get("curv_threshold",              0.08),
            output_z              = d.get("output_z",                    78.0),
            output_strength_scale = d.get("output_strength_scale",       7.0),
            output_strength_min   = d.get("output_strength_min",         3.0),
            output_strength_max   = d.get("output_strength_max",         8.0),
            parvo_weight          = d.get("parvo_weight",                1.0),
            magno_weight          = d.get("magno_weight",                0.3),
            min_pts               = d.get("min_pts",                     10),
        )
        b.health    = d.get("health", b.health)
        b._injected = True
        return b
