# modules/combinador_junction.py
#
# =====================================================================
# BLOQUE COMBINADOR DE JUNCTIONS — Arquitectura de Bloques v0.1
# =====================================================================
#
# Posición en el pipeline:
#   BordesV (z=70) + BordesH (z=72) + Barra (z=74) + Slash (z=76)
#                        ↓
#               [JunctionBlock]
#                        ↓ junction_z=86
#   → LetraDetector
#
# Qué hace:
#   Un "junction" (punto de cruce) ocurre cuando en la misma celda de
#   la grilla coinciden señales de AL MENOS DOS tipos de borde distintos
#   con suficiente intensidad.
#
#   Esto detecta:
#     - T-junction: BordesV ALTO + BordesH ALTO en la misma celda
#       → indica que una barra horizontal "termina" sobre una vertical
#       → muy presente en T, A (en la intersección del trazo central)
#     - X-junction: BordesV + BordesH + Barra + Slash coinciden
#       → indica cruce de dos líneas diagonales = presente en X
#     - L-junction: BordesV ALTO + BordesH ALTO pero WITHOUT diagonal
#       → presente en E (esquinas internas) y A
#
#   Señal de junction = mínimo de las dos señales más fuertes presentes
#   en cada celda. Si solo hay un tipo, no hay junction.
#
# Valor discriminante por letra:
#   X  — junctions ALTOS con diagonal presente (X-junction)
#   A  — junction MEDIO en la intersección de las barras (T-junction)
#   T  — junction ALTO en el centro superior (T-junction: H sobre V)
#   E  — junctions MEDIOS en las esquinas internas (L-junctions)
#   O  — junctions BAJOS (contorno sin cruces reales)
#
# =====================================================================

import time
import numpy as np
from typing import Tuple, List

try:
    from modules.monitor_block import BaseBlock
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from modules.monitor_block import BaseBlock


class JunctionBlock(BaseBlock):
    """
    Combinador de junctions — Bloque 3b de la arquitectura.

    Detecta puntos de cruce entre distintos tipos de borde:
      - T-junction (V+H coincidentes)
      - X-junction (V+H+diagonal coincidentes)
      - L-junction (V+H en esquina, sin diagonal)

    Recibe el frame completo de activaciones y filtra por z.
    Emite en junction_z=86.
    """

    BLOCK_TYPE = "combinador_junction"

    def __init__(
        self,
        zone_z: Tuple[float, float] = (130.0, 140.0),
        grid_res: int = 20,
        v_z:     float = 70.0,
        h_z:     float = 72.0,
        barra_z: float = 74.0,
        slash_z: float = 76.0,
        z_tol:   float = 1.0,
        junction_threshold: float = 0.35,
        junction_z: float = 86.0,
        output_strength_scale: float = 9.0,
        output_strength_min:   float = 3.0,
        output_strength_max:   float = 9.0,
        min_bordes: int = 10,
        log_every: int = 25,
    ):
        super().__init__(
            block_id   = self.BLOCK_TYPE,
            zone_z     = zone_z,
            n_immortal = 0,
            log_every  = log_every,
        )

        self.grid_res              = grid_res
        self.v_z                   = v_z
        self.h_z                   = h_z
        self.barra_z               = barra_z
        self.slash_z               = slash_z
        self.z_tol                 = z_tol
        self.junction_threshold    = junction_threshold
        self.junction_z            = junction_z
        self.output_strength_scale = output_strength_scale
        self.output_strength_min   = output_strength_min
        self.output_strength_max   = output_strength_max
        self.min_bordes            = min_bordes

        N = grid_res
        self._v_grid     = np.zeros((N, N), dtype=np.float32)
        self._h_grid     = np.zeros((N, N), dtype=np.float32)
        self._barra_grid = np.zeros((N, N), dtype=np.float32)
        self._slash_grid = np.zeros((N, N), dtype=np.float32)

        self.health.update({
            "frames_procesados":   0,
            "frames_descartados":  0,
            "junctions_emitidos":  0,
            "pts_v": 0, "pts_h": 0, "pts_barra": 0, "pts_slash": 0,
            "junc_mean":           0.0,
            "local_loss":          1.0,
        })

        self._acc_junctions = 0
        self._acc_loss      = 0.0
        self._last_log_time = 0.0
        self._injected = True

    # ------------------------------------------------------------------
    # MÉTODO PRINCIPAL
    # ------------------------------------------------------------------

    def process(self, activations: list) -> list:
        """
        Recibe TODAS las activaciones del frame (múltiples z).
        Detecta junctions donde coinciden ≥2 tipos de borde.
        """
        if not activations:
            return []

        self._update_count += 1

        # Separar por canal
        acts_v     = [a for a in activations if abs(float(a[2]) - self.v_z)     <= self.z_tol]
        acts_h     = [a for a in activations if abs(float(a[2]) - self.h_z)     <= self.z_tol]
        acts_barra = [a for a in activations if abs(float(a[2]) - self.barra_z) <= self.z_tol]
        acts_slash = [a for a in activations if abs(float(a[2]) - self.slash_z) <= self.z_tol]

        self.health["pts_v"]     = len(acts_v)
        self.health["pts_h"]     = len(acts_h)
        self.health["pts_barra"] = len(acts_barra)
        self.health["pts_slash"] = len(acts_slash)

        # Necesitamos al menos V y H para detectar junctions
        if len(acts_v) < self.min_bordes or len(acts_h) < self.min_bordes:
            self.health["frames_descartados"] += 1
            self._acc_loss += 1.0
            return []

        self.health["frames_procesados"] += 1

        # Construir grillas
        self._build_grid(acts_v,     self._v_grid)
        self._build_grid(acts_h,     self._h_grid)
        self._build_grid(acts_barra, self._barra_grid)
        self._build_grid(acts_slash, self._slash_grid)

        # Normalizar cada grilla
        grids = []
        for g in [self._v_grid, self._h_grid, self._barra_grid, self._slash_grid]:
            gmax = float(g.max())
            grids.append(g / gmax if gmax > 1e-6 else np.zeros_like(g))

        v_n, h_n, b_n, s_n = grids

        # Señal de junction = suma ponderada de mínimos por pares
        # T-junction: V ∩ H
        junction_vh = np.minimum(v_n, h_n)
        # X-junction: V ∩ H ∩ (barra o slash) — cualquier diagonal
        diag_n = np.maximum(b_n, s_n)
        junction_x  = np.minimum(junction_vh, diag_n)

        # Señal final: mezcla ponderada (T + X)
        # X-junction pesa más porque es más discriminante para X
        junction_map = 0.6 * junction_vh + 0.4 * junction_x

        # Emitir
        junction_acts = self._emit_junctions(junction_map)

        # Métricas
        n_junc = len(junction_acts)
        self._acc_junctions += n_junc
        self.health["junctions_emitidos"] = n_junc

        active = junction_map[junction_map > self.junction_threshold]
        self.health["junc_mean"] = float(active.mean()) if active.size > 0 else 0.0

        avg_bordes = (len(acts_v) + len(acts_h)) / 2.0
        ratio = n_junc / max(1, avg_bordes)
        ideal_lo, ideal_hi = 0.05, 0.25
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
            self._acc_junctions = 0
            self._acc_loss      = 0.0

        return junction_acts

    # ------------------------------------------------------------------
    # SUB-FUNCIONES
    # ------------------------------------------------------------------

    def _build_grid(self, activations: list, grid: np.ndarray) -> None:
        grid.fill(0.0)
        for act in activations:
            try:
                x, y, strength = float(act[0]), float(act[1]), float(act[3])
            except (IndexError, TypeError, ValueError):
                continue
            col = int(min(x, 0.9999) * self.grid_res)
            row = int(min(y, 0.9999) * self.grid_res)
            grid[row, col] += strength

    def _emit_junctions(self, junction_map: np.ndarray) -> list:
        rows, cols = np.where(junction_map > self.junction_threshold)
        if len(rows) == 0:
            return []
        result   = []
        grid_res = self.grid_res
        for r, c in zip(rows, cols):
            strength = float(np.clip(
                float(junction_map[r, c]) * self.output_strength_scale,
                self.output_strength_min,
                self.output_strength_max
            ))
            x = (c + 0.5) / grid_res
            y = (r + 0.5) / grid_res
            result.append((x, y, self.junction_z, strength))
        return result

    # ------------------------------------------------------------------
    # LOG
    # ------------------------------------------------------------------

    def _emit_log(self):
        h        = self.health
        frames_d = min(self._update_count, self.log_every) if self._update_count > 0 else 1
        avg_junc = self._acc_junctions / max(1, frames_d)
        avg_loss = self._acc_loss      / max(1, frames_d)
        print(
            f"✚ [Junctions] "
            f"frames={h['frames_procesados']} desc={h['frames_descartados']} | "
            f"v={h['pts_v']} h={h['pts_h']} b={h['pts_barra']} s={h['pts_slash']} | "
            f"junctions={avg_junc:.0f} | "
            f"junc_mean={h['junc_mean']:.3f} | "
            f"loss={avg_loss:.3f} | "
            f"output_z={self.junction_z}",
            flush=True
        )

    # ------------------------------------------------------------------
    # REGISTRO Y SERIALIZACIÓN
    # ------------------------------------------------------------------

    def register(self, net):
        net.modules[self.block_id] = self
        print(
            f"✚ [Junctions] Registrado | "
            f"grid={self.grid_res}x{self.grid_res} | "
            f"v_z={self.v_z} h_z={self.h_z} | "
            f"threshold={self.junction_threshold} | "
            f"junction_z={self.junction_z} | "
            f"log_every={self.log_every}",
            flush=True
        )

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "grid_res":              self.grid_res,
            "v_z":                   self.v_z,
            "h_z":                   self.h_z,
            "barra_z":               self.barra_z,
            "slash_z":               self.slash_z,
            "z_tol":                 self.z_tol,
            "junction_threshold":    self.junction_threshold,
            "junction_z":            self.junction_z,
            "output_strength_scale": self.output_strength_scale,
            "output_strength_min":   self.output_strength_min,
            "output_strength_max":   self.output_strength_max,
            "min_bordes":            self.min_bordes,
        })
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "JunctionBlock":
        b = cls(
            zone_z                = tuple(d.get("zone_z",                (130.0, 140.0))),
            grid_res              = d.get("grid_res",                    20),
            v_z                   = d.get("v_z",                         70.0),
            h_z                   = d.get("h_z",                         72.0),
            barra_z               = d.get("barra_z",                     74.0),
            slash_z               = d.get("slash_z",                     76.0),
            z_tol                 = d.get("z_tol",                       1.0),
            junction_threshold    = d.get("junction_threshold",          0.35),
            junction_z            = d.get("junction_z",                  86.0),
            output_strength_scale = d.get("output_strength_scale",        9.0),
            output_strength_min   = d.get("output_strength_min",          3.0),
            output_strength_max   = d.get("output_strength_max",          9.0),
            min_bordes            = d.get("min_bordes",                   10),
        )
        b.health    = d.get("health", b.health)
        b._injected = True
        return b
