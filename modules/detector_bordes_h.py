# modules/detector_bordes_h.py
#
# =====================================================================
# BLOQUE DETECTOR DE BORDES HORIZONTALES — Arquitectura de Bloques v0.4
# =====================================================================
#
# Cambios respecto a v0.3 (idénticos a los de detector_bordes_v.py v0.4):
#   - La grilla se NORMALIZA a [0,1] antes de calcular el gradiente.
#     En v0.3 el gradiente operaba sobre valores crudos (acumulación de
#     strength*weight por celda). Con threshold=0.8 fijo y frames densos,
#     la mayoría de las celdas lo superaban → ratio muy alto → local_loss alto.
#     Ahora threshold=0.04 opera sobre escala relativa [0,1]: "borde con al
#     menos 4% del gradiente máximo del frame", independiente de la densidad.
#   - edge_threshold: 0.8 (absoluto, escala cruda) → 0.04 (relativo, [0,1]).
#   - output_strength_scale: 0.8 → 8.0 (el gradiente normalizado vive en
#     [0, ~0.5]; con 0.8 la fuerza salía siempre clippeada al mínimo 3.0).
#   - _compute_gradient_y recibe la grilla normalizada como argumento
#     (firma explícita, no lee self._grid directamente).
#   - local_loss usa rango [10%-25%] en vez de punto fijo 15%. Consistente
#     con EsquinaBlock y BordesV v0.4.
#   - Log muestra grad_max/grad_mean sobre grilla normalizada ([0,1]).
#
# Lo que NO cambia:
#   - BLOCK_TYPE = "detector_bordes_h"
#   - _compute_gradient_y usa axis=0 (bordes horizontales), no axis=1
#   - output_z = 72.0 (distinto de BordesV=70.0)
#   - zone_z = (80.0, 92.0)
#   - Toda la interfaz, serialización y registro sin cambios.
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


class HorizontalEdgeDetectorBlock(BaseBlock):
    """
    Detector de bordes horizontales — Bloque 2 de la arquitectura (paralelo a BordesV).

    Detecta cambios bruscos de intensidad en dirección vertical
    (= bordes horizontales en la imagen: techos y travesaños de letras).

    v0.4: la grilla se normaliza a [0,1] antes del gradiente. El threshold
    opera sobre escala relativa, estabilizando la cantidad de bordes emitidos
    independientemente de la densidad de la escena.

    Parámetros:
        zone_z            — zona lógica en el espacio 3D
        grid_res          — resolución de la grilla NxN
        edge_threshold    — gradiente mínimo en escala [0,1] (0.04 = 4% del máximo)
        output_z          — z donde se inyectan los bordes (default: 72.0)
        output_strength_scale — factor: gradient_norm * scale = strength
        output_strength_min / max — clip de fuerza de salida
        parvo_weight      — peso del canal parvo (formas estáticas)
        magno_weight      — peso del canal magno (movimiento)
        log_every         — frecuencia del log de diagnóstico
    """

    BLOCK_TYPE = "detector_bordes_h"

    def __init__(
        self,
        zone_z: Tuple[float, float] = (80.0, 92.0),
        grid_res: int = 20,
        edge_threshold: float = 0.04,          # ← era 0.8 absoluto; ahora 0.04 relativo [0,1]
        output_z: float = 72.0,
        output_strength_scale: float = 8.0,    # ← era 0.8; ajustado a escala [0,1] de la grilla
        output_strength_min: float = 3.0,
        output_strength_max: float = 8.0,
        parvo_weight: float = 1.0,
        magno_weight: float = 0.3,
        log_every: int = 25,
    ):
        super().__init__(
            block_id   = self.BLOCK_TYPE,
            zone_z     = zone_z,
            n_immortal = 0,
            log_every  = log_every,
        )

        self.grid_res              = grid_res
        self.edge_threshold        = edge_threshold
        self.output_z              = output_z
        self.output_strength_scale = output_strength_scale
        self.output_strength_min   = output_strength_min
        self.output_strength_max   = output_strength_max
        self.parvo_weight          = parvo_weight
        self.magno_weight          = magno_weight

        self._grid = np.zeros((grid_res, grid_res), dtype=np.float32)

        self.health.update({
            "frames_procesados":  0,
            "puntos_entrada":     0,
            "bordes_emitidos":    0,
            "ratio_bordes":       0.0,
            "gradiente_max":      0.0,   # sobre grilla normalizada [0,1]
            "gradiente_mean":     0.0,   # sobre grilla normalizada [0,1]
            "local_loss":         1.0,
        })

        self._acc_entrada = 0
        self._acc_bordes  = 0
        self._acc_loss    = 0.0

        self._last_log_time = 0.0

        self._injected = True

    # ------------------------------------------------------------------
    # MÉTODO PRINCIPAL
    # ------------------------------------------------------------------

    def process(self, activations: list) -> list:
        """
        Recibe output del Transductor (mismo input que BordesV).
        Produce activaciones de bordes HORIZONTALES en output_z=72.
        """
        if not activations:
            return []

        self._update_count += 1
        self.health["frames_procesados"] += 1
        n_entrada = len(activations)
        self._acc_entrada += n_entrada
        self.health["puntos_entrada"] = n_entrada

        # 1. Construir grilla
        self._build_grid(activations)

        # 2. Normalizar a [0,1] antes del gradiente
        grid_max = float(self._grid.max())
        if grid_max < 1e-6:
            if (self._update_count % self.log_every == 0 or
                    (time.time() - self._last_log_time) > 45.0):
                self._emit_log()
                self._last_log_time = time.time()
                self._acc_entrada = 0
                self._acc_bordes  = 0
                self._acc_loss    = 0.0
            return []

        grid_norm = self._grid / grid_max   # [0,1]

        # 3. Gradiente vertical sobre grilla normalizada → bordes horizontales
        gradient = self._compute_gradient_y(grid_norm)

        # 4. Umbralizar y emitir
        edge_acts = self._emit_edges(gradient)

        # 5. Métricas
        n_bordes = len(edge_acts)
        self._acc_bordes += n_bordes
        self.health["bordes_emitidos"] = n_bordes

        ratio = n_bordes / max(1, n_entrada)
        self.health["ratio_bordes"] = ratio

        abs_grad = np.abs(gradient)
        self.health["gradiente_max"]  = float(abs_grad.max())
        nonzero = abs_grad[abs_grad > 0]
        self.health["gradiente_mean"] = float(nonzero.mean()) if nonzero.size > 0 else 0.0

        # local_loss: rango [10%-25%] → loss=0 (igual que BordesV v0.4 y EsquinaBlock)
        ideal_lo, ideal_hi = 0.10, 0.25
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
            self._acc_bordes  = 0
            self._acc_loss    = 0.0

        return edge_acts

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

    def _compute_gradient_y(self, grid_norm: np.ndarray) -> np.ndarray:
        """
        Gradiente VERTICAL de la grilla normalizada → bordes horizontales.
        np.gradient(axis=0): diferencias entre filas (dirección y).
        Resultado en [-1, 1] aprox. (entrada en [0,1]).
        Un gradiente grande entre filas adyacentes = borde horizontal.
        """
        return np.gradient(grid_norm, axis=0)   # ← axis=0, no axis=1

    def _emit_edges(self, gradient: np.ndarray) -> list:
        abs_grad = np.abs(gradient)
        rows, cols = np.where(abs_grad > self.edge_threshold)
        if len(rows) == 0:
            return []
        result   = []
        grid_res = self.grid_res
        for r, c in zip(rows, cols):
            strength = float(np.clip(
                float(abs_grad[r, c]) * self.output_strength_scale,
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
        frames_desde_ultimo = min(self._update_count, self.log_every) if self._update_count > 0 else 1
        avg_entrada = self._acc_entrada / max(1, frames_desde_ultimo)
        avg_bordes  = self._acc_bordes  / max(1, frames_desde_ultimo)
        avg_loss    = self._acc_loss    / max(1, frames_desde_ultimo)
        avg_ratio   = avg_bordes / max(1, avg_entrada)
        print(
            f"🔍 [BordesH] "
            f"frames={h['frames_procesados']} | "
            f"entrada={avg_entrada:.0f}pts | "
            f"bordes={avg_bordes:.0f} ({avg_ratio*100:.0f}%) | "
            f"grad_max={h['gradiente_max']:.3f} "   # ahora en [0,1]
            f"grad_mean={h['gradiente_mean']:.3f} | "
            f"local_loss={avg_loss:.3f} | "
            f"output_z={self.output_z}",
            flush=True
        )

    # ------------------------------------------------------------------
    # REGISTRO Y SERIALIZACIÓN
    # ------------------------------------------------------------------

    def register(self, net):
        net.modules[self.block_id] = self
        print(
            f"🔍 [BordesH] Registrado | "
            f"grid={self.grid_res}x{self.grid_res} | "
            f"threshold={self.edge_threshold} (normalizado [0,1]) | "
            f"output_z={self.output_z} | "
            f"log_every={self.log_every} | "
            f"parvo_w={self.parvo_weight} magno_w={self.magno_weight}",
            flush=True
        )

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "grid_res":              self.grid_res,
            "edge_threshold":        self.edge_threshold,
            "output_z":              self.output_z,
            "output_strength_scale": self.output_strength_scale,
            "output_strength_min":   self.output_strength_min,
            "output_strength_max":   self.output_strength_max,
            "parvo_weight":          self.parvo_weight,
            "magno_weight":          self.magno_weight,
        })
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "HorizontalEdgeDetectorBlock":
        b = cls(
            zone_z                = tuple(d.get("zone_z",                (80.0, 92.0))),
            grid_res              = d.get("grid_res",                    20),
            edge_threshold        = d.get("edge_threshold",              0.04),
            output_z              = d.get("output_z",                    72.0),
            output_strength_scale = d.get("output_strength_scale",       8.0),
            output_strength_min   = d.get("output_strength_min",         3.0),
            output_strength_max   = d.get("output_strength_max",         8.0),
            parvo_weight          = d.get("parvo_weight",                1.0),
            magno_weight          = d.get("magno_weight",                0.3),
        )
        b.health    = d.get("health", b.health)
        b._injected = True
        return b