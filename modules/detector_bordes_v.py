# modules/detector_bordes_v.py
#
# =====================================================================
# BLOQUE DETECTOR DE BORDES VERTICALES — Arquitectura de Bloques v0.4
# =====================================================================
#
# Cambios respecto a v0.3:
#   - La grilla se NORMALIZA a [0,1] antes de calcular el gradiente.
#     En v0.3 el gradiente operaba sobre valores crudos (suma de strength*weight
#     por celda), que podían llegar a 30-40 en frames densos. Con threshold=0.8
#     fijo, un frame de 129pts generaba ~46% de bordes → local_loss=0.60.
#     Ahora threshold opera sobre escala [0,1]: 0.04 ≈ "borde que tiene al menos
#     4% del gradiente máximo del frame", independiente de cuántos puntos entren.
#   - edge_threshold cambia de 0.8 (absoluto, escala cruda) a 0.04 (relativo, [0,1]).
#   - _compute_gradient_x ahora recibe la grilla normalizada como argumento
#     (no lee self._grid directamente) para que la firma sea explícita.
#   - local_loss usa rango [10%-25%] igual que EsquinaBlock (era punto fijo 15%).
#     Un rango da loss=0 para cualquier ratio dentro del rango aceptable, no solo
#     para el punto exacto — más robusto ante variaciones naturales entre frames.
#   - _emit_log muestra también grad_max y grad_mean sobre la grilla normalizada
#     (valores ahora en [0,1], más fáciles de interpretar).
#   - Resto del código sin cambios (misma interfaz, misma serialización).
# =====================================================================

import time
import numpy as np
import logging
from typing import List, Tuple, Optional

try:
    from modules.monitor_block import BaseBlock
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from modules.monitor_block import BaseBlock


class VerticalEdgeDetectorBlock(BaseBlock):
    """
    Detector de bordes verticales — Bloque 2 de la arquitectura.

    Detecta cambios bruscos de intensidad en dirección horizontal
    (= bordes verticales en la imagen: flancos de letras como I, T, L, H).

    v0.4: la grilla se normaliza a [0,1] antes del gradiente. El threshold
    opera sobre escala relativa (0.04 = 4% del gradiente máximo del frame),
    lo que estabiliza la cantidad de bordes emitidos independientemente de
    la densidad de la escena.

    Parámetros configurables:
        zone_z            — zona lógica del bloque en el espacio 3D de la red
        grid_res          — resolución de la grilla NxN (default: 20×20 bins)
        edge_threshold    — gradiente mínimo en escala [0,1] para emitir borde
                            (0.04 = 4% del gradiente máximo del frame)
        output_z          — z donde se inyectan los bordes detectados en la red
        output_strength_scale — factor de escala: gradient_norm * scale = strength
        output_strength_min   — clip mínimo de fuerza
        output_strength_max   — clip máximo de fuerza
        parvo_weight      — peso del canal parvo (z≈45) en la grilla acumulada
        magno_weight      — peso del canal magno (z≈30)
        log_every         — cada cuántos .process() se imprime el log
    """

    BLOCK_TYPE = "detector_bordes_v"

    def __init__(
        self,
        zone_z: Tuple[float, float] = (65.0, 80.0),
        grid_res: int = 20,
        edge_threshold: float = 0.04,          # ← era 0.8 absoluto; ahora 0.04 relativo [0,1]
        output_z: float = 70.0,
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
            "frames_procesados":   0,
            "puntos_entrada":      0,
            "bordes_emitidos":     0,
            "ratio_bordes":        0.0,
            "gradiente_max":       0.0,   # sobre grilla normalizada [0,1]
            "gradiente_mean":      0.0,   # sobre grilla normalizada [0,1]
            "local_loss":          1.0,
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
        Toma las activaciones del Transductor y produce activaciones
        de bordes verticales en output_z.

        Entrada:
            activations — lista de tuplas (x, y, z, strength)
                          output del Transductor: strength en [3.0, 8.0]
                          z=30 (magno) o z=45 (parvo)

        Salida:
            lista de tuplas (x, y, output_z, strength) con SOLO
            los bordes detectados. vision.py las concatena a las originales.
            Lista vacía si no hay bordes detectables.
        """
        if not activations:
            return []

        self._update_count += 1
        self.health["frames_procesados"] += 1
        n_entrada = len(activations)
        self._acc_entrada += n_entrada
        self.health["puntos_entrada"] = n_entrada

        # 1. Construir la grilla 2D acumulando fuerza por celda
        self._build_grid(activations)

        # 2. Normalizar la grilla a [0,1] antes del gradiente
        #    Esto hace que edge_threshold opere en escala relativa:
        #    0.04 = "4% del máximo de activación del frame".
        #    Un frame denso (130pts) y uno escaso (20pts) producen
        #    cantidades comparables de bordes, no depende de la escala cruda.
        grid_max = float(self._grid.max())
        if grid_max < 1e-6:
            # Grilla vacía o ruido puro — no hay bordes
            if (self._update_count % self.log_every == 0 or
                    (time.time() - self._last_log_time) > 45.0):
                self._emit_log()
                self._last_log_time = time.time()
                self._acc_entrada = 0
                self._acc_bordes  = 0
                self._acc_loss    = 0.0
            return []

        grid_norm = self._grid / grid_max   # [0,1]

        # 3. Calcular gradiente horizontal sobre la grilla normalizada
        gradient = self._compute_gradient_x(grid_norm)

        # 4. Umbralizar y emitir activaciones
        edge_acts = self._emit_edges(gradient)

        # 5. Actualizar métricas
        n_bordes = len(edge_acts)
        self._acc_bordes += n_bordes
        self.health["bordes_emitidos"] = n_bordes

        ratio = n_bordes / max(1, n_entrada)
        self.health["ratio_bordes"] = ratio

        abs_grad = np.abs(gradient)
        self.health["gradiente_max"]  = float(abs_grad.max())
        nonzero = abs_grad[abs_grad > 0]
        self.health["gradiente_mean"] = float(nonzero.mean()) if nonzero.size > 0 else 0.0

        # local_loss: rango aceptable [10%-25%] → loss=0 (igual que EsquinaBlock)
        # Más robusto que punto fijo 15%: no penaliza variaciones naturales.
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
    # SUB-FUNCIÓN 1 — Construcción de la grilla 2D
    # ------------------------------------------------------------------

    def _build_grid(self, activations: list) -> None:
        """
        Acumula la fuerza de cada activación en la celda (row, col) correspondiente.
        Canal parvo (z≈45): más peso. Canal magno (z≈30): peso bajo.
        La normalización posterior hace que los pesos absolutos no sean críticos.
        """
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

            w = self.parvo_weight if abs(z - 45.0) < 5.0 else self.magno_weight
            self._grid[row, col] += strength * w

    # ------------------------------------------------------------------
    # SUB-FUNCIÓN 2 — Gradiente horizontal
    # ------------------------------------------------------------------

    def _compute_gradient_x(self, grid_norm: np.ndarray) -> np.ndarray:
        """
        Calcula el gradiente horizontal de la grilla NORMALIZADA.
        Resultado en [-1, 1] aprox. (grilla de entrada en [0,1]).

        np.gradient(axis=1):
          - Puntos interiores: (grid[i,j+1] - grid[i,j-1]) / 2
          - Bordes: diferencia unilateral

        Un gradiente grande → borde vertical en esa posición (independiente del signo).
        """
        return np.gradient(grid_norm, axis=1)

    # ------------------------------------------------------------------
    # SUB-FUNCIÓN 3 — Umbralización y emisión
    # ------------------------------------------------------------------

    def _emit_edges(self, gradient: np.ndarray) -> list:
        """
        Genera activaciones en celdas donde |gradiente| > edge_threshold.
        gradient está en escala normalizada [0,1] aprox.
        edge_threshold=0.04 → detecta bordes con al menos 4% del gradiente máximo.
        """
        abs_grad = np.abs(gradient)
        rows, cols = np.where(abs_grad > self.edge_threshold)

        if len(rows) == 0:
            return []

        result = []
        grid_res = self.grid_res

        for r, c in zip(rows, cols):
            grad_val = float(abs_grad[r, c])

            strength = float(np.clip(
                grad_val * self.output_strength_scale,
                self.output_strength_min,
                self.output_strength_max
            ))

            x = (c + 0.5) / grid_res
            y = (r + 0.5) / grid_res

            result.append((x, y, self.output_z, strength))

        return result

    # ------------------------------------------------------------------
    # LOG DE DIAGNÓSTICO
    # ------------------------------------------------------------------

    def _emit_log(self):
        h   = self.health
        frames_desde_ultimo = min(self._update_count, self.log_every) if self._update_count > 0 else 1
        avg_entrada = self._acc_entrada / max(1, frames_desde_ultimo)
        avg_bordes  = self._acc_bordes  / max(1, frames_desde_ultimo)
        avg_loss    = self._acc_loss    / max(1, frames_desde_ultimo)
        avg_ratio   = avg_bordes / max(1, avg_entrada)

        print(
            f"🔍 [BordesV] "
            f"frames={h['frames_procesados']} | "
            f"entrada={avg_entrada:.0f}pts | "
            f"bordes={avg_bordes:.0f} ({avg_ratio*100:.0f}%) | "
            f"grad_max={h['gradiente_max']:.3f} "    # ahora en [0,1]
            f"grad_mean={h['gradiente_mean']:.3f} | "
            f"local_loss={avg_loss:.3f} | "
            f"output_z={self.output_z}",
            flush=True
        )

    # ------------------------------------------------------------------
    # REGISTRO EN LA RED
    # ------------------------------------------------------------------

    def register(self, net):
        net.modules[self.block_id] = self
        print(
            f"🔍 [BordesV] Registrado | "
            f"grid={self.grid_res}x{self.grid_res} | "
            f"threshold={self.edge_threshold} (normalizado [0,1]) | "
            f"output_z={self.output_z} | "
            f"log_every={self.log_every} | "
            f"parvo_w={self.parvo_weight} magno_w={self.magno_weight}",
            flush=True
        )

    # ------------------------------------------------------------------
    # SERIALIZACIÓN
    # ------------------------------------------------------------------

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
    def from_dict(cls, d: dict) -> "VerticalEdgeDetectorBlock":
        b = cls(
            zone_z                = tuple(d.get("zone_z",                (65.0, 80.0))),
            grid_res              = d.get("grid_res",                    20),
            edge_threshold        = d.get("edge_threshold",              0.04),
            output_z              = d.get("output_z",                    70.0),
            output_strength_scale = d.get("output_strength_scale",       8.0),
            output_strength_min   = d.get("output_strength_min",         3.0),
            output_strength_max   = d.get("output_strength_max",         8.0),
            parvo_weight          = d.get("parvo_weight",                1.0),
            magno_weight          = d.get("magno_weight",                0.3),
        )
        b.health    = d.get("health", b.health)
        b._injected = True
        return b