# modules/detector_diagonales.py
#
# =====================================================================
# BLOQUE DETECTOR DE DIAGONALES — Arquitectura de Bloques v0.1
# =====================================================================
#
# Posición en el pipeline:
#   Transductor → BordesV (z=70) → BordesH (z=72)
#                      ↓ transductor_out (mismo input que V y H)
#              [DiagonalDetectorBlock]
#                      ↓ barra_z=74  slash_z=76
#              → Esquinas (z=80) — recibe las 4 señales
#              → LetraDetector (z=37-53)
#
# Qué hace:
#   Opera sobre la grilla 20×20 de intensidades del canal parvo/magno
#   (mismo input que BordesV y BordesH — el output del Transductor).
#   Calcula dos gradientes diagonales independientes:
#
#   Diagonal BARRA (tipo barra): compara cada celda (i,j) con (i+1, j+1).
#     Detecta flancos que bajan de izquierda a derecha → presente en X, A.
#
#   Diagonal SLASH (/): compara cada celda (i,j) con (i+1, j-1).
#     Detecta flancos que bajan de derecha a izquierda → presente en X, A, E(esquina).
#
#   Emite activaciones en barra_z y slash_z respectivamente.
#   El EsquinaBlock recibe las 4 señales (v_z, h_z, barra_z, slash_z)
#   cuando se configure para ello. Por ahora ya genera el flujo de datos.
#
# Implementación del gradiente diagonal:
#   Se calcula como la diferencia entre la grilla y su versión desplazada:
#     grad_barra[i,j] = grid[i+1,j+1] - grid[i,j]   (interior: NxN→(N-1)×(N-1))
#     grad_slash[i,j] = grid[i+1,j-1] - grid[i,j]
#   El resultado se redimensiona a NxN con padding edge para mantener
#   la misma resolución que los otros detectores.
#
# Misma interfaz que VerticalEdgeDetectorBlock:
#   - process(activations) → list of (x, y, z, strength)
#   - register(net)
#   - to_dict() / from_dict()
#   - BLOCK_TYPE para BLOCK_REGISTRY
# =====================================================================

import time
import numpy as np
import logging
from typing import List, Tuple

try:
    from modules.monitor_block import BaseBlock
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from modules.monitor_block import BaseBlock


class DiagonalDetectorBlock(BaseBlock):
    """
    Detector de diagonales — Bloque 2c de la arquitectura.

    Detecta cambios bruscos de intensidad en las dos direcciones diagonales:
      - Diagonal BARRA (tipo barra): top-left → bottom-right. Emite en barra_z.
      - Diagonal SLASH (/): top-right → bottom-left. Emite en slash_z.

    Opera sobre el mismo input del Transductor que BordesV y BordesH
    (activaciones z=30 magno, z=45 parvo), por lo que recibe transductor_out
    directamente.

    Valor discriminante por letra:
      X  — barra ALTA, slash ALTA   (los dos brazos diagonales)
      A  — barra MEDIA, slash MEDIA (los dos lados del triángulo)
      O  — barra BAJA,  slash BAJA  (sin diagonales pronunciadas)
      T  — barra BAJA,  slash BAJA  (solo horizontales + vertical)
      E  — barra BAJA,  slash BAJA  (solo horizontales + vertical izquierdo)

    Parámetros configurables:
        zone_z                — zona lógica del bloque en el espacio 3D
        grid_res              — resolución de la grilla NxN (debe coincidir con V y H)
        edge_threshold        — gradiente mínimo en [0,1] para emitir diagonal
        barra_z               — z donde se inyectan las diagonales '\'
        slash_z               — z donde se inyectan las diagonales '/'
        output_strength_scale — fuerza = gradiente_norm * scale
        output_strength_min   — clip mínimo de fuerza
        output_strength_max   — clip máximo de fuerza
        parvo_weight          — peso del canal parvo (z≈45)
        magno_weight          — peso del canal magno (z≈30)
        log_every             — cada cuántos .process() se imprime el log
    """

    BLOCK_TYPE = "detector_diagonales"

    def __init__(
        self,
        zone_z: Tuple[float, float] = (100.0, 110.0),
        grid_res: int = 20,
        edge_threshold: float = 0.04,
        barra_z: float = 74.0,
        slash_z: float = 76.0,
        output_strength_scale: float = 8.0,
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
        self.barra_z               = barra_z
        self.slash_z               = slash_z
        self.output_strength_scale = output_strength_scale
        self.output_strength_min   = output_strength_min
        self.output_strength_max   = output_strength_max
        self.parvo_weight          = parvo_weight
        self.magno_weight          = magno_weight

        self._grid = np.zeros((grid_res, grid_res), dtype=np.float32)

        self.health.update({
            "frames_procesados":  0,
            "puntos_entrada":     0,
            "bordes_barra":       0,   # diagonales \ emitidas
            "bordes_slash":       0,   # diagonales / emitidas
            "ratio_barra":        0.0,
            "ratio_slash":        0.0,
            "gradiente_max":      0.0,
            "gradiente_mean":     0.0,
            "local_loss":         1.0,
        })

        self._acc_entrada       = 0
        self._acc_barra         = 0
        self._acc_slash         = 0
        self._acc_loss          = 0.0
        self._last_log_time     = 0.0
        self._injected          = True

    # ------------------------------------------------------------------
    # MÉTODO PRINCIPAL
    # ------------------------------------------------------------------

    def process(self, activations: list) -> list:
        """
        Recibe el output del Transductor (misma señal que BordesV y BordesH).
        Devuelve activaciones de diagonales en barra_z y slash_z concatenadas.

        Entrada:
            activations — lista de (x, y, z, strength) del Transductor
                          z=30 (magno) o z=45 (parvo)

        Salida:
            lista de (x, y, barra_z, strength) ++ (x, y, slash_z, strength)
            Lista vacía si no hay diagonales detectables.
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

        # 2. Normalizar a [0,1] — misma lógica que BordesV
        grid_max = float(self._grid.max())
        if grid_max < 1e-6:
            self._maybe_log()
            return []

        grid_norm = self._grid / grid_max

        # 3. Calcular gradientes diagonales sobre la grilla normalizada
        grad_barra = self._compute_gradient_barra(grid_norm)
        grad_slash = self._compute_gradient_slash(grid_norm)

        # 4. Umbralizar y emitir activaciones para cada diagonal
        acts_barra = self._emit_edges(grad_barra, self.barra_z)
        acts_slash = self._emit_edges(grad_slash, self.slash_z)

        # 5. Métricas
        n_barra = len(acts_barra)
        n_slash = len(acts_slash)
        self._acc_barra += n_barra
        self._acc_slash += n_slash
        self.health["bordes_barra"] = n_barra
        self.health["bordes_slash"] = n_slash
        self.health["ratio_barra"]  = n_barra / max(1, n_entrada)
        self.health["ratio_slash"]  = n_slash / max(1, n_entrada)

        # gradiente combinado para métricas generales
        grad_combined = np.maximum(np.abs(grad_barra), np.abs(grad_slash))
        self.health["gradiente_max"]  = float(grad_combined.max())
        nonzero = grad_combined[grad_combined > 0]
        self.health["gradiente_mean"] = float(nonzero.mean()) if nonzero.size > 0 else 0.0

        # local_loss: rango aceptable [5%-30%] para cada diagonal independiente
        # (diagonales son menos densas que bordes V/H en letras como O, T, E)
        ratio_avg = (self.health["ratio_barra"] + self.health["ratio_slash"]) / 2.0
        ideal_lo, ideal_hi = 0.05, 0.30
        if ideal_lo <= ratio_avg <= ideal_hi:
            local_loss = 0.0
        elif ratio_avg < ideal_lo:
            local_loss = min(1.0, (ideal_lo - ratio_avg) / ideal_lo)
        else:
            local_loss = min(1.0, (ratio_avg - ideal_hi) / ideal_hi)

        self._acc_loss += local_loss
        self.health["local_loss"] = local_loss

        self._maybe_log()

        return acts_barra + acts_slash

    # ------------------------------------------------------------------
    # SUB-FUNCIÓN 1 — Construcción de la grilla
    # ------------------------------------------------------------------

    def _build_grid(self, activations: list) -> None:
        """Igual que en BordesV — acumula strength en (row, col) con pesos parvo/magno."""
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
    # SUB-FUNCIÓN 2 — Gradiente diagonal BARRA (tipo barra)
    # ------------------------------------------------------------------

    def _compute_gradient_barra(self, grid_norm: np.ndarray) -> np.ndarray:
        """
        Gradiente diagonal descendente (tipo barra): grid[i+1,j+1] - grid[i,j].

        Calcula la diferencia entre la grilla y su versión desplazada
        una celda en diagonal hacia abajo-derecha. El resultado se
        redimensiona con padding edge a (N, N) para mantener la resolución.

        Un valor alto indica un flanco que sube de izquierda a derecha
        cuando se mira la imagen de arriba a abajo → borde diagonal tipo barra.
        """
        N = self.grid_res
        # Desplazamiento: grid[1:, 1:] - grid[:-1, :-1] → shape (N-1, N-1)
        diff = grid_norm[1:, 1:] - grid_norm[:-1, :-1]   # (N-1, N-1)

        # Pad a (N, N) replicando bordes — conserva posición espacial
        result = np.zeros((N, N), dtype=np.float32)
        result[:-1, :-1] = diff
        result[-1,  :-1] = diff[-1, :]   # última fila: replica
        result[:-1, -1]  = diff[:, -1]   # última col: replica
        result[-1,  -1]  = diff[-1, -1]  # esquina
        return result

    # ------------------------------------------------------------------
    # SUB-FUNCIÓN 3 — Gradiente diagonal SLASH (/)
    # ------------------------------------------------------------------

    def _compute_gradient_slash(self, grid_norm: np.ndarray) -> np.ndarray:
        """
        Gradiente diagonal ascendente (/): grid[i+1,j-1] - grid[i,j].

        Diferencia entre la grilla y su versión desplazada una celda
        en diagonal hacia abajo-izquierda. El resultado se redimensiona
        con padding edge a (N, N).

        Un valor alto indica un flanco que sube de derecha a izquierda → borde /.
        """
        N = self.grid_res
        # Desplazamiento: grid[1:, :-1] - grid[:-1, 1:] → shape (N-1, N-1)
        diff = grid_norm[1:, :-1] - grid_norm[:-1, 1:]   # (N-1, N-1)

        result = np.zeros((N, N), dtype=np.float32)
        result[:-1, 1:] = diff
        result[-1,  1:] = diff[-1, :]
        result[:-1, 0]  = diff[:, 0]
        result[-1,  0]  = diff[-1, 0]
        return result

    # ------------------------------------------------------------------
    # SUB-FUNCIÓN 4 — Umbralización y emisión
    # ------------------------------------------------------------------

    def _emit_edges(self, gradient: np.ndarray, out_z: float) -> list:
        """
        Genera activaciones donde |gradiente| > edge_threshold.
        Igual que BordesV._emit_edges pero con out_z como parámetro.
        """
        abs_grad = np.abs(gradient)
        rows, cols = np.where(abs_grad > self.edge_threshold)

        if len(rows) == 0:
            return []

        result   = []
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
            result.append((x, y, out_z, strength))

        return result

    # ------------------------------------------------------------------
    # LOG Y HELPERS
    # ------------------------------------------------------------------

    def _maybe_log(self):
        _now = time.time()
        if (self._update_count % self.log_every == 0 or
                (_now - self._last_log_time) > 45.0):
            self._emit_log()
            self._last_log_time = _now
            self._acc_entrada   = 0
            self._acc_barra     = 0
            self._acc_slash     = 0
            self._acc_loss      = 0.0

    def _emit_log(self):
        h      = self.health
        frames = max(1, min(self._update_count, self.log_every))
        avg_entrada = self._acc_entrada / frames
        avg_barra   = self._acc_barra   / frames
        avg_slash   = self._acc_slash   / frames
        avg_loss    = self._acc_loss    / frames
        avg_ratio_b = avg_barra / max(1, avg_entrada)
        avg_ratio_s = avg_slash / max(1, avg_entrada)

        print(
            f"╲╱ [Diag] "
            f"frames={h['frames_procesados']} | "
            f"entrada={avg_entrada:.0f}pts | "
            f"barra={avg_barra:.0f} ({avg_ratio_b*100:.0f}%) | "
            f"slash={avg_slash:.0f} ({avg_ratio_s*100:.0f}%) | "
            f"grad_max={h['gradiente_max']:.3f} "
            f"grad_mean={h['gradiente_mean']:.3f} | "
            f"local_loss={avg_loss:.3f} | "
            f"barra_z={self.barra_z} slash_z={self.slash_z}",
            flush=True
        )

    # ------------------------------------------------------------------
    # REGISTRO EN LA RED
    # ------------------------------------------------------------------

    def register(self, net):
        net.modules[self.block_id] = self
        print(
            f"╲╱ [Diag] Registrado | "
            f"grid={self.grid_res}x{self.grid_res} | "
            f"threshold={self.edge_threshold} (normalizado [0,1]) | "
            f"barra_z={self.barra_z} slash_z={self.slash_z} | "
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
            "barra_z":               self.barra_z,
            "slash_z":               self.slash_z,
            "output_strength_scale": self.output_strength_scale,
            "output_strength_min":   self.output_strength_min,
            "output_strength_max":   self.output_strength_max,
            "parvo_weight":          self.parvo_weight,
            "magno_weight":          self.magno_weight,
        })
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "DiagonalDetectorBlock":
        b = cls(
            zone_z                = tuple(d.get("zone_z",                (100.0, 110.0))),
            grid_res              = d.get("grid_res",                    20),
            edge_threshold        = d.get("edge_threshold",              0.04),
            barra_z               = d.get("barra_z",                     74.0),
            slash_z               = d.get("slash_z",                     76.0),
            output_strength_scale = d.get("output_strength_scale",       8.0),
            output_strength_min   = d.get("output_strength_min",         3.0),
            output_strength_max   = d.get("output_strength_max",         8.0),
            parvo_weight          = d.get("parvo_weight",                1.0),
            magno_weight          = d.get("magno_weight",                0.3),
        )
        b.health    = d.get("health", b.health)
        b._injected = True
        return b
