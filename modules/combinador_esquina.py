# modules/combinador_esquina.py
#
# =====================================================================
# BLOQUE COMBINADOR DE ESQUINAS — Arquitectura de Bloques v0.5
# =====================================================================
#
# Cambios respecto a v0.4:
#   - Opción A: mínimo de bordes requerido antes de procesar.
#     Si acts_v < min_bordes_v o acts_h < min_bordes_h, devuelve []
#     directamente sin calcular nada. Los frames con poca señal son
#     transiciones entre estímulos — no tienen información útil para
#     detectar esquinas y solo generaban ⬆️ por amplificación de ruido
#     al normalizar grillas casi vacías.
#   - min_bordes_v=25, min_bordes_h=20 por defecto (derivados de los
#     logs: frames ✅ tenían bordes_v≥32 y bordes_h≥26; frames ⬆️
#     tenían bordes_v=22-23 y bordes_h=17-20).
#   - El log reporta cuántos frames fueron descartados por bajo conteo
#     (campo "frames_descartados" en health).
#   - Resto del código sin cambios.
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


class EsquinaBlock(BaseBlock):
    """
    Combinador de esquinas — Bloque 3 de la arquitectura.

    Detecta co-ocurrencia espacial de bordes verticales (z≈70) y
    horizontales (z≈72). Emite activaciones en output_z=80 donde
    ambas señales coinciden en la misma celda de la grilla normalizada.

    v0.5: mínimo de bordes requerido (min_bordes_v, min_bordes_h).
    Frames con poca señal se descartan antes de normalizar para evitar
    amplificación de ruido en grillas casi vacías.

    Parámetros configurables:
        zone_z            — zona lógica del bloque en el espacio 3D
        grid_res          — resolución de la grilla (debe coincidir con detectores)
        v_z               — z de la señal de bordes verticales (BordesV output)
        h_z               — z de la señal de bordes horizontales (BordesH output)
        z_tol             — tolerancia para filtrar por z (±z_tol)
        corner_threshold  — umbral sobre [0,1]
        smooth            — si True, aplica suavizado 3×3 antes del umbral
        output_z          — z donde se inyectan las esquinas detectadas
        output_strength_scale — escala: corner_signal * scale → strength
        output_strength_min / max — clip de fuerza de salida
        min_bordes_v      — mínimo de puntos de BordesV para procesar el frame
        min_bordes_h      — mínimo de puntos de BordesH para procesar el frame
        log_every         — frecuencia del log de diagnóstico
    """

    BLOCK_TYPE = "combinador_esquina"

    def __init__(
        self,
        zone_z: Tuple[float, float] = (92.0, 100.0),
        grid_res: int = 20,
        v_z: float = 70.0,
        h_z: float = 72.0,
        z_tol: float = 1.0,
        corner_threshold: float = 0.45,
        smooth: bool = False,
        output_z: float = 80.0,
        output_strength_scale: float = 8.0,
        output_strength_min: float = 3.0,
        output_strength_max: float = 8.0,
        min_bordes_v: int = 15,        # ← nuevo: mínimo de pts BordesV por frame
        min_bordes_h: int = 12,        # ← nuevo: mínimo de pts BordesH por frame
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
        self.z_tol                 = z_tol
        self.corner_threshold      = corner_threshold
        self.smooth                = smooth
        self.output_z              = output_z
        self.output_strength_scale = output_strength_scale
        self.output_strength_min   = output_strength_min
        self.output_strength_max   = output_strength_max
        self.min_bordes_v          = min_bordes_v
        self.min_bordes_h          = min_bordes_h

        self._v_grid = np.zeros((grid_res, grid_res), dtype=np.float32)
        self._h_grid = np.zeros((grid_res, grid_res), dtype=np.float32)

        self.health.update({
            "frames_procesados":   0,
            "frames_descartados":  0,   # ← nuevo: frames ignorados por bajo conteo
            "pts_bordes_v":        0,
            "pts_bordes_h":        0,
            "esquinas_emitidas":   0,
            "cs_mean":             0.0,
            "local_loss":          1.0,
        })

        self._acc_ok    = 0
        self._acc_over  = 0
        self._acc_under = 0

        self._last_bv      = 0
        self._last_bh      = 0
        self._acc_esquinas = 0
        self._acc_loss     = 0.0

        self._last_log_time = 0.0

        self._injected = True

    # ------------------------------------------------------------------
    # MÉTODO PRINCIPAL
    # ------------------------------------------------------------------

    def process(self, activations: list) -> list:
        """
        Recibe TODAS las activaciones del pipeline (z=30,45,70,72,...).
        Filtra por z, aplica mínimo de bordes, normaliza y detecta co-ocurrencia.

        Retorna lista vacía si:
          - Falta alguna de las dos señales de borde, o
          - Alguna señal tiene menos puntos que el mínimo requerido
            (frame de transición — señal insuficiente para detectar esquinas).
        """
        if not activations:
            return []

        self._update_count += 1

        # 1. Separar señales por z
        acts_v = [a for a in activations if abs(float(a[2]) - self.v_z) <= self.z_tol]
        acts_h = [a for a in activations if abs(float(a[2]) - self.h_z) <= self.z_tol]

        self.health["pts_bordes_v"] = len(acts_v)
        self.health["pts_bordes_h"] = len(acts_h)
        self._last_bv = len(acts_v)
        self._last_bh = len(acts_h)

        _now = time.time()

        # 2. Descartar si alguna señal está por debajo del mínimo
        #    Frames de transición tienen pocas activaciones → normalizar
        #    una grilla casi vacía amplifica ruido → falsos positivos.
        if len(acts_v) < self.min_bordes_v or len(acts_h) < self.min_bordes_h:
            self.health["frames_descartados"] += 1
            self._acc_loss += 1.0
            self.health["local_loss"] = 1.0
            if (self._update_count % self.log_every == 0 or
                    (_now - self._last_log_time) > 45.0):
                self._emit_log()
                self._last_log_time = _now
                self._acc_esquinas = 0
                self._acc_loss     = 0.0
            return []

        self.health["frames_procesados"] += 1

        # 3. Acumular en grillas
        self._build_grid(acts_v, self._v_grid)
        self._build_grid(acts_h, self._h_grid)

        # 4. Normalizar cada grilla a [0,1]
        v_max = float(self._v_grid.max())
        h_max = float(self._h_grid.max())

        if v_max < 1e-6 or h_max < 1e-6:
            return []

        v_norm = self._v_grid / v_max
        h_norm = self._h_grid / h_max

        # 5. Señal de esquina = mínimo normalizado
        corner_signal = np.minimum(v_norm, h_norm)

        if self.smooth:
            corner_signal = self._smooth_3x3(corner_signal)

        # 6. Umbralizar y emitir
        corner_acts = self._emit_corners(corner_signal)

        # 7. Métricas
        n_esquinas = len(corner_acts)
        self._acc_esquinas += n_esquinas
        self.health["esquinas_emitidas"] = n_esquinas

        active_cells = corner_signal[corner_signal > self.corner_threshold]
        self.health["cs_mean"] = float(active_cells.mean()) if len(active_cells) > 0 else 0.0

        avg_bordes = (len(acts_v) + len(acts_h)) / 2.0
        ratio      = n_esquinas / max(1, avg_bordes)
        ideal_lo, ideal_hi = 0.10, 0.25
        if ideal_lo <= ratio <= ideal_hi:
            local_loss = 0.0
            self._acc_ok += 1
        elif ratio < ideal_lo:
            local_loss = min(1.0, (ideal_lo - ratio) / ideal_lo)
            self._acc_under += 1
        else:
            local_loss = min(1.0, (ratio - ideal_hi) / ideal_hi)
            self._acc_over += 1
        self._acc_loss += local_loss
        self.health["local_loss"] = local_loss

        if (self._update_count % self.log_every == 0 or
                (_now - self._last_log_time) > 45.0):
            self._emit_log()
            self._last_log_time = _now
            self._acc_esquinas = 0
            self._acc_loss     = 0.0

        return corner_acts

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

    def _smooth_3x3(self, grid: np.ndarray) -> np.ndarray:
        padded = np.pad(grid, 1, mode='edge')
        result = np.zeros_like(grid)
        for dr in range(3):
            for dc in range(3):
                result += padded[dr:dr + self.grid_res, dc:dc + self.grid_res]
        return result / 9.0

    def _emit_corners(self, corner_signal: np.ndarray) -> list:
        rows, cols = np.where(corner_signal > self.corner_threshold)
        if len(rows) == 0:
            return []
        result   = []
        grid_res = self.grid_res
        for r, c in zip(rows, cols):
            strength = float(np.clip(
                float(corner_signal[r, c]) * self.output_strength_scale,
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
        h        = self.health
        frames_d = min(self._update_count, self.log_every) if self._update_count > 0 else 1
        avg_esq  = self._acc_esquinas / max(1, frames_d)
        avg_loss = self._acc_loss     / max(1, frames_d)

        bv = self._last_bv
        bh = self._last_bh
        avg_bordes = (bv + bh) / 2.0
        ratio = avg_esq / max(1, avg_bordes)

        total_rango = max(1, self._acc_ok + self._acc_over + self._acc_under)
        pct_ok      = 100 * self._acc_ok // total_rango
        rango_str   = f"✅{self._acc_ok} ⬆️{self._acc_over} ⬇️{self._acc_under} ({pct_ok}%ok)"
        rango_actual = "✅" if 0.10 <= ratio <= 0.25 else ("⬆️ " if ratio > 0.25 else "⬇️ ")

        print(
            f"🔲 [Esquinas] "
            f"frames={h['frames_procesados']} desc={h['frames_descartados']} | "
            f"bordes_v={bv}pts bordes_h={bh}pts | "
            f"esquinas={avg_esq:.0f} ({ratio*100:.0f}%) {rango_actual}| "
            f"cs_mean={h['cs_mean']:.2f} | "
            f"loss={avg_loss:.3f} [{rango_str}] | "
            f"output_z={self.output_z}",
            flush=True
        )

        self._acc_ok    = 0
        self._acc_over  = 0
        self._acc_under = 0

    # ------------------------------------------------------------------
    # REGISTRO Y SERIALIZACIÓN
    # ------------------------------------------------------------------

    def register(self, net):
        net.modules[self.block_id] = self
        print(
            f"🔲 [Esquinas] Registrado | "
            f"grid={self.grid_res}x{self.grid_res} | "
            f"v_z={self.v_z} h_z={self.h_z} | "
            f"threshold={self.corner_threshold} | "
            f"min_bordes v={self.min_bordes_v} h={self.min_bordes_h} | "
            f"smooth={self.smooth} | "
            f"log_every={self.log_every} | "
            f"output_z={self.output_z}",
            flush=True
        )

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "grid_res":              self.grid_res,
            "v_z":                   self.v_z,
            "h_z":                   self.h_z,
            "z_tol":                 self.z_tol,
            "corner_threshold":      self.corner_threshold,
            "smooth":                self.smooth,
            "output_z":              self.output_z,
            "output_strength_scale": self.output_strength_scale,
            "output_strength_min":   self.output_strength_min,
            "output_strength_max":   self.output_strength_max,
            "min_bordes_v":          self.min_bordes_v,
            "min_bordes_h":          self.min_bordes_h,
        })
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "EsquinaBlock":
        b = cls(
            zone_z                = tuple(d.get("zone_z",           (92.0, 100.0))),
            grid_res              = d.get("grid_res",               20),
            v_z                   = d.get("v_z",                    70.0),
            h_z                   = d.get("h_z",                    72.0),
            z_tol                 = d.get("z_tol",                  1.0),
            corner_threshold      = d.get("corner_threshold",       0.45),
            smooth                = d.get("smooth",                 False),
            output_z              = d.get("output_z",               80.0),
            output_strength_scale = d.get("output_strength_scale",  8.0),
            output_strength_min   = d.get("output_strength_min",    3.0),
            output_strength_max   = d.get("output_strength_max",    8.0),
            min_bordes_v          = d.get("min_bordes_v",           15),
            min_bordes_h          = d.get("min_bordes_h",           12),
        )
        b.health    = d.get("health", b.health)
        b._injected = True
        return b