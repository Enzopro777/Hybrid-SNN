# modules/detector_letras.py
#
# =====================================================================
# BLOQUE DETECTOR DE LETRAS — Arquitectura de Bloques v0.2
# =====================================================================
#
# Cambios respecto a v0.1:
#
#   FRENTE 1 — Falso positivo X/A
#   ----------------------------------
#   La confusión X≈A surgía porque ambas comparten diagonales fuertes y
#   simetría. Solución: añadir el feature "junction_density" (densidad de
#   cruces en z=86) que separa la X (cruces centrales en un nodo) de la A
#   (sin cruce central — el ápice es un vértice, no un cruce). Además se
#   añade "bh_bottom_ratio" (bordesH en mitad inferior): la A tiene el
#   travesaño EN el tercio medio-bajo, la X no tiene bordesH significativos
#   en ninguna zona. Con estos dos discriminadores nuevos se eleva la
#   penalización de A en la fórmula de X y viceversa.
#
#   FRENTE 2 — Evidencia nunca dispara por propagación sináptica
#   ------------------------------------------------------------------
#   Se añade auto_wire_evidence_ports(): conecta los puertos de evidencia
#   directamente a las neuronas-relé de la grilla sensorial (que sí reciben
#   spikes del Transductor). Esto crea un camino de propagación directo
#   Transductor → relé → evidencia_X sin depender de que existan conexiones
#   sinápticas largas en la red aleatoria.
#
#   FRENTE 3 — CausalProbe: mismos timestamps para todas las letras
#   ------------------------------------------------------------------
#   El problema era que _inject_evidence() disparaba neuronas de evidencia
#   de la letra top-1 con la misma base de tiempo que el bloque las procesó,
#   haciendo que todas las letras empezaran a acumular evidencia al mismo t.
#   Solución: solo la letra ganadora recibe el spike; todas las demás quedan
#   en 0 hasta que genuinamente ganen un frame. Esto ya estaba en la lógica,
#   pero el Probe observaba port.value calculado desde update_evidence_values
#   que promediaba spikes recientes de neuronas que podían estar activas por
#   otras razones. No se cambia el Probe; se corrige la fuente: ahora solo
#   disparamos spikes de la letra ganadora, y el rate se controla mejor con
#   un pulse_interval más fino (100ms en vez de 200ms).
# =====================================================================

import time
import numpy as np
import logging
from typing import List, Optional

try:
    from modules.monitor_block import BaseBlock
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from modules.monitor_block import BaseBlock


class LetraDetectorBlock(BaseBlock):
    """
    Detector de letras — Bloque 5 de la arquitectura v0.2.

    Cambios v0.2:
    - Features nuevos: junction_density, bh_bottom_ratio
    - Fórmulas de score revisadas para separar mejor X de A
    - auto_wire_evidence_ports(): conecta puertos de evidencia a relés de la grilla
    - pulse_interval reducido a 100ms para mejor resolución causal
    """

    BLOCK_TYPE = "detector_letras"

    def __init__(
        self,
        zone_z: tuple = (55.0, 65.0),
        grid_res: int = 20,
        v_z: float = 70.0,
        h_z: float = 72.0,
        barra_z: float = 74.0,
        slash_z: float = 76.0,
        c_z: float = 80.0,
        j_z: float = 86.0,          # z de junctions (nuevo)
        curv_z: float = 78.0,       # v0.3: z de curvas (CurvaDetector)
        z_tol: float = 1.0,
        letra_threshold: float = 0.55,
        inject_strength: float = 6.0,
        min_pts_bv: int = 8,
        min_pts_bh: int = 6,
        log_every: int = 25,
    ):
        super().__init__(
            block_id   = self.BLOCK_TYPE,
            zone_z     = zone_z,
            n_immortal = 0,
            log_every  = log_every,
        )

        self.grid_res         = grid_res
        self.v_z              = v_z
        self.h_z              = h_z
        self.barra_z          = barra_z
        self.slash_z          = slash_z
        self.c_z              = c_z
        self.j_z              = j_z
        self.curv_z           = curv_z      # v0.3
        self.z_tol            = z_tol
        self.letra_threshold  = letra_threshold
        self.inject_strength  = inject_strength
        self.min_pts_bv       = min_pts_bv
        self.min_pts_bh       = min_pts_bh

        self._v_grid     = np.zeros((grid_res, grid_res), dtype=np.float32)
        self._h_grid     = np.zeros((grid_res, grid_res), dtype=np.float32)
        self._barra_grid = np.zeros((grid_res, grid_res), dtype=np.float32)
        self._slash_grid = np.zeros((grid_res, grid_res), dtype=np.float32)
        self._c_grid     = np.zeros((grid_res, grid_res), dtype=np.float32)
        self._j_grid     = np.zeros((grid_res, grid_res), dtype=np.float32)  # junctions
        self._curv_grid  = np.zeros((grid_res, grid_res), dtype=np.float32)  # v0.3 curvas
        # v1.6.3: mapas estructurales derivados, no nuevos sensores.
        self._center_junction_grid = np.zeros((grid_res, grid_res), dtype=np.float32)
        self._closure_grid = np.zeros((grid_res, grid_res), dtype=np.float32)
        self._top_bar_grid = np.zeros((grid_res, grid_res), dtype=np.float32)
        self._mid_bar_grid = np.zeros((grid_res, grid_res), dtype=np.float32)
        self._bottom_bar_grid = np.zeros((grid_res, grid_res), dtype=np.float32)
        self._center_vertical_grid = np.zeros((grid_res, grid_res), dtype=np.float32)
        self._left_vertical_grid = np.zeros((grid_res, grid_res), dtype=np.float32)
        self._diag_cross_grid = np.zeros((grid_res, grid_res), dtype=np.float32)
        self._apex_grid = np.zeros((grid_res, grid_res), dtype=np.float32)
        self._closure_ring_grid = np.zeros((grid_res, grid_res), dtype=np.float32)

        # Referencia al LetterIOModule — se conecta en simulation.py
        self._lang_module = None

        self.health.update({
            "frames_procesados":  0,
            "frames_descartados": 0,
            "detecciones":        {sym: 0 for sym in ["X", "O", "T", "A", "E"]},
            "scores_ultimo":      {sym: 0.0 for sym in ["X", "O", "T", "A", "E"]},
            "local_loss":         1.0,
            # 1.6.22: auditoría explícita del sesgo de X. No alimenta el readout.
            "x_signal_audit": {"x": 0.0, "others_mean": 0.0, "excess": 0.0, "margin": 0.0, "rank": None, "alarm": False},
            "x_bias_alarm_count": 0,
        })

        self._acc_loss    = 0.0
        self._acc_det     = {sym: 0 for sym in ["X", "O", "T", "A", "E"]}
        self._last_log_time = 0.0
        # OBSERVADOR: nunca inyecta evidencia; la clasificación heurística es telemetría.
        self._injected = False
        self.observer_only = True
        self.last_prediction = None

        # Ground truth — cruce con autoplay
        self._gt_total    = 0
        self._gt_correct  = 0
        self._gt_false_pos = 0
        self._gt_miss      = 0

        # Pulso de evidencia: 100ms (más fino que 200ms anterior)
        self.evidence_pulse_interval_ms = 100.0
        self._last_evidence_pulse_t = {sym: -1e12 for sym in ["X", "O", "T", "A", "E"]}

    # ------------------------------------------------------------------
    # FRENTE 2: conectar puertos de evidencia a neuronas relé de la grilla
    # ------------------------------------------------------------------

    def auto_wire_evidence_ports(self, net) -> None:
        """Mantiene evidencia_* como puertos observacionales, sin cableado neural.

        El detector heurístico describe la entrada para diagnóstico; el aprendizaje
        de letras debe depender exclusivamente del FeatureBus → readout neuronal.
        """
        self._evidence_ports_neural_wiring_disabled = True
        if self._lang_module is None:
            return
        for sym in getattr(self._lang_module, 'symbols', []):
            port = self._lang_module.input_ports.get(f"evidencia_{sym}")
            if port is not None:
                port.connected_neurons = []


    def process(self, activations: list, net=None, t: float = 0.0) -> list:
        """
        Recibe activaciones del frame completo. Calcula scores por letra.
        Inyecta spikes SOLO en la letra top-1 cuando supera el umbral.
        Devuelve lista vacía (no agrega al pipeline).
        """
        if not activations:
            return []

        self._update_count += 1
        _now = time.time()

        # 1. Separar señales por z
        acts_v     = [a for a in activations if abs(float(a[2]) - self.v_z)     <= self.z_tol]
        acts_h     = [a for a in activations if abs(float(a[2]) - self.h_z)     <= self.z_tol]
        acts_barra = [a for a in activations if abs(float(a[2]) - self.barra_z) <= self.z_tol]
        acts_slash = [a for a in activations if abs(float(a[2]) - self.slash_z) <= self.z_tol]
        acts_c     = [a for a in activations if abs(float(a[2]) - self.c_z)     <= self.z_tol]
        acts_j     = [a for a in activations if abs(float(a[2]) - self.j_z)     <= self.z_tol]
        acts_curv  = [a for a in activations if abs(float(a[2]) - self.curv_z) <= self.z_tol]  # v0.3

        # FIX v1.5.9: diagnóstico de curv_z — avisa si nunca llegan curvas
        if not hasattr(self, '_curv_warn_count'):
            self._curv_warn_count = 0
            self._curv_seen_once = False
        if acts_curv:
            self._curv_seen_once = True
        elif not self._curv_seen_once:
            self._curv_warn_count += 1
            if self._curv_warn_count == 50:
                logging.warning(
                    f"[LetraDetector] 50 frames sin activaciones en curv_z={self.curv_z:.1f} "
                    f"(z_tol={self.z_tol}). Verificar que CurvaDetector.output_z={self.curv_z:.1f}. "
                    f"Total activaciones en frame: {len(activations)}, "
                    f"zs presentes: {sorted(set(round(float(a[2]),1) for a in activations[:50]))}"
                )

        # 2. Descartar frames con señal insuficiente
        if len(acts_v) < self.min_pts_bv or len(acts_h) < self.min_pts_bh:
            self.health["frames_descartados"] += 1
            self._acc_loss += 1.0
            if (self._update_count % self.log_every == 0 or
                    (_now - self._last_log_time) > 45.0):
                self._emit_log()
                self._last_log_time = _now
                self._acc_loss = 0.0
                self._acc_det  = {sym: 0 for sym in self._acc_det}
            return []

        self.health["frames_procesados"] += 1

        # 3. Construir grillas
        self._build_grid(acts_v,     self._v_grid)
        self._build_grid(acts_h,     self._h_grid)
        self._build_grid(acts_barra, self._barra_grid)
        self._build_grid(acts_slash, self._slash_grid)
        self._build_grid(acts_c,     self._c_grid)
        self._build_grid(acts_j,     self._j_grid)
        self._build_grid(acts_curv,  self._curv_grid)  # v0.3

        # 4.1 Combinadores estructurales sobre detectores simples.
        self._build_structural_maps()
        # 4. Calcular features
        features = self._compute_features()

        # v1.6: representación tipada para el readout.
        if self._lang_module is not None and net is not None:
            try:
                self._lang_module.emit_feature_maps(net, {
                    "vertical": self._v_grid, "horizontal": self._h_grid,
                    "diag_back": self._barra_grid, "diag_slash": self._slash_grid,
                    "curve": self._curv_grid, "corner": self._c_grid,
                    "junction": self._j_grid,
                    "occupancy": np.maximum.reduce([self._v_grid, self._h_grid, self._barra_grid, self._slash_grid, self._curv_grid, self._c_grid, self._j_grid]),
                    "center_junction": self._center_junction_grid,
                    "closure": self._closure_grid,
                    "top_bar": self._top_bar_grid,
                    "mid_bar": self._mid_bar_grid,
                    "bottom_bar": self._bottom_bar_grid,
                    "center_vertical": self._center_vertical_grid,
                    "left_vertical": self._left_vertical_grid,
                    "diag_cross": self._diag_cross_grid,
                    "apex": self._apex_grid,
                    "closure_ring": self._closure_ring_grid,
                }, t=t)
            except Exception as _bus_exc:
                logging.debug(f"[FeatureBus] emisión omitida: {_bus_exc}")

        # 5. Score por letra
        scores     = self._score_letters(features)
        detectadas = [sym for sym, sc in scores.items() if sc >= self.letra_threshold]

        self.health["scores_ultimo"] = {sym: round(sc, 3) for sym, sc in scores.items()}
        # 1.6.22: medir la presión anómala de X frente al resto sin tocar el clasificador neuronal.
        _x_score = float(scores.get("X", 0.0))
        _other_scores = [float(v) for k, v in scores.items() if k != "X"]
        _others_mean = float(np.mean(_other_scores)) if _other_scores else 0.0
        _x_excess = _x_score - _others_mean
        _ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        _x_rank = next((i + 1 for i, (k, _) in enumerate(_ordered) if k == "X"), None)
        _x_alarm = bool(_x_rank == 1 and _x_excess > 0.18)
        if _x_alarm:
            self.health["x_bias_alarm_count"] = int(self.health.get("x_bias_alarm_count", 0)) + 1
        self.health["x_signal_audit"] = {
            "x": round(_x_score, 4),
            "others_mean": round(_others_mean, 4),
            "excess": round(_x_excess, 4),
            "margin": round(_x_excess, 4),
            "rank": _x_rank,
            "alarm": _x_alarm,
        }
        # FIX v1.5.9: registrar curv_per_bv en health para diagnóstico via causal_probe
        self.health["curv_per_bv"] = round(float(features.get("curv_per_bv", 0.0)), 4)
        self.health["curv_grid_max"] = round(float(self._curv_grid.max()), 4)
        self.health["structure_features"] = {k: round(float(features.get(k, 0.0)), 4) for k in ("v_left_ratio", "v_center_ratio", "h_top_ratio", "h_mid_ratio", "h_bottom_ratio", "curve_density", "center_occupancy", "junction_density", "junction_center_ratio", "diag_simetria", "center_void", "closure_ring_density")}

        # 6. Clasificación heurística + escritura directa de port.value
        # v1.9: los puertos evidencia_* son observacionales. Se conserva su valor
        # para telemetría del detector, pero nunca se usa como entrada neural del readout.
        top_sym = max(detectadas, key=lambda s: scores[s]) if detectadas else None
        sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_margin = (float(sorted_scores[0][1]) - float(sorted_scores[1][1])) if len(sorted_scores) > 1 else (float(sorted_scores[0][1]) if sorted_scores else 0.0)
        self.health["top_margin"] = round(top_margin, 4)
        self.health["top_symbol"] = top_sym
        self.last_prediction = top_sym
        if top_sym is not None:
            self.health["detecciones"][top_sym] += 1
            self._acc_det[top_sym] += 1

        # v1.6.9: el score heurístico vive en un canal separado.
        # evidencia_* queda reservado para observabilidad neural y no debe
        # convertirse accidentalmente en un clasificador hard-coded.
        if self._lang_module is not None and t > 1.0:
            try:
                self._lang_module.update_detector_evidence(scores, t=t)
            except Exception as _det_ev_exc:
                logging.debug(f"[DetectorEvidence] actualización omitida: {_det_ev_exc}")

        # 7. local_loss
        n_det = len(detectadas)
        if n_det == 1:
            local_loss = 0.0
        elif n_det == 0:
            local_loss = 0.5
        else:
            local_loss = min(1.0, (n_det - 1) * 0.4)

        self._acc_loss += local_loss
        self.health["local_loss"] = local_loss

        if (self._update_count % self.log_every == 0 or
                (_now - self._last_log_time) > 45.0):
            self._emit_log()
            self._last_log_time = _now
            self._acc_loss = 0.0
            self._acc_det  = {sym: 0 for sym in self._acc_det}

        return []

    # ------------------------------------------------------------------
    # FEATURES DISCRIMINANTES (v0.2 — añade junction_density y bh_bottom)
    # ------------------------------------------------------------------

    def _build_structural_maps(self) -> None:
        """Combina detectores simples en señales estructurales de baja dimensión."""
        G = self.grid_res
        yy, xx = np.mgrid[0:G, 0:G]
        cx0, cx1 = G * 0.33, G * 0.67
        central = ((xx >= cx0) & (xx < cx1) & (yy >= cx0) & (yy < cx1)).astype(np.float32)
        top = (yy < G/3).astype(np.float32)
        mid = ((yy >= G/3) & (yy < 2*G/3)).astype(np.float32)
        bottom = (yy >= 2*G/3).astype(np.float32)
        center_x = ((xx >= G/3) & (xx < 2*G/3)).astype(np.float32)
        left_x = (xx < G/3).astype(np.float32)
        rim = ((xx < G*0.25) | (xx >= G*0.75) | (yy < G*0.25) | (yy >= G*0.75)).astype(np.float32)
        center = np.maximum.reduce([self._v_grid, self._h_grid, self._barra_grid, self._slash_grid, self._curv_grid, self._c_grid, self._j_grid])
        scale = max(float(center.max()), 1e-6)
        occ = center / scale
        j = self._j_grid / max(float(self._j_grid.max()), 1e-6)
        c = self._curv_grid / max(float(self._curv_grid.max()), 1e-6)
        db = self._barra_grid / max(float(self._barra_grid.max()), 1e-6)
        ds = self._slash_grid / max(float(self._slash_grid.max()), 1e-6)
        bv = self._v_grid / max(float(self._v_grid.max()), 1e-6)
        bh = self._h_grid / max(float(self._h_grid.max()), 1e-6)
        # Cruce de diagonales: ambas orientaciones coinciden espacialmente.
        self._diag_cross_grid[:] = np.minimum(db, ds)
        self._center_junction_grid[:] = j * central
        # Cierre proxy: curvatura/contorno fuerte en el perímetro y baja ocupación central.
        center_void = np.clip(1.0 - occ, 0.0, 1.0)
        self._closure_grid[:] = c * rim * (0.6 + 0.4 * center_void)
        # Cierre más estricto: continuidad de curva en el anillo + vacío central.
        inner = ((xx >= G*0.25) & (xx < G*0.75) & (yy >= G*0.25) & (yy < G*0.75)).astype(np.float32)
        ring = ((xx >= G*0.20) & (xx < G*0.80) & (yy >= G*0.20) & (yy < G*0.80) & (inner == 0)).astype(np.float32)
        ring_curve = c * ring
        self._closure_ring_grid[:] = ring_curve * (0.7 + 0.3 * center_void)
        self._top_bar_grid[:] = bh * top
        self._mid_bar_grid[:] = bh * mid
        self._bottom_bar_grid[:] = bh * bottom
        self._center_vertical_grid[:] = bv * center_x
        self._left_vertical_grid[:] = bv * left_x
        self._apex_grid[:] = np.maximum(db, ds) * top * (0.7 + 0.3 * (1.0 - j * central))

    def _compute_features(self) -> dict:
        """
        Calcula los features discriminantes a partir de las grillas.
        v0.2 añade junction_density y bh_bottom_ratio para separar X de A.
        """
        G = self.grid_res

        # Grillas normalizadas
        bv    = self._v_grid     / max(self._v_grid.max(),     1e-6)
        bh    = self._h_grid     / max(self._h_grid.max(),     1e-6)
        diag_b = self._barra_grid / max(self._barra_grid.max(), 1e-6)
        diag_s = self._slash_grid / max(self._slash_grid.max(), 1e-6)
        esq   = self._c_grid     / max(self._c_grid.max(),     1e-6)
        junc  = self._j_grid     / max(self._j_grid.max(),     1e-6)

        # Versiones binarias
        bv_b   = (bv    > 0.1).astype(np.float32)
        bh_b   = (bh    > 0.1).astype(np.float32)
        db_b   = (diag_b > 0.1).astype(np.float32)
        ds_b   = (diag_s > 0.1).astype(np.float32)
        esq_b  = (esq   > 0.1).astype(np.float32)
        junc_b = (junc  > 0.1).astype(np.float32)

        bv_total   = bv_b.sum()  + 1e-6
        bh_total   = bh_b.sum()  + 1e-6
        esq_total  = esq_b.sum() + 1e-6
        junc_total = junc_b.sum()

        return {
            # ── Features v0.1 ────────────────────────────────────────────
            "esq_per_bv":          esq_total / bv_total,
            "esq_top_ratio":       esq_b[:G//3, :].sum() / esq_total,
            "esq_mid_ratio":       esq_b[G//3:2*G//3, :].sum() / esq_total,
            "bh_top_ratio":        bh_b[:3, :].sum() / bh_total,
            "bv_center_ratio":     bv_b[:, 7:13].sum() / bv_total,
            "bv_left_ratio":       bv_b[:, :5].sum() / bv_total,
            "bh_mid":              bh_b[6:11, :].sum(),
            "bh_top3":             bh_b[:3, :].sum(),
            "esq_topcenter":       esq_b[:6, 5:15].sum(),
            "diag_barra_per_bv":   db_b.sum() / bv_total,
            "diag_slash_per_bv":   ds_b.sum() / bv_total,
            "diag_simetria":       1.0 - abs(db_b.sum() - ds_b.sum()) / (db_b.sum() + ds_b.sum() + 1e-6),
            "diag_barra_bottom":   db_b[G//2:, :].sum() / (db_b.sum() + 1e-6),

            # ── Features v0.2 (nuevos) ───────────────────────────────────

            # Densidad de junctions en la zona central (filas 7-13, cols 7-13)
            # X: cruce central pronunciado (~0.6 del total)
            # A: ápice arriba, sin cruce central real (~0.15)
            "junction_center_ratio": (
                junc_b[G//3:2*G//3, G//3:2*G//3].sum() / (junc_total + 1e-6)
            ),

            # BordesH en tercio inferior (filas 14-20)
            # X: casi sin bordesH abajo (~0.05)
            # A: sin bordesH abajo tampoco (~0.05) — no discrimina aquí
            # E: bordesH distribuidos incluyendo abajo (~0.35)
            "bh_bottom_ratio":     bh_b[2*G//3:, :].sum() / bh_total,

            # Presencia de junctions por encima del centro (filas 0-7)
            # A: tiene vértice arriba con un junction puntual (~0.3)
            # X: sin junctions en la parte alta (~0.1)
            "junction_top_ratio":  (
                junc_b[:G//3, :].sum() / (junc_total + 1e-6)
            ),

            # Ratio absoluto de junctions — X tiene muchos cruces
            # X: alto (~0.3 del grid), A: bajo (~0.08), O: muy bajo
            "junction_density":    junc_total / (G * G),

            # ── Feature v0.3 — discriminante O vs X ─────────────────────
            # Medido en datos reales: O≈0.77, X≈0.49 (ratio 1.57x)
            # La O es una curva pura → alto curv_per_bv
            # La X son diagonales rectas → bajo curv_per_bv
            "curv_per_bv": (
                (self._curv_grid / max(float(self._curv_grid.max()), 1e-6) > 0.1
                 ).astype("float32").sum() / bv_total
                if hasattr(self, "_curv_grid") and float(self._curv_grid.max()) > 1e-6
                else 0.0
            ),
            "v_left_ratio": float(bv[:, :G//3].sum() / (bv.sum() + 1e-6)),
            "v_center_ratio": float(bv[:, G//3:2*G//3].sum() / (bv.sum() + 1e-6)),
            "h_top_ratio": float(bh[:G//3, :].sum() / (bh.sum() + 1e-6)),
            "h_mid_ratio": float(bh[G//3:2*G//3, :].sum() / (bh.sum() + 1e-6)),
            "h_bottom_ratio": float(bh[2*G//3:, :].sum() / (bh.sum() + 1e-6)),
            "curve_density": float((self._curv_grid > 0.1 * max(float(self._curv_grid.max()), 1e-6)).mean()),
            "center_occupancy": float(np.maximum.reduce([bv, bh, diag_b, diag_s, esq, junc])[G//3:2*G//3, G//3:2*G//3].mean()),
            "center_void": float(np.clip(1.0 - np.maximum.reduce([bv, bh, diag_b, diag_s, esq, junc])[G//3:2*G//3, G//3:2*G//3].mean(), 0.0, 1.0)),
            "closure_ring_density": float((self._closure_ring_grid > 0.12 * max(float(self._closure_ring_grid.max()), 1e-6)).mean()),
        }

    def _score_letters(self, f: dict) -> dict:
        """
        Score en [0,1] para cada letra.
        v0.3: discriminación O/X mejorada con curv_per_bv como feature clave.
        """
        scores = {}

        # --- X ---
        # Diagonales simétricas + esquinas centrales + cruce central de junctions
        # Penalización si hay junction en top (sería la A), sin cruce central,
        # o muchas curvas (señal de O).
        c1 = min(1.0, f["esq_per_bv"]           / 0.45)    # esquinas/bordesV alto
        c2 = min(1.0, f["esq_mid_ratio"]         / 0.38)    # esquinas en zona media
        c3 = 1.0 - min(1.0, f["bh_top_ratio"]    / 0.50)   # T tiene bh_top alto → penalizar
        c4 = min(1.0, f["diag_barra_per_bv"]     / 0.40)   # diagonal \ presente
        c5 = min(1.0, f["diag_slash_per_bv"]     / 0.40)   # diagonal / presente
        c6 = f["diag_simetria"]                              # ambas diagonales simétricas
        c7 = min(1.0, f["junction_center_ratio"] / 0.40)   # cruce central X
        c8 = 1.0 - min(1.0, f["junction_top_ratio"] / 0.25)
        c9_not_O = 1.0 - min(1.0, f["esq_per_bv"] / 0.60)
        # FIX v0.3: penalizar X si muchas curvas (señal de O — medido O≈0.77, X≈0.49)
        # FIX v1.5.9: threshold recalibrado a 0.63 (punto medio O=0.77, X=0.49)
        # Con 0.55: X con curv=0.49 tenía c10=0.11 (débil). Con 0.63: c10=0.22 (2x mejor separación).
        c10_not_O_curv = 1.0 - min(1.0, f.get("curv_per_bv", 0.0) / 0.63)
        scores["X"] = (c1 * 0.12 + c2 * 0.10 + c3 * 0.06
                     + c4 * 0.11 + c5 * 0.11 + c6 * 0.07
                     + c7 * 0.12 + c8 * 0.09 + c9_not_O * 0.10
                     + c10_not_O_curv * 0.12)
        self.health["x_component_audit"] = {
            "esquinas_bordes": round(float(c1), 4),
            "esquinas_media": round(float(c2), 4),
            "anti_T": round(float(c3), 4),
            "diag_back": round(float(c4), 4),
            "diag_slash": round(float(c5), 4),
            "diag_simetria": round(float(c6), 4),
            "junction_centro": round(float(c7), 4),
            "anti_A": round(float(c8), 4),
            "anti_O_esquinas": round(float(c9_not_O), 4),
            "anti_O_curvatura": round(float(c10_not_O_curv), 4),
        }

        # --- O ---
        # v1.7: O debe representar una estructura cerrada, no sólo curvatura.
        # Se combinan curva + cierre de anillo + vacío central + ausencia de cruce.
        c1 = min(1.0, f.get("curv_per_bv", 0.0) / 0.63)
        c2 = min(1.0, f.get("closure_ring_density", 0.0) / 0.18)
        c3 = f.get("center_void", 0.0)
        c4 = 1.0 - min(1.0, f["junction_center_ratio"] / 0.30)
        c5 = 1.0 - min(1.0, f["diag_barra_per_bv"] / 0.42)
        c6 = min(1.0, f["diag_simetria"] / 0.95)
        scores["O"] = (c1 * 0.28 + c2 * 0.24 + c3 * 0.18
                     + c4 * 0.14 + c5 * 0.10 + c6 * 0.06)
        # --- T ---
        # BordesH concentrados arriba + bordesV en centro + SIN diagonales
        c1 = min(1.0, f["bh_top_ratio"]               / 0.75)
        c2 = min(1.0, f["bv_center_ratio"]            / 0.60)
        c3 = 1.0 - min(1.0, f["bv_left_ratio"]        / 0.50)
        c4 = 1.0 - min(1.0, f["esq_per_bv"]           / 0.45)
        c5 = 1.0 - min(1.0, f["diag_barra_per_bv"]    / 0.20)
        c6 = 1.0 - min(1.0, f["diag_slash_per_bv"]    / 0.20)
        scores["T"] = (c1 * 0.28 + c2 * 0.28 + c3 * 0.12
                     + c4 * 0.12 + c5 * 0.10 + c6 * 0.10)

        # --- A ---
        # Diagonales + travesaño medio + junction en vértice ARRIBA (no centro)
        # NUEVO v0.2: penalizar si junction_center_ratio alto (sería X)
        c1 = min(1.0, f["esq_topcenter"]              / 12.0)
        c2 = min(1.0, f["bh_mid"]                     / 30.0)
        c3 = 1.0 - min(1.0, f["bh_top3"]              / 15.0)
        c4 = min(1.0, f["diag_barra_per_bv"]          / 0.30)
        c5 = min(1.0, f["diag_slash_per_bv"]          / 0.30)
        c6 = min(1.0, f["diag_barra_bottom"]          / 0.60)
        # NUEVO v0.2: junction en top (ápice) y NO en centro
        c7 = min(1.0, f["junction_top_ratio"]         / 0.25)  # tiene ápice
        c8 = 1.0 - min(1.0, f["junction_center_ratio"] / 0.40)  # no tiene cruce X
        scores["A"] = (c1 * 0.18 + c2 * 0.18 + c3 * 0.12
                     + c4 * 0.12 + c5 * 0.12 + c6 * 0.08
                     + c7 * 0.10 + c8 * 0.10)

        # --- E ---
        # BordesV muy a la izquierda + travesaño medio + SIN diagonales
        c1 = min(1.0, f["bv_left_ratio"]              / 0.65)
        c2 = min(1.0, f["bh_mid"]                     / 35.0)
        c3 = 1.0 - min(1.0, f["bv_center_ratio"]      / 0.40)
        c4 = 1.0 - min(1.0, f["diag_barra_per_bv"]    / 0.20)
        c5 = 1.0 - min(1.0, f["diag_slash_per_bv"]    / 0.20)
        c6 = min(1.0, f["bh_bottom_ratio"]            / 0.30)  # barras abajo de la E
        scores["E"] = (c1 * 0.28 + c2 * 0.24 + c3 * 0.14
                     + c4 * 0.12 + c5 * 0.11 + c6 * 0.11)

        return scores

    # ------------------------------------------------------------------
    # INYECCIÓN DE EVIDENCIA
    # ------------------------------------------------------------------

    def _inject_evidence(self, net, sym: str, t: float):
        """Compatibilidad histórica: deliberadamente no inyecta neuronas.

        v1.9 elimina este camino para preservar la separación observador/readout.
        """
        return 0

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

    # ------------------------------------------------------------------
    # LOG DE DIAGNÓSTICO
    # ------------------------------------------------------------------

    def _emit_log(self):
        h      = self.health
        frames = max(1, min(self._update_count, self.log_every))
        avg_loss = self._acc_loss / frames

        curv_max = float(self._curv_grid.max()) if hasattr(self, '_curv_grid') else -1.0
        scores_str = " | ".join(
            f"{sym}:{sc:.2f}{'✅' if sc >= self.letra_threshold else ''}"
            for sym, sc in h["scores_ultimo"].items()
        )
        det_str = " ".join(
            f"{sym}×{self._acc_det[sym]}"
            for sym in self._acc_det if self._acc_det[sym] > 0
        ) or "ninguna"
        xa = h.get("x_signal_audit", {}) or {}

        print(
            f"🔤 [Letras] "
            f"frames={h['frames_procesados']} desc={h['frames_descartados']} | "
            f"det={det_str} | "
            f"scores=[{scores_str}] | "
            f"curv_grid_max={curv_max:.4f} | "
            f"X-audit=raw:{xa.get('x', 0.0):.2f} vsμ:{xa.get('others_mean', 0.0):.2f} "
            f"Δ:{xa.get('excess', 0.0):.2f} rank:{xa.get('rank', '-')}{'⚠️' if xa.get('alarm') else ''} | "
            f"loss={avg_loss:.3f}",
            flush=True
        )

    # ------------------------------------------------------------------
    # REGISTRO Y SERIALIZACIÓN
    # ------------------------------------------------------------------

    def register(self, net):
        net.modules[self.block_id] = self
        print(
            f"🔤 [Letras] Registrado v0.2 | "
            f"threshold={self.letra_threshold} | "
            f"inject_strength={self.inject_strength} | "
            f"pulse_interval_ms={self.evidence_pulse_interval_ms:.0f} | "
            f"min_pts bv={self.min_pts_bv} bh={self.min_pts_bh} | "
            f"log_every={self.log_every}",
            flush=True
        )

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "grid_res":         self.grid_res,
            "v_z":              self.v_z,
            "h_z":              self.h_z,
            "barra_z":          self.barra_z,
            "slash_z":          self.slash_z,
            "c_z":              self.c_z,
            "j_z":              self.j_z,
            "curv_z":           self.curv_z,
            "z_tol":            self.z_tol,
            "letra_threshold":  self.letra_threshold,
            "inject_strength":  self.inject_strength,
            "min_pts_bv":       self.min_pts_bv,
            "min_pts_bh":       self.min_pts_bh,
            "evidence_pulse_interval_ms": self.evidence_pulse_interval_ms,
            "observer_only": self.observer_only,
        })
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LetraDetectorBlock":
        b = cls(
            zone_z          = tuple(d.get("zone_z",          (55.0, 65.0))),
            grid_res        = d.get("grid_res",              20),
            v_z             = d.get("v_z",                   70.0),
            h_z             = d.get("h_z",                   72.0),
            barra_z         = d.get("barra_z",               74.0),
            slash_z         = d.get("slash_z",               76.0),
            c_z             = d.get("c_z",                   80.0),
            curv_z          = d.get("curv_z",                78.0),    # FIX v1.5.9: indentación corregida
            j_z             = d.get("j_z",                   86.0),
            z_tol           = d.get("z_tol",                 1.0),
            letra_threshold = d.get("letra_threshold",       0.55),
            inject_strength = d.get("inject_strength",       6.0),
            min_pts_bv      = d.get("min_pts_bv",            8),
            min_pts_bh      = d.get("min_pts_bh",            6),
        )
        b.evidence_pulse_interval_ms = float(
            d.get("evidence_pulse_interval_ms", 100.0)
        )
        b._last_evidence_pulse_t = {sym: -1e12 for sym in ["X", "O", "T", "A", "E"]}
        b.health = d.get("health", b.health)
        b.observer_only = bool(d.get("observer_only", True))
        b._injected = False
        return b

    def record_ground_truth(self, letra_real: str):
        """Registra por separado el desempeño del detector heurístico observacional."""
        prediction = self.last_prediction
        self._gt_total += 1
        if prediction == letra_real:
            self._gt_correct += 1
            resultado = "✅ CORRECTO"
        elif prediction is None:
            self._gt_miss += 1
            resultado = "⚠️ NO DETECTÓ (ninguna)"
        else:
            self._gt_false_pos += 1
            resultado = f"❌ HEURÍSTICO: predijo {prediction} para GT={letra_real}"
        acc = self._gt_correct / self._gt_total if self._gt_total > 0 else 0.0
        print(
            f"🧪 [GroundTruth-heurístico] GT={letra_real} | pred={prediction or 'ninguna'} | "
            f"{resultado} | acc={acc:.2%} ({self._gt_correct}/{self._gt_total})",
            flush=True,
        )
