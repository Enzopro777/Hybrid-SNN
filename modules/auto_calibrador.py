# modules/auto_calibrador.py
#
# =====================================================================
# AUTO-CALIBRADOR DE DETECTORES — v0.1
# =====================================================================
#
# Problema que resuelve:
#   Cada detector tiene umbrales (edge_threshold, corner_threshold, etc.)
#   que fueron ajustados manualmente y dependen de la densidad de señal
#   real del Transductor. Si la imagen es más densa o escasa de lo esperado,
#   todos los detectores fallan o emiten ruido masivo.
#
#   Con el auto-calibrador:
#   1. Se ejecuta una fase de "calentamiento" de N frames al arranque.
#   2. Durante esa fase mide estadísticas reales de la señal:
#      - Cuántos puntos salen del Transductor por frame (densidad)
#      - Cuántos bordes detecta cada detector
#      - El ratio real de detección vs entrada
#   3. Ajusta los umbrales de cada detector para que el ratio caiga
#      dentro de una banda objetivo (RATIO_TARGET_LO … RATIO_TARGET_HI).
#   4. Guarda los umbrales calibrados en JSON para que persistan entre arranques.
#   5. Al arrancar con persistencia, carga los umbrales guardados y los aplica.
#
# Uso:
#   En simulation.py, antes de sim.run():
#       from modules.auto_calibrador import AutoCalibrador
#       calib = AutoCalibrador(sim)
#       calib.run_or_load()   # calibra si es necesario, carga si ya existe
#
# La calibración es NO-DESTRUCTIVA: no bloquea la simulación, corre en
# modo "seco" (no inyecta activaciones a la red) durante la fase de
# calentamiento, y luego devuelve el control con todos los detectores
# ya ajustados.
#
# =====================================================================

import json
import os
import time
import numpy as np
from typing import Dict, Any

# Ruta por defecto para guardar calibraciones
CALIB_FILE = "detector_calibration.json"

# Cuántos frames de calentamiento usar para medir la señal
WARMUP_FRAMES = 80

# Ratio objetivo de detecciones respecto a puntos de entrada
# (bordes / entrada). Ajustar estos según el comportamiento deseado.
RATIO_TARGET_LO = 0.08
RATIO_TARGET_HI = 0.22

# Factor de ajuste por iteración de calibración (multiplicativo)
CALIB_STEP_UP   = 1.15   # si ratio > hi: sube threshold x1.15 (menos detecciones)
CALIB_STEP_DOWN = 0.88   # si ratio < lo: baja threshold x0.88 (más detecciones)

# Límites absolutos de los umbrales (seguridad)
THRESHOLD_MIN = 0.005
THRESHOLD_MAX = 0.98

# Número de iteraciones de ajuste por calibración
CALIB_ITERATIONS = 6


class AutoCalibrador:
    """
    Auto-calibrador de detectores visuales.

    Mide la densidad real de señal del Transductor y ajusta los
    umbrales de todos los detectores simples y combinadores para
    que el ratio de detección caiga en la banda objetivo.

    Atributos principales:
        sim       — referencia a SimulationEngine
        calib_path — ruta al JSON de calibración persistente
        verbose    — si True, imprime logs detallados
    """

    def __init__(self, sim, calib_path: str = CALIB_FILE, verbose: bool = True):
        self.sim        = sim
        self.calib_path = calib_path
        self.verbose    = verbose
        self._stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # PUNTO DE ENTRADA PRINCIPAL
    # ------------------------------------------------------------------

    def run_or_load(self) -> bool:
        """
        Carga calibración previa si existe y es reciente (<7 días),
        o ejecuta una calibración fresca.
        Devuelve True si se usó calibración cargada, False si se calibró.
        """
        if self._calib_exists_and_fresh():
            self._load_and_apply()
            return True
        else:
            self.run_full_calibration()
            return False

    def run_full_calibration(self):
        """
        Ejecuta la calibración completa:
        1. Warmup: recolecta estadísticas de señal real.
        2. Ajuste iterativo de umbrales.
        3. Guarda calibración en JSON.
        """
        print("\n" + "="*60)
        print("🔧 AUTO-CALIBRADOR: iniciando calibración de detectores")
        print(f"   Warmup: {WARMUP_FRAMES} frames | "
              f"ratio objetivo: [{RATIO_TARGET_LO:.0%}, {RATIO_TARGET_HI:.0%}]")
        print("="*60)

        # 1. Recolectar estadísticas base
        stats = self._warmup()
        if not stats:
            print("⚠️  [AutoCalib] Warmup vacío — sin señal del Transductor. "
                  "Revisá que la fuente visual esté activa.")
            return

        self._stats = stats
        self._print_stats(stats)

        # 2. Ajustar detectores simples
        self._calibrate_simple_detectors(stats)

        # 3. Ajustar combinadores
        # FIX v1.5.9: los thresholds de combinadores (corner_threshold, junction_threshold)
        # fueron calibrados manualmente para discriminar letras y NO deben ser tocados
        # por el auto_calibrador. La heurística basada en densidad del transductor no
        # considera accuracy por letra y puede colapsar la discriminación O vs X.
        # self._calibrate_combiners(stats)  ← deshabilitado

        # 4. Guardar
        self._save_calibration()

        print("\n✅ [AutoCalib] Calibración completada y guardada en "
              f"'{self.calib_path}'")
        print("="*60 + "\n")

    # ------------------------------------------------------------------
    # WARMUP: MEDIR SEÑAL REAL
    # ------------------------------------------------------------------

    def _warmup(self) -> Dict[str, Any]:
        """
        Ejecuta N frames de la fuente visual sin inyectar a la red.
        Recolecta estadísticas de densidad de transductor y bordes.
        """
        import random

        sim = self.sim
        scr = getattr(sim, 'scr_processor', None)
        if scr is None:
            print("⚠️  [AutoCalib] Sin scr_processor — warmup sintético.")
            return self._synthetic_warmup()

        letters = getattr(sim, '_letter_vocabulary',
                  getattr(sim.net, 'letter_vocabulary',
                  ['X', 'O', 'T', 'A', 'E']))

        # Acumuladores
        n_transductor_acc = []
        n_bordes_v_acc    = []
        n_bordes_h_acc    = []
        n_barra_acc       = []
        n_slash_acc       = []
        n_curvas_acc      = []

        transductor = getattr(sim, 'transductor', None)
        det_v       = getattr(sim, 'edge_detector_v', None)
        det_h       = getattr(sim, 'edge_detector_h', None)
        det_diag    = getattr(sim, 'diag_detector', None)
        det_curvas  = getattr(sim, 'curva_detector', None)

        if self.verbose:
            print(f"   Calentando con {WARMUP_FRAMES} frames de letras aleatorias...")

        for i in range(WARMUP_FRAMES):
            letter = random.choice(letters)
            pos    = (0.5, 0.5)
            frame  = scr.generate_letter_frame(letter, pos)
            coords, _, _ = scr.get_activity_coords(focus_pt=pos, virtual_frame=frame)
            scale = getattr(scr, 'scale_factor', 1.0)
            import cv2
            resized = cv2.resize(frame, None, fx=scale, fy=scale)
            static  = scr.get_static_contrast_coords(resized, pos)

            raw_acts = []
            for item in (coords or []):
                try:
                    if len(item) >= 4:
                        x, y, p_type, val = float(item[0]), float(item[1]), int(item[2]), float(item[3])
                    else:
                        continue
                    z = 45.0 if p_type == 1 else 30.0
                    raw_acts.append((x, y, z, val))
                except Exception:
                    continue
            for item in (static or []):
                try:
                    x, y, val = float(item[0]), float(item[1]), float(item[2])
                    raw_acts.append((x, y, 45.0, val))
                except Exception:
                    continue

            if not raw_acts:
                continue

            # Pasar por transductor
            if transductor is not None:
                trans_out = transductor.process(raw_acts)
            else:
                trans_out = raw_acts

            n_transductor_acc.append(len(trans_out))

            # Medir output de cada detector (sin sumar a safe_activations)
            if det_v is not None:
                bv = det_v.process(list(trans_out))
                n_bordes_v_acc.append(len(bv))

            if det_h is not None:
                bh = det_h.process(list(trans_out))
                n_bordes_h_acc.append(len(bh))

            if det_diag is not None:
                dd = det_diag.process(list(trans_out))
                barra = [a for a in dd if abs(float(a[2]) - 74.0) < 1.0]
                slash  = [a for a in dd if abs(float(a[2]) - 76.0) < 1.0]
                n_barra_acc.append(len(barra))
                n_slash_acc.append(len(slash))

            if det_curvas is not None:
                dc = det_curvas.process(list(trans_out))
                n_curvas_acc.append(len(dc))

        def _safe_stats(arr):
            if not arr:
                return {"mean": 0.0, "std": 0.0, "p25": 0.0, "p75": 0.0}
            a = np.array(arr, dtype=float)
            return {
                "mean": float(a.mean()),
                "std":  float(a.std()),
                "p25":  float(np.percentile(a, 25)),
                "p75":  float(np.percentile(a, 75)),
            }

        return {
            "transductor":   _safe_stats(n_transductor_acc),
            "bordes_v":      _safe_stats(n_bordes_v_acc),
            "bordes_h":      _safe_stats(n_bordes_h_acc),
            "barra":         _safe_stats(n_barra_acc),
            "slash":         _safe_stats(n_slash_acc),
            "curvas":        _safe_stats(n_curvas_acc),
            "n_frames":      WARMUP_FRAMES,
            "timestamp":     time.time(),
        }

    def _synthetic_warmup(self) -> Dict[str, Any]:
        """Warmup con grilla aleatoria cuando no hay scr_processor."""
        import random

        transductor = getattr(self.sim, 'transductor', None)
        det_v       = getattr(self.sim, 'edge_detector_v', None)
        det_h       = getattr(self.sim, 'edge_detector_h', None)
        det_diag    = getattr(self.sim, 'diag_detector', None)
        det_curvas  = getattr(self.sim, 'curva_detector', None)

        n_transductor_acc = []
        n_bordes_v_acc    = []
        n_bordes_h_acc    = []
        n_barra_acc       = []
        n_slash_acc       = []
        n_curvas_acc      = []

        for _ in range(WARMUP_FRAMES):
            n = random.randint(80, 200)
            raw = [(random.random(), random.random(),
                    45.0 if random.random() > 0.3 else 30.0,
                    random.uniform(0.3, 1.0)) for _ in range(n)]

            trans_out = transductor.process(raw) if transductor else raw
            n_transductor_acc.append(len(trans_out))

            if det_v:     n_bordes_v_acc.append(len(det_v.process(list(trans_out))))
            if det_h:     n_bordes_h_acc.append(len(det_h.process(list(trans_out))))
            if det_diag:
                dd = det_diag.process(list(trans_out))
                n_barra_acc.append(len([a for a in dd if abs(float(a[2])-74.0)<1.0]))
                n_slash_acc.append(len([a for a in dd if abs(float(a[2])-76.0)<1.0]))
            if det_curvas: n_curvas_acc.append(len(det_curvas.process(list(trans_out))))

        def _safe_stats(arr):
            if not arr:
                return {"mean": 0.0, "std": 0.0, "p25": 0.0, "p75": 0.0}
            a = np.array(arr, dtype=float)
            return {"mean": float(a.mean()), "std": float(a.std()),
                    "p25": float(np.percentile(a, 25)),
                    "p75": float(np.percentile(a, 75))}

        return {
            "transductor":   _safe_stats(n_transductor_acc),
            "bordes_v":      _safe_stats(n_bordes_v_acc),
            "bordes_h":      _safe_stats(n_bordes_h_acc),
            "barra":         _safe_stats(n_barra_acc),
            "slash":         _safe_stats(n_slash_acc),
            "curvas":        _safe_stats(n_curvas_acc),
            "n_frames":      WARMUP_FRAMES,
            "timestamp":     time.time(),
        }

    # ------------------------------------------------------------------
    # AJUSTE DE UMBRALES
    # ------------------------------------------------------------------

    def _calibrate_simple_detectors(self, stats: Dict):
        """
        Ajusta edge_threshold / curv_threshold de detectores simples
        para que el ratio de salida caiga en [RATIO_TARGET_LO, RATIO_TARGET_HI].
        """
        mean_entrada = max(1.0, stats["transductor"]["mean"])

        detectors_cfg = [
            ("edge_detector_v",   "bordes_v",   "edge_threshold"),
            ("edge_detector_h",   "bordes_h",   "edge_threshold"),
            ("curva_detector",    "curvas",      "curv_threshold"),
        ]

        for attr_name, stat_key, threshold_attr in detectors_cfg:
            det = getattr(self.sim, attr_name, None)
            if det is None:
                continue
            current_threshold = getattr(det, threshold_attr, None)
            if current_threshold is None:
                continue

            mean_salida = stats.get(stat_key, {}).get("mean", 0.0)
            ratio = mean_salida / mean_entrada

            if self.verbose:
                print(f"\n   [{attr_name}] threshold={current_threshold:.4f} "
                      f"| ratio actual={ratio:.3f} "
                      f"| objetivo=[{RATIO_TARGET_LO:.2f}, {RATIO_TARGET_HI:.2f}]")

            new_threshold = self._adjust_threshold(
                current_threshold, ratio, attr_name
            )
            setattr(det, threshold_attr, new_threshold)

            if self.verbose and abs(new_threshold - current_threshold) > 1e-6:
                print(f"     → ajustado a {new_threshold:.4f} "
                      f"({'+' if new_threshold > current_threshold else ''}"
                      f"{(new_threshold-current_threshold)/current_threshold*100:.1f}%)")

    def _calibrate_combiners(self, stats: Dict):
        """
        Ajusta corner_threshold / junction_threshold de combinadores.
        Los combinadores tienen entrada = media de bordes_v + bordes_h,
        así que usamos eso como referencia.
        """
        mean_bv = max(1.0, stats["bordes_v"]["mean"])
        mean_bh = max(1.0, stats["bordes_h"]["mean"])
        mean_entrada_comb = (mean_bv + mean_bh) / 2.0

        combiners_cfg = [
            ("combinador_esquina",  "corner_threshold"),
            ("junction_detector",   "junction_threshold"),
        ]

        for attr_name, threshold_attr in combiners_cfg:
            comb = getattr(self.sim, attr_name, None)
            if comb is None:
                continue
            current_threshold = getattr(comb, threshold_attr, None)
            if current_threshold is None:
                continue

            if self.verbose:
                print(f"\n   [{attr_name}] threshold={current_threshold:.4f} "
                      f"(combinador — ajuste conservador)")

            # Para combinadores no tenemos stats de warmup directas,
            # así que hacemos un ajuste basado en la densidad del transductor.
            # Si la entrada es densa, subimos un poco el threshold; si es escasa, bajamos.
            mean_entrada = max(1.0, stats["transductor"]["mean"])
            density_factor = mean_entrada / 100.0  # 100 pts = referencia neutra

            if density_factor > 1.3:
                # Señal densa: subir threshold para ser más selectivos
                new_t = min(THRESHOLD_MAX, current_threshold * 1.10)
            elif density_factor < 0.7:
                # Señal escasa: bajar threshold para ser más sensibles
                new_t = max(THRESHOLD_MIN, current_threshold * 0.90)
            else:
                new_t = current_threshold

            setattr(comb, threshold_attr, new_t)
            if self.verbose and abs(new_t - current_threshold) > 1e-6:
                print(f"     → ajustado a {new_t:.4f}")

    def _adjust_threshold(self, current: float, ratio: float, name: str) -> float:
        """
        Ajusta iterativamente un threshold hasta que el ratio proyectado
        cae en la banda objetivo.

        El ajuste es multiplicativo:
          ratio > hi → threshold * CALIB_STEP_UP   (menos detecciones)
          ratio < lo → threshold * CALIB_STEP_DOWN  (más detecciones)
        """
        t = current
        for _ in range(CALIB_ITERATIONS):
            if RATIO_TARGET_LO <= ratio <= RATIO_TARGET_HI:
                break
            if ratio > RATIO_TARGET_HI:
                t = min(THRESHOLD_MAX, t * CALIB_STEP_UP)
                # proyectar nuevo ratio (heurística: ratio ∝ 1/threshold)
                ratio = ratio * (current / max(t, 1e-6))
            else:
                t = max(THRESHOLD_MIN, t * CALIB_STEP_DOWN)
                ratio = ratio * (current / max(t, 1e-6))
        return float(np.clip(t, THRESHOLD_MIN, THRESHOLD_MAX))

    # ------------------------------------------------------------------
    # PERSISTENCIA
    # ------------------------------------------------------------------

    def _save_calibration(self):
        """Guarda los umbrales actuales de todos los detectores en JSON."""
        data = {
            "timestamp":   time.time(),
            "warmup_stats": self._stats,
            "thresholds":  self._collect_thresholds(),
        }
        try:
            with open(self.calib_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️  [AutoCalib] No se pudo guardar calibración: {e}")

    def _load_and_apply(self):
        """Carga calibración guardada y aplica umbrales a los detectores."""
        try:
            with open(self.calib_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"⚠️  [AutoCalib] Error al cargar calibración: {e}")
            return

        age_hours = (time.time() - data.get("timestamp", 0)) / 3600
        print(f"\n🔧 [AutoCalib] Cargando calibración guardada "
              f"(edad: {age_hours:.1f}h)")

        thresholds = data.get("thresholds", {})
        self._apply_thresholds(thresholds)

        print("✅ [AutoCalib] Umbrales restaurados desde persistencia.")

    def _calib_exists_and_fresh(self, max_age_days: float = 7.0) -> bool:
        if not os.path.exists(self.calib_path):
            return False
        try:
            with open(self.calib_path, "r") as f:
                data = json.load(f)
            age = (time.time() - data.get("timestamp", 0)) / 86400
            return age < max_age_days
        except Exception:
            return False

    def _collect_thresholds(self) -> dict:
        """Recolecta los umbrales actuales de todos los detectores."""
        result = {}
        detector_map = {
            "edge_detector_v":    ["edge_threshold"],
            "edge_detector_h":    ["edge_threshold"],
            "diag_detector":      ["edge_threshold"],
            "curva_detector":     ["curv_threshold"],
            "simetria_detector":  ["sym_threshold"],
            "combinador_esquina": ["corner_threshold", "min_bordes_v", "min_bordes_h"],
            "junction_detector":  ["junction_threshold", "min_bordes"],
        }
        for attr_name, param_list in detector_map.items():
            obj = getattr(self.sim, attr_name, None)
            if obj is None:
                continue
            result[attr_name] = {}
            for param in param_list:
                val = getattr(obj, param, None)
                if val is not None:
                    result[attr_name][param] = val
        return result

    def _apply_thresholds(self, thresholds: dict):
        """Aplica umbrales guardados a los detectores."""
        for attr_name, params in thresholds.items():
            obj = getattr(self.sim, attr_name, None)
            if obj is None:
                continue
            for param, val in params.items():
                if hasattr(obj, param):
                    setattr(obj, param, val)
                    if self.verbose:
                        print(f"   [{attr_name}] {param} = {val}")

    # ------------------------------------------------------------------
    # DIAGNÓSTICO
    # ------------------------------------------------------------------

    def _print_stats(self, stats: Dict):
        print(f"\n   📊 Estadísticas de señal ({stats['n_frames']} frames):")
        for key in ["transductor", "bordes_v", "bordes_h", "barra", "slash", "curvas"]:
            s = stats.get(key, {})
            mean = s.get("mean", 0.0)
            std  = s.get("std",  0.0)
            if mean > 0:
                print(f"      {key:<16} media={mean:6.1f}  std={std:5.1f}  "
                      f"p25={s.get('p25', 0):.0f}  p75={s.get('p75', 0):.0f}")

    def print_status(self):
        """Imprime el estado actual de todos los umbrales."""
        print("\n🔧 [AutoCalib] Umbrales actuales:")
        th = self._collect_thresholds()
        for name, params in th.items():
            for p, v in params.items():
                print(f"   {name}.{p} = {v:.4f}")

    def force_recalibrate(self):
        """Fuerza una calibración nueva, ignorando la persistencia."""
        if os.path.exists(self.calib_path):
            os.remove(self.calib_path)
        self.run_full_calibration()
