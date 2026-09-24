# modules/causal_probe.py
"""Instrumentación causal ligera para la arquitectura SNN.

Objetivo Fase 1:
- registrar la cadena funcional bloque -> evidencia -> decisión;
- aprovechar las neuronas inmortales existentes como referencias estables,
  sin crear neuronas nuevas ni alterar su dinámica;
- separar actividad natural de actividad conocida como electroshock;
- mantener una traza agregada y barata en CPU/RAM;
- producir candidatos a pruebas causales posteriores.

IMPORTANTE: esta versión es observacional/no invasiva. No modifica pesos,
conexiones ni umbrales de la red.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
import hashlib
import uuid
from datetime import datetime, timezone
from collections import defaultdict, deque
from copy import deepcopy
from typing import Any, Dict, Iterable, Optional

import numpy as np

try:
    from config import (
        CAUSAL_VISION_TELEMETRY_ENABLED,
        CAUSAL_VISION_TRACE_STRIDE,
        CAUSAL_VISION_TARGET_RADIUS,
        CAUSAL_VISION_SACCADE_STEP,
        CAUSAL_VISION_FIXATION_STEP,
        CAUSAL_VISION_MIN_FIXATION_FRAMES,
    )
except Exception:
    CAUSAL_VISION_TELEMETRY_ENABLED = True
    CAUSAL_VISION_TRACE_STRIDE = 1
    CAUSAL_VISION_TARGET_RADIUS = 0.10
    CAUSAL_VISION_SACCADE_STEP = 0.04
    CAUSAL_VISION_FIXATION_STEP = 0.01
    CAUSAL_VISION_MIN_FIXATION_FRAMES = 3


class CausalProbe:
    """Registrador de actividad funcional y analizador de precedencia temporal."""

    VERSION = "causal-probe-v1.5"
    LEGACY_VERSION_V1_4 = "causal-probe-v1.4"
    SCHEMA_VERSION = "causal-probe-results-v2"
    PROJECT_VERSION = "IA-1.6.29f"

    # Salidas principales que existen hoy en vision.py.
    BLOCKS = (
        "transductor",
        "bordes_v",
        "bordes_h",
        "diagonales",
        "esquinas",
        "curvas",
        "simetria",
        "junctions",
        "letras",
    )

    SYMBOLS = ("X", "O", "T", "A", "E")

    def __init__(
        self,
        net=None,
        max_events: int = 4000,
        flush_every: int = 100,
        output_path: str = "causal_probe_results.json",
    ):
        self.net = net
        self.max_events = int(max_events)
        self.flush_every = int(max(1, flush_every))
        self.output_path = output_path
        self.enabled = True
        self.session_id = uuid.uuid4().hex
        self.started_at = time.time()
        self._events_total = 0

        self._lock = threading.RLock()
        self._event_counter = 0
        self._trial_counter = 0
        self._active_trial: Optional[Dict[str, Any]] = None
        self._events = deque(maxlen=self.max_events)
        self._trials = deque(maxlen=500)
        self._block_acc = defaultdict(lambda: {
            "frames": 0,
            "output_count": 0,
            "strength_sum": 0.0,
            "last_t": None,
        })
        self._symbol_acc = defaultdict(lambda: {
            "frames": 0,
            "evidence_sum": 0.0,
            "evidence_peak": 0.0,
            "evidence_active_frames": 0,
        })

        self.metrics = {
            "frames": 0,
            "natural_spike_samples": 0,
            "immortal_spike_samples": 0,
            "blocks_seen": 0,
            "trials_completed": 0,
            "last_update": 0.0,
        }

        # Últimos valores y tiempos; sirven para estudiar precedencia.
        self._last_block = {}
        self._last_evidence = {sym: 0.0 for sym in self.SYMBOLS}
        self._last_evidence_t = {sym: None for sym in self.SYMBOLS}

    # ------------------------------------------------------------------
    # UTILIDADES
    # ------------------------------------------------------------------

    @staticmethod
    def _iso_utc(ts: Any) -> Optional[str]:
        try:
            return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
        except Exception:
            return None

    @staticmethod
    def _sha256_file(path: str) -> Optional[str]:
        try:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return None

    def _source_manifest(self, net=None) -> dict:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        names = [
            "modules/causal_probe.py",
            "modules/language_io.py",
            "modules/detector_letras.py",
            "engine/simulation.py",
            "config.py",
        ]
        manifest = {}
        for name in names:
            path = os.path.join(root, name)
            digest = self._sha256_file(path)
            if digest:
                manifest[name] = digest
        return manifest

    def _runtime_contract(self, net=None) -> dict:
        modules = {}
        if net is not None and hasattr(net, "modules"):
            for name, module in net.modules.items():
                modules[str(name)] = module.__class__.__name__
        contract = {
            "project_version": self.PROJECT_VERSION,
            "probe_version": self.VERSION,
            "schema_version": self.SCHEMA_VERSION,
            "clock": {
                "simulation_time_unit": "ms",
                "wall_clock": "unix_seconds",
                "generated_at_iso_utc": self._iso_utc(time.time()),
            },
            "temporal_core_version": (
                getattr(getattr(net, "temporal_core", None), "VERSION", None)
                if net is not None else None
            ),
            "measurement_semantics": {
                "block_count": "observable outputs/items in a block for one frame",
                "block_strength": "sum of non-negative intensity/strength values",
                "evidence_value": "language input port value for each symbol; neural observational channel",
                "evidence_active_threshold": "max(lang.edge_threshold, 0.10)",
                "natural_spikes": "active neurons with last_spike within 50 ms of observed frame",
                "causal_status": "observational_only_until_intervention",
                "vision_telemetry": "per-frame normalized fovea trajectory; observational; no attention reward applied",
                "readout_delivery_diagnostics": "source_event_to_pool_membrane_threshold snapshot",
            },
            "readout_contract": {
                "evidence_ports_role": "neural observational pathway; not the learned classifier",
                "detector_hypothesis_role": "heuristic detector score kept separate from evidence_*",
                "readout_validity_rule": "prediction_match is not sufficient; learned_readout requires source_or_pool_spikes",
                "prediction_match_semantics": "gt_equals_prediction is reported separately from readout activity/validity",
                "learned_readout_validity": "source_fired_or_pool_spike_required",
                "queue_diagnostics": "production_vs_consumption_and_future_event_pressure",
                "queue_control_policy": "synaptic_backpressure_before_hard_cap; sensory_input_preserved",
                "energy_scope": "neuronal_and_regional_energy_are_reported_separately",
                "learned_readout_role": "neuronal classifier pathway evaluated from readout pools",
                "latent_workspace": "post-perception recurrent latent workspace; guides bridge learning, not decision authority by default",
                "latent_workspace_authority_default": False,
                "readout_evidence_scoring": "instantaneous_spikes_plus_temporal_surprise_plus_presence_minus_slow_positive_history_bias",
                "source_selection": "cortical_representation",
                "source_selection_rationale": "readout sources must be explicitly role-tagged cortical_representation; FeatureBus, language infrastructure and monitor immortals are excluded",
                "cortical_representation_diagnostics": "trial source identity, uniqueness, temporal bins and cross-trial Jaccard overlap",
                "frame_timing_diagnostics": "wall duration, simulation duration, per-frame cadence and queue pressure",
                "prediction_authority_during_letter_trials": "temporal_pool_spike_activity_when_readout_ready",
            "readout_source_vthresh": 2.5,
            "readout_source_input_strength_expected": [3.0, 8.0],
            "prediction_recording": "causal_probe_end_trial_uses_authoritative_trial_prediction",
                "queue_governor": "causal_utility_budget_plus_safe_event_aggregation; severe_backlog_coalescing_reserved_for_noncausal_load",
                "queue_backpressure": {"high_watermark": 22000, "low_watermark": 14000, "severe_load_watermark": 23000, "frames_never_skipped": True, "causal_utility_budget": True, "safe_event_aggregation": True, "ready_backlog_coalescing": True},
            },
            "network": {
                "neurons": int(getattr(net, "n", 0)) if net is not None else None,
                "max_neurons": int(getattr(net, "max_neurons", 0)) if net is not None else None,
                "current_time": self._safe_float(getattr(net, "current_time", 0.0)) if net is not None else None,
            },
            "symbols": list(self.SYMBOLS),
            "blocks_expected": list(self.BLOCKS),
            "modules_present": modules,
        }
        contract["source_manifest"] = self._source_manifest(net)
        workspace_root = os.path.basename(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
        declared_norm = self.PROJECT_VERSION.lower().replace("ia-", "").replace("ia ", "")
        workspace_norm = workspace_root.lower().replace("ia-", "").replace("ia ", "")
        contract["execution_identity"] = {
            "workspace_root": workspace_root,
            "probe_module_path": os.path.abspath(__file__),
            "declared_project_version": self.PROJECT_VERSION,
            "version_label_consistent": declared_norm in workspace_norm,
        }
        return contract

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            value = float(value)
            return value if math.isfinite(value) else default
        except Exception:
            return default

    @staticmethod
    def _activation_strengths(items: Any) -> tuple[int, float, float]:
        """(cantidad, suma_strength, max_strength) para una salida de bloque."""
        if not items:
            return 0, 0.0, 0.0
        count = 0
        total = 0.0
        peak = 0.0
        for item in items:
            try:
                if isinstance(item, dict):
                    val = item.get("intensity", item.get("strength", 0.0))
                elif len(item) >= 4:
                    val = item[3]
                else:
                    val = 1.0
                val = max(0.0, float(val))
            except Exception:
                val = 0.0
            count += 1
            total += val
            peak = max(peak, val)
        return count, total, peak

    # ------------------------------------------------------------------
    # IA 1.6.29c — observabilidad de visión y decisión
    # ------------------------------------------------------------------

    @staticmethod
    def _vision_snapshot(vision) -> Optional[dict]:
        if vision is None:
            return None
        try:
            if hasattr(vision, "get_fovea_telemetry"):
                data = dict(vision.get_fovea_telemetry() or {})
            else:
                data = {
                    "x": float(getattr(vision, "fovea_center_x", 0.5)),
                    "y": float(getattr(vision, "fovea_center_y", 0.5)),
                    "radius": float(getattr(getattr(vision, "fovea_radius_port", None), "value", 0.15)),
                    "vx": float(getattr(vision, "vel_x", 0.0)),
                    "vy": float(getattr(vision, "vel_y", 0.0)),
                }
                data["speed"] = float(math.hypot(data["vx"], data["vy"]))
            return {
                "x": round(float(np.clip(data.get("x", 0.5), 0.0, 1.0)), 5),
                "y": round(float(np.clip(data.get("y", 0.5), 0.0, 1.0)), 5),
                "radius": round(float(np.clip(data.get("radius", 0.15), 0.0, 0.5)), 5),
                "vx": round(float(data.get("vx", 0.0)), 6),
                "vy": round(float(data.get("vy", 0.0)), 6),
                "speed": round(float(max(0.0, data.get("speed", 0.0))), 6),
            }
        except Exception:
            return None

    @staticmethod
    def _new_vision_trial_state(vision=None, target_xy=None) -> dict:
        start = CausalProbe._vision_snapshot(vision) or {
            "x": 0.5, "y": 0.5, "radius": 0.15, "vx": 0.0, "vy": 0.0, "speed": 0.0
        }
        target = target_xy if target_xy is not None and len(target_xy) >= 2 else (0.5, 0.5)
        tx = float(np.clip(target[0], 0.0, 1.0)); ty = float(np.clip(target[1], 0.0, 1.0))
        return {
            "enabled": bool(CAUSAL_VISION_TELEMETRY_ENABLED),
            "coordinate_space": "normalized_0_1",
            "target": {
                "x": round(tx, 5),
                "y": round(ty, 5),
                "radius": round(float(CAUSAL_VISION_TARGET_RADIUS), 5),
            },
            # Posición antes de cualquier preparación/reset del trial.
            "fovea_start_pre_trial": dict(start),
            "fovea_start_effective": None,
            "fovea_end": None,
            "fovea_mean": {"x": 0.0, "y": 0.0},
            "min_distance_to_target": None,
            "mean_distance_to_target": 0.0,
            "frames_in_target": 0,
            "ratio_in_target": 0.0,
            "path_length": 0.0,
            "largest_step": 0.0,
            "largest_speed": 0.0,
            "saccade_count": 0,
            "fixation_count": 0,
            "trace_stride": int(max(1, CAUSAL_VISION_TRACE_STRIDE)),
            "trace": [],
            "_trace_frames": 0,
            "_sum_x": 0.0,
            "_sum_y": 0.0,
            "_sum_distance": 0.0,
            "_prev": None,
            "_fixation_run": 0,
            "_inside_run": 0,
            "_frame_count": 0,
        }

    @staticmethod
    def _finish_vision_trial_state(state: dict) -> dict:
        count = int(state.get("_frame_count", 0))
        if count > 0:
            state["fovea_mean"] = {
                "x": round(float(state.get("_sum_x", 0.0)) / count, 5),
                "y": round(float(state.get("_sum_y", 0.0)) / count, 5),
            }
            state["mean_distance_to_target"] = round(float(state.get("_sum_distance", 0.0)) / count, 5)
            state["ratio_in_target"] = round(float(state.get("frames_in_target", 0)) / count, 5)
        state["fovea_end"] = dict(state.get("_prev")) if state.get("_prev") else state.get("fovea_start_pre_trial")
        if int(state.get("_fixation_run", 0)) >= int(CAUSAL_VISION_MIN_FIXATION_FRAMES):
            state["fixation_count"] = int(state.get("fixation_count", 0)) + 1
        state["trace_stride"] = int(max(1, state.get("trace_stride", 1)))
        for key in ("_trace_frames", "_sum_x", "_sum_y", "_sum_distance", "_prev", "_fixation_run", "_inside_run", "_frame_count"):
            state.pop(key, None)
        return state

    @staticmethod
    def _decision_audit_from_readout(readout_activity: Optional[dict], authoritative_prediction: Optional[str]) -> dict:
        pools = {}
        if isinstance(readout_activity, dict):
            pools = readout_activity.get("pool_spikes") or {}
        pool_spike_counts = {str(k): int(max(0, int(v))) for k, v in pools.items()}
        spike_total = sum(pool_spike_counts.values())
        spike_argmax = max(pool_spike_counts, key=pool_spike_counts.get) if spike_total > 0 else None
        r1627 = readout_activity.get("readout_1627", {}) if isinstance(readout_activity, dict) else {}
        drive = {str(k): float(v) for k, v in (r1627.get("pool_drive_ema") or {}).items()}
        rate = {str(k): float(v) for k, v in (r1627.get("pool_rate_ema") or {}).items()}
        inhibition = {str(k): float(v) for k, v in (r1627.get("pool_inhibition") or {}).items()}
        bridge = readout_activity.get("latent_readout_bridge", {}) if isinstance(readout_activity, dict) else {}
        bridge_pools = bridge.get("pools", {}) if isinstance(bridge, dict) else {}
        for sym, item in bridge_pools.items():
            if sym not in inhibition:
                try:
                    inhibition[str(sym)] = float(item.get("pool_inhibition", 0.0))
                except Exception:
                    inhibition[str(sym)] = 0.0
        drive_argmax = max(drive, key=drive.get) if drive else None
        rate_argmax = max(rate, key=rate.get) if rate else None
        delivery = readout_activity.get("pool_delivery_diagnostics") if isinstance(readout_activity, dict) else {}
        threshold_mean = {}
        threshold_min = {}
        for sym, diag in (delivery or {}).items():
            try:
                events = int(diag.get("events", 0))
                threshold_mean[str(sym)] = round(float(diag.get("threshold_sum", 0.0)) / events, 6) if events else 0.0
                threshold_min[str(sym)] = float(diag.get("effective_threshold_min", 0.0)) if events else 0.0
            except Exception:
                threshold_mean[str(sym)] = 0.0
                threshold_min[str(sym)] = 0.0
        authoritative = str(authoritative_prediction) if authoritative_prediction is not None else None
        pool_level = {}
        for sym in sorted(set(pool_spike_counts) | set(drive) | set(rate) | set(inhibition) | set(threshold_mean)):
            pool_level[sym] = {
                "spikes": int(pool_spike_counts.get(sym, 0)),
                "spike_share": round(float(pool_spike_counts.get(sym, 0)) / max(1, spike_total), 5),
                "drive_ema": round(float(drive.get(sym, 0.0)), 6),
                "rate_ema": round(float(rate.get(sym, 0.0)), 6),
                "inhibition": round(float(inhibition.get(sym, 0.0)), 6),
                "threshold_mean": round(float(threshold_mean.get(sym, 0.0)), 6),
                "threshold_min": round(float(threshold_min.get(sym, 0.0)), 6),
            }
        return {
            "version": "1.6.29f-decision-audit-v4",
            "authoritative_prediction": authoritative,
            "pool_spike_argmax": spike_argmax,
            "pool_drive_argmax": drive_argmax,
            "pool_rate_argmax": rate_argmax,
            "argmax_matches_authority": bool(authoritative is not None and authoritative == spike_argmax),
            "prediction_path_consistent": bool(authoritative == spike_argmax) if authoritative is not None else False,
            "history_bias_index": float(max(0.0, drive.get(spike_argmax, 0.0) - (sum(drive.values()) / max(1, len(drive))))) if spike_argmax in drive and drive else 0.0,
            "spike_total": int(spike_total),
            "pools": pool_level,
        }

    # ------------------------------------------------------------------
    # API DE OBSERVACIÓN
    # ------------------------------------------------------------------

    def begin_trial(self, label: str, ground_truth: Optional[str] = None, t: float = 0.0, expected_frames: Optional[int] = None, vision=None, target_xy=None) -> bool:
        """Abre una ventana de medición para un trial.

        No sobrescribe silenciosamente un trial ya activo: la sonda es una
        autoridad de medición y debe tener como máximo una ventana abierta.
        """
        with self._lock:
            if self._active_trial is not None:
                return False
            self._trial_counter += 1
            self._active_trial = {
                "id": self._trial_counter,
                "label": str(label),
                "ground_truth": ground_truth,
                "start_t": self._safe_float(t),
                "end_t": None,
                "frames": 0,
                "expected_frames": int(expected_frames) if expected_frames is not None else None,
                "complete_frames": False,
                "block_counts": defaultdict(int),
                "block_strength": defaultdict(float),
                "evidence_peak": {sym: 0.0 for sym in self.SYMBOLS},
                "evidence_frames": {sym: 0 for sym in self.SYMBOLS},
                "evidence_first_t": {sym: None for sym in self.SYMBOLS},
                "evidence_last_t": {sym: None for sym in self.SYMBOLS},
                "prediction": None,
                "prediction_source": None,
                "certainty": 0.0,
                "termination_reason": None,
                # FIX 1.4.9: telemetría de readout
                "readout_frames":            0,
                "readout_sources_fired_sum": 0,
                "readout_pool_spikes":       {},
                "readout_weight_mean_last":  {},
                "readout_weight_delta_last": {},
                "readout_source_spikes": 0,
                "readout_source_unique": 0,
                "readout_pool_unique": {},
                "detector_scores_peak": {sym: 0.0 for sym in self.SYMBOLS},
                "readout_pool_delivery_diagnostics": {},
                "readout_pool_delivery_samples": [],
                "cortical_source_indices": [],
                "cortical_source_spike_counts": {},
                "cortical_source_bin_counts": {},
                "frame_timing": [],
                "frame_timing_summary": {},
                "vision_telemetry": self._new_vision_trial_state(vision=vision, target_xy=target_xy),
                "decision_audit": {},
            }
            return True

    def end_trial(self, prediction: Optional[str] = None, certainty: float = 0.0, t: float = 0.0, prediction_source: Optional[str] = None, termination_reason: Optional[str] = None, readout_activity: Optional[dict] = None) -> Optional[dict]:
        with self._lock:
            if self._active_trial is None:
                return None
            trial = self._active_trial
            trial["end_t"] = self._safe_float(t)
            trial["prediction"] = prediction
            trial["prediction_source"] = prediction_source
            trial["certainty"] = self._safe_float(certainty)
            trial["termination_reason"] = str(termination_reason) if termination_reason else (
                "completed" if trial.get("expected_frames") is None or trial.get("frames") == int(trial.get("expected_frames")) else "incomplete_exit"
            )
            trial["termination_frame"] = int(trial.get("frames", 0))
            if trial.get("vision_telemetry"):
                trial["vision_telemetry"] = self._finish_vision_trial_state(trial["vision_telemetry"])
            # v1.6.17: el cierre del Probe usa el snapshot congelado por
            # LanguageIO, de modo que no depende de la última observación de
            # frame ni de actividad tardía posterior al asentamiento.
            if readout_activity:
                trial["readout_sources_fired_sum"] = int(readout_activity.get("source_spikes", 0))
                trial["readout_source_spikes"] = int(readout_activity.get("source_spikes", 0))
                trial["readout_source_unique"] = int(readout_activity.get("source_unique", 0))
                trial["readout_pool_spikes"] = {
                    str(sym): int(readout_activity.get("pool_spikes", {}).get(sym, 0))
                    for sym in self.SYMBOLS
                }
                trial["readout_pool_unique"] = {
                    str(sym): int(readout_activity.get("pool_unique", {}).get(sym, 0))
                    for sym in self.SYMBOLS
                }
                trial["readout_pool_spikes_total"] = int(readout_activity.get("pool_spikes_total", 0))
                trial["readout_pool_ledger_consistent"] = bool(readout_activity.get("pool_ledger_consistent", False))
                trial["readout_pool_delivery_diagnostics"] = {
                    str(sym): dict(readout_activity.get("pool_delivery_diagnostics", {}).get(sym, {}))
                    for sym in self.SYMBOLS
                }
                trial["readout_pool_delivery_samples"] = list(readout_activity.get("pool_delivery_samples", []))
                if readout_activity.get("readout_1627"):
                    trial["readout_1627"] = deepcopy(readout_activity.get("readout_1627"))
                if readout_activity.get("latent_workspace"):
                    trial["latent_workspace"] = deepcopy(readout_activity.get("latent_workspace"))
                if readout_activity.get("queue_trial_audit"):
                    trial["queue_trial_audit"] = deepcopy(readout_activity.get("queue_trial_audit"))
                if readout_activity.get("latent_readout_bridge"):
                    trial["latent_readout_bridge"] = deepcopy(readout_activity.get("latent_readout_bridge"))
                if readout_activity.get("latent_readout_error_ledger"):
                    trial["latent_readout_error_ledger"] = deepcopy(readout_activity.get("latent_readout_error_ledger"))
                trial["decision_audit"] = self._decision_audit_from_readout(readout_activity, prediction)
            trial["block_counts"] = dict(trial["block_counts"])
            trial["block_strength"] = dict(trial["block_strength"])
            trial["evidence_first_t"] = dict(trial["evidence_first_t"])
            trial["evidence_last_t"] = dict(trial["evidence_last_t"])
            if readout_activity:
                trial["cortical_source_indices"] = list(readout_activity.get("source_indices", []))
                trial["cortical_source_spike_counts"] = dict(readout_activity.get("source_spike_counts", {}))
                trial["cortical_source_bin_counts"] = dict(readout_activity.get("source_bin_counts", {}))
            ft = list(trial.get("frame_timing", []))
            if ft:
                sim_d = [float(x["simulation_duration_ms"]) for x in ft if x.get("simulation_duration_ms") is not None]
                wall_d = [float(x["wall_duration_s"]) for x in ft if x.get("wall_duration_s") is not None]
                q_b = [int(x["queue_before"]) for x in ft if x.get("queue_before") is not None]
                q_a = [int(x["queue_after"]) for x in ft if x.get("queue_after") is not None]
                trial["frame_timing_summary"] = {
                    "frames_measured": len(ft),
                    "simulation_duration_ms_total": float(sum(sim_d)) if sim_d else 0.0,
                    "simulation_duration_ms_median": float(np.median(sim_d)) if sim_d else 0.0,
                    "wall_duration_s_total": float(sum(wall_d)) if wall_d else 0.0,
                    "wall_duration_s_median": float(np.median(wall_d)) if wall_d else 0.0,
                    "simulation_ms_per_wall_second": (float(sum(sim_d)) / float(sum(wall_d))) if wall_d and sum(wall_d) > 1e-9 else None,
                    "queue_peak_before": max(q_b) if q_b else None,
                    "queue_peak_after": max(q_a) if q_a else None,
                    "queue_delta_total": (q_a[-1] - q_b[0]) if q_b and q_a else None,
                }
            # Build diagnóstico 1.6.10: end_trial ocurre después de
            # apply_readout_learning(), por lo que aquí podemos capturar el
            # estado post-plasticidad sin confundirlo con la muestra pre-aprendizaje.
            try:
                _lang = getattr(self.net, "modules", {}).get("lenguaje") if self.net is not None else None
                if _lang is not None and hasattr(_lang, "get_readout_telemetry"):
                    _rt_final = _lang.get_readout_telemetry(self.net, t=float(t))
                    trial["readout_synapse_stats_final"] = {
                        _s: {
                            "count": int(_rt_final.get(f"readout_{_s}_synapse_count", 0)),
                            "min": float(_rt_final.get(f"readout_{_s}_weight_min", 0.0)),
                            "max": float(_rt_final.get(f"readout_{_s}_weight_max", 0.0)),
                            "std": float(_rt_final.get(f"readout_{_s}_weight_std", 0.0)),
                        }
                        for _s in getattr(_lang, "symbols", [])
                    }
                    _audit = _rt_final.get("learning_audit")
                    if _audit:
                        trial["readout_learning_audit"] = deepcopy(_audit)
            except Exception:
                pass
            expected = trial.get("expected_frames")
            if expected is None:
                trial["complete_frames"] = True
            else:
                trial["complete_frames"] = (trial["frames"] == int(expected))
            self._trials.append(deepcopy(trial))
            self._active_trial = None
            self.metrics["trials_recorded"] = int(self.metrics.get("trials_recorded", 0)) + 1
            if trial["complete_frames"]:
                self.metrics["trials_completed"] += 1
            result = self._trial_causal_summary(trial)
            result["frames_expected"] = expected
            result["frames_observed"] = trial["frames"]
            result["frames_complete"] = trial["complete_frames"]
        self.flush()
        return result

    def observe_frame(self, *, t: float, outputs: Optional[Dict[str, Iterable]] = None, net=None,
                       wall_started: Optional[float] = None, wall_finished: Optional[float] = None,
                       simulation_started: Optional[float] = None, queue_before: Optional[int] = None,
                       queue_after: Optional[int] = None, vision=None):
        """Registra una instantánea de los bloques y de la actividad neuronal."""
        if not self.enabled:
            return
        net = net or self.net
        now_t = self._safe_float(t)

        with self._lock:
            self.metrics["frames"] += 1
            self.metrics["last_update"] = time.time()

            event = {"t": now_t, "blocks": {}, "evidence": {}}

            if self._active_trial is not None:
                self._active_trial["frames"] += 1
                if wall_started is not None or wall_finished is not None or simulation_started is not None:
                    ft = {
                        "frame": int(self._active_trial["frames"]),
                        "simulation_t_start": self._safe_float(simulation_started, now_t) if simulation_started is not None else now_t,
                        "simulation_t_end": now_t,
                        "simulation_duration_ms": self._safe_float(now_t - float(simulation_started)) if simulation_started is not None else None,
                        "wall_duration_s": self._safe_float(float(wall_finished) - float(wall_started)) if wall_started is not None and wall_finished is not None else None,
                        "queue_before": int(queue_before) if queue_before is not None else None,
                        "queue_after": int(queue_after) if queue_after is not None else None,
                    }
                    self._active_trial["frame_timing"].append(ft)

                vt = self._active_trial.get("vision_telemetry")
                if vt is not None and bool(vt.get("enabled", True)):
                    snap = self._vision_snapshot(vision)
                    if snap is not None:
                        target = vt.get("target", {})
                        tx = float(target.get("x", 0.5)); ty = float(target.get("y", 0.5))
                        dx = float(snap["x"] - tx); dy = float(snap["y"] - ty)
                        distance = float(math.hypot(dx, dy))
                        target_radius = float(CAUSAL_VISION_TARGET_RADIUS)
                        step = 0.0
                        prev = vt.get("_prev")
                        if prev is not None:
                            step = float(math.hypot(snap["x"] - prev["x"], snap["y"] - prev["y"]))
                            vt["path_length"] += step
                        if vt.get("fovea_start_effective") is None:
                            vt["fovea_start_effective"] = dict(snap)
                        if distance < (vt.get("min_distance_to_target") if vt.get("min_distance_to_target") is not None else distance):
                            vt["min_distance_to_target"] = distance
                        in_target = bool(distance <= target_radius)
                        vt["frames_in_target"] += int(in_target)
                        vt["_frame_count"] += 1
                        vt["_sum_x"] += snap["x"]
                        vt["_sum_y"] += snap["y"]
                        vt["_sum_distance"] += distance
                        vt["largest_step"] = max(float(vt.get("largest_step", 0.0)), step)
                        vt["largest_speed"] = max(float(vt.get("largest_speed", 0.0)), float(snap.get("speed", 0.0)))
                        saccade_step = float(CAUSAL_VISION_SACCADE_STEP)
                        fixation_step = float(CAUSAL_VISION_FIXATION_STEP)
                        min_fix = int(CAUSAL_VISION_MIN_FIXATION_FRAMES)
                        if step >= saccade_step:
                            vt["saccade_count"] += 1
                        if step <= fixation_step:
                            vt["_fixation_run"] += 1
                        else:
                            if vt.get("_fixation_run", 0) >= min_fix:
                                vt["fixation_count"] += 1
                            vt["_fixation_run"] = 0
                        stride = int(max(1, vt.get("trace_stride", 1)))
                        frame_no = int(self._active_trial["frames"])
                        if frame_no == 1 or (frame_no - 1) % stride == 0:
                            vt["trace"].append({
                                "frame": frame_no,
                                "t_ms": round(now_t - float(self._active_trial.get("start_t", now_t)), 5),
                                "x": snap["x"], "y": snap["y"],
                                "radius": snap["radius"],
                                "dx": round(snap["x"] - (prev["x"] if prev else snap["x"]), 5),
                                "dy": round(snap["y"] - (prev["y"] if prev else snap["y"]), 5),
                                "step": round(step, 5),
                                "distance_to_target": round(distance, 5),
                                "in_target": in_target,
                                "speed": snap["speed"],
                            })
                        vt["_prev"] = dict(snap)

            if outputs:
                for block_name, items in outputs.items():
                    count, strength_sum, peak = self._activation_strengths(items)
                    event["blocks"][block_name] = {
                        "count": count,
                        "strength": round(strength_sum, 5),
                        "peak": round(peak, 5),
                    }
                    acc = self._block_acc[block_name]
                    acc["frames"] += 1
                    acc["output_count"] += count
                    acc["strength_sum"] += strength_sum
                    acc["last_t"] = now_t
                    self._last_block[block_name] = now_t

                    if self._active_trial is not None:
                        self._active_trial["block_counts"][block_name] += count
                        self._active_trial["block_strength"][block_name] += strength_sum

            # Telemetría del detector de letras. El bloque actualmente devuelve
            # una lista vacía al pipeline porque su salida útil son los scores y
            # los spikes de evidencia. Para la sonda, esos scores son la salida
            # observable del bloque "letras".
            if net is not None:
                try:
                    detector = getattr(net, "modules", {}).get("detector_letras")
                    if detector is not None:
                        health = getattr(detector, "health", {})
                        scores = dict(health.get("scores_ultimo", {}) or {})
                        if scores:
                            top_sym = max(scores, key=scores.get)
                            top_score = self._safe_float(scores.get(top_sym, 0.0))
                            event["blocks"]["letras"] = {
                                "count": 1 if top_score > 0.0 else 0,
                                "strength": round(top_score, 5),
                                "peak": round(top_score, 5),
                                "symbol": top_sym,
                                "scores": {k: round(self._safe_float(v), 5) for k, v in scores.items()},
                                # FIX v1.5.9: exponer curv_per_bv para diagnóstico causal
                                "curv_per_bv": round(self._safe_float(health.get("curv_per_bv", -1.0)), 4),
                                "curv_grid_max": round(self._safe_float(health.get("curv_grid_max", -1.0)), 4),
                            }
                            acc = self._block_acc["letras"]
                            acc["frames"] += 1
                            acc["output_count"] += 1 if top_score > 0.0 else 0
                            acc["strength_sum"] += top_score
                            acc["last_t"] = now_t
                            self._last_block["letras"] = now_t
                            if self._active_trial is not None:
                                self._active_trial["block_counts"]["letras"] += 1 if top_score > 0.0 else 0
                                self._active_trial["block_strength"]["letras"] += top_score
                                for _s, _v in scores.items():
                                    self._active_trial["detector_scores_peak"][_s] = max(
                                        self._active_trial["detector_scores_peak"].get(_s, 0.0),
                                        self._safe_float(_v)
                                    )
                except Exception:
                    pass

            # Evidencia derivada de los puertos del lenguaje.
            if net is not None:
                lang = getattr(net, "modules", {}).get("lenguaje") if hasattr(net, "modules") else None
                if lang is not None:
                    for sym in getattr(lang, "symbols", self.SYMBOLS):
                        port = lang.input_ports.get(f"evidencia_{sym}")
                        if port is None:
                            continue
                        value = self._safe_float(getattr(port, "value", 0.0))
                        event["evidence"][sym] = round(value, 5)
                        sacc = self._symbol_acc[sym]
                        sacc["frames"] += 1
                        sacc["evidence_sum"] += value
                        sacc["evidence_peak"] = max(sacc["evidence_peak"], value)
                        # v0.6: umbral más alto para contar frames y fijar first_t;
                        # reduce los falsos positivos de timestamp cuando todas las
                        # letras suben un poco por ruido de grilla compartida.
                        _active_threshold = max(
                            getattr(lang, "edge_threshold", 0.02), 0.10
                        )
                        if value > _active_threshold:
                            sacc["evidence_active_frames"] += 1

                        prev = self._last_evidence[sym]
                        if value > prev:
                            self._last_evidence_t[sym] = now_t
                        self._last_evidence[sym] = value

                        if self._active_trial is not None:
                            tr = self._active_trial
                            tr["evidence_peak"][sym] = max(tr["evidence_peak"][sym], value)
                            if value > _active_threshold:
                                tr["evidence_frames"][sym] += 1
                                if tr["evidence_first_t"][sym] is None:
                                    tr["evidence_first_t"][sym] = now_t
                                tr["evidence_last_t"][sym] = now_t

            # Actividad neuronal: solo se contabiliza; no se modifica la red.
            if net is not None and hasattr(net, "last_spike"):
                try:
                    n = int(net.n)
                    with net.lock:
                        last_sp = np.asarray(net.last_spike[:n]).copy()
                        active = np.asarray(net.active[:n]).copy()
                    recent = active & (last_sp > 0.0) & (last_sp >= now_t - 50.0) & (last_sp <= now_t)
                    self.metrics["natural_spike_samples"] += int(np.sum(recent))

                    monitor = getattr(net, "modules", {}).get("monitor_lenguaje") if hasattr(net, "modules") else None
                    immortal = set()
                    if monitor is not None:
                        immortal = {int(i) for i in getattr(monitor, "immortal_neurons", []) if int(i) < n}
                    if immortal:
                        self.metrics["immortal_spike_samples"] += sum(1 for i in immortal if recent[i])
                except Exception:
                    pass

            # --- Telemetría de readout neuronal (FIX 1.4.9) ---
            if net is not None:
                try:
                    _lang = getattr(net, "modules", {}).get("lenguaje") if hasattr(net, "modules") else None
                    if _lang is not None and hasattr(_lang, "get_readout_telemetry"):
                        rt = _lang.get_readout_telemetry(net, t=now_t)
                        event["readout"] = rt
                        if self._active_trial is not None:
                            tr = self._active_trial
                            tr["readout_frames"] += 1
                            tr_rt = _lang.get_trial_readout_activity() if hasattr(_lang, "get_trial_readout_activity") else {}
                            tr["readout_sources_fired_sum"] = int(tr_rt.get("source_spikes", 0))
                            tr["readout_source_spikes"] = int(tr_rt.get("source_spikes", 0))
                            tr["readout_source_unique"] = int(tr_rt.get("source_unique", 0))
                            tr["readout_pool_spikes"] = {
                                _s: int(tr_rt.get("pool_spikes", {}).get(_s, 0))
                                for _s in getattr(_lang, "symbols", [])
                            }
                            tr["readout_pool_unique"] = {
                                _s: int(tr_rt.get("pool_unique", {}).get(_s, 0))
                                for _s in getattr(_lang, "symbols", [])
                            }
                            for _s in getattr(_lang, "symbols", []):
                                tr["readout_weight_mean_last"][_s] = rt.get(f"readout_{_s}_weight_mean", 0.0)
                                tr["readout_weight_delta_last"][_s] = rt.get(f"readout_{_s}_weight_delta", 0.0)
                                tr.setdefault("readout_synapse_stats", {})[_s] = {
                                    "count": int(rt.get(f"readout_{_s}_synapse_count", 0)),
                                    "min": float(rt.get(f"readout_{_s}_weight_min", 0.0)),
                                    "max": float(rt.get(f"readout_{_s}_weight_max", 0.0)),
                                    "std": float(rt.get(f"readout_{_s}_weight_std", 0.0)),
                                }
                            if rt.get("learning_audit"):
                                tr["readout_learning_audit"] = deepcopy(rt.get("learning_audit"))
                except Exception:
                    pass

            self._events.append({"id": self._event_counter, **event})
            self._event_counter += 1
            self._events_total += 1
            if outputs:
                self.metrics["blocks_seen"] = len({k for e in self._events for k in e.get("blocks", {}).keys()})

            if self._event_counter % self.flush_every == 0:
                # No hace I/O bajo el lock. Solo marcamos que hay que guardar.
                pass

        if self._event_counter % self.flush_every == 0:
            self.flush()

    # ------------------------------------------------------------------
    # ANALISIS
    # ------------------------------------------------------------------

    @staticmethod
    def _jaccard(a, b):
        sa, sb = set(a or []), set(b or [])
        if not sa and not sb:
            return 1.0
        return len(sa & sb) / float(max(1, len(sa | sb)))

    def _vision_diagnostics(self, trials):
        # El contador interno se elimina al cerrar el trial; la existencia del
        # bloque y de una traza distingue los trials ya instrumentados.
        rows = [tr.get("vision_telemetry") for tr in trials if tr.get("complete_frames") and tr.get("vision_telemetry")]
        if not rows:
            return {"status": "insufficient_data", "trials_measured": 0}
        def mean(key, default=0.0):
            vals = [float(r.get(key, default) or default) for r in rows]
            return float(np.mean(vals)) if vals else default
        by_class = {}
        for tr in trials:
            vt = tr.get("vision_telemetry")
            if not tr.get("complete_frames") or not vt:
                continue
            label = str(tr.get("ground_truth"))
            bucket = by_class.setdefault(label, [])
            bucket.append(vt)
        per_class = {}
        for label, group in by_class.items():
            per_class[label] = {
                "trials": len(group),
                "mean_start_distance_pre_trial": float(np.mean([math.hypot(float(x.get("fovea_start_pre_trial", {}).get("x", 0.5)) - float(x.get("target", {}).get("x", 0.5)), float(x.get("fovea_start_pre_trial", {}).get("y", 0.5)) - float(x.get("target", {}).get("y", 0.5))) for x in group])),
                "mean_ratio_in_target": float(np.mean([float(x.get("ratio_in_target", 0.0)) for x in group])),
                "mean_path_length": float(np.mean([float(x.get("path_length", 0.0)) for x in group])),
                "mean_saccade_count": float(np.mean([float(x.get("saccade_count", 0.0)) for x in group])),
                "mean_fixation_count": float(np.mean([float(x.get("fixation_count", 0.0)) for x in group])),
            }
        return {
            "status": "ok",
            "trials_measured": len(rows),
            "mean_ratio_in_target": mean("ratio_in_target"),
            "mean_path_length": mean("path_length"),
            "mean_saccade_count": mean("saccade_count"),
            "mean_fixation_count": mean("fixation_count"),
            "mean_distance_to_target": mean("mean_distance_to_target"),
            "per_class": per_class,
        }

    def _pool_health_diagnostics(self, trials):
        labels = list(getattr(self.net, "symbols", []) or self.SYMBOLS)
        out = {}
        complete = [tr for tr in trials if tr.get("complete_frames")]
        for sym in labels:
            target_rows = [tr for tr in complete if tr.get("ground_truth") == sym and tr.get("readout_pool_spikes") is not None]
            non_target_rows = [tr for tr in complete if tr.get("ground_truth") != sym and tr.get("readout_pool_spikes") is not None]
            def avg(rows, field):
                vals = []
                for tr in rows:
                    bridge = (tr.get("latent_readout_bridge") or {}).get("pools", {}).get(sym, {})
                    if field == "spikes": vals.append(float((tr.get("readout_pool_spikes") or {}).get(sym, 0)))
                    else: vals.append(float(bridge.get(field, 0.0)))
                return float(np.mean(vals)) if vals else 0.0
            false_rate = float(np.mean([int((tr.get("readout_pool_spikes") or {}).get(sym, 0) > 0) for tr in non_target_rows])) if non_target_rows else 0.0
            out[sym] = {
                "target_trials": len(target_rows),
                "non_target_trials": len(non_target_rows),
                "mean_target_spikes": avg(target_rows, "spikes"),
                "mean_non_target_spikes": avg(non_target_rows, "spikes"),
                "non_target_false_activation_rate": false_rate,
                "mean_drive_ema_target": avg(target_rows, "pool_drive_ema"),
                "mean_drive_ema_non_target": avg(non_target_rows, "pool_drive_ema"),
                "mean_inhibition_target": avg(target_rows, "pool_inhibition"),
                "mean_inhibition_non_target": avg(non_target_rows, "pool_inhibition"),
            }
        return {"status": "ok" if complete else "insufficient_data", "per_symbol": out}

    def _plasticity_diagnostics(self, trials):
        rows = [tr.get("readout_learning_audit") for tr in trials if tr.get("complete_frames") and tr.get("readout_learning_audit")]
        if not rows:
            return {"status": "insufficient_data", "trials_measured": 0}
        return {
            "status": "ok",
            "trials_measured": len(rows),
            "mean_synapses_changed": float(np.mean([float(x.get("synapses_changed_exact", x.get("changed_reported", 0))) for x in rows])),
            "mean_sum_abs_delta": float(np.mean([float(x.get("sum_abs_delta", 0.0)) for x in rows])),
            "mean_max_abs_delta": float(np.mean([float(x.get("max_abs_delta", 0.0)) for x in rows])),
        }

    def _representation_diagnostics(self, trials):
        rows = [tr for tr in trials if tr.get("complete_frames") and tr.get("cortical_source_indices")]
        by_label = defaultdict(list)
        for tr in rows:
            by_label[str(tr.get("ground_truth"))].append(tr)
        same, different = [], []
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                jac = self._jaccard(rows[i].get("cortical_source_indices"), rows[j].get("cortical_source_indices"))
                if rows[i].get("ground_truth") == rows[j].get("ground_truth"):
                    same.append(jac)
                else:
                    different.append(jac)
        per_class = {}
        for label, group in by_label.items():
            uniques = [len(set(tr.get("cortical_source_indices", []))) for tr in group]
            spikes = [int(sum((tr.get("cortical_source_spike_counts") or {}).values())) for tr in group]
            per_class[label] = {
                "trials": len(group),
                "mean_unique_sources": float(np.mean(uniques)) if uniques else 0.0,
                "mean_source_spikes": float(np.mean(spikes)) if spikes else 0.0,
            }
        return {
            "status": "ok" if rows else "insufficient_data",
            "complete_trials_measured": len(rows),
            "mean_pairwise_jaccard_same_label": float(np.mean(same)) if same else None,
            "mean_pairwise_jaccard_different_label": float(np.mean(different)) if different else None,
            "pair_count_same_label": len(same),
            "pair_count_different_label": len(different),
            "per_class": per_class,
        }

    def _frame_timing_diagnostics(self, trials):
        rows = [tr for tr in trials if tr.get("complete_frames") and tr.get("frame_timing_summary")]
        summaries = [tr["frame_timing_summary"] for tr in rows]
        sim_ms = [s["simulation_ms_per_wall_second"] for s in summaries if s.get("simulation_ms_per_wall_second") is not None]
        return {
            "status": "ok" if rows else "insufficient_data",
            "complete_trials_measured": len(rows),
            "mean_simulation_ms_per_wall_second": float(np.mean(sim_ms)) if sim_ms else None,
            "median_simulation_ms_per_wall_second": float(np.median(sim_ms)) if sim_ms else None,
            "mean_frame_simulation_duration_ms": float(np.mean([s["simulation_duration_ms_median"] for s in summaries])) if summaries else None,
            "mean_frame_wall_duration_s": float(np.mean([s["wall_duration_s_median"] for s in summaries])) if summaries else None,
            "max_queue_before": max([s["queue_peak_before"] for s in summaries if s.get("queue_peak_before") is not None], default=None),
            "max_queue_after": max([s["queue_peak_after"] for s in summaries if s.get("queue_peak_after") is not None], default=None),
        }

    def _trial_causal_summary(self, trial: dict) -> dict:
        """Genera pistas de causalidad observacional, no una prueba definitiva."""
        gt = trial.get("ground_truth")
        pred = trial.get("prediction")
        scores = trial.get("evidence_peak", {})
        winner = max(scores, key=scores.get) if scores else None

        first_times = {
            k: v for k, v in trial.get("evidence_first_t", {}).items() if v is not None
        }
        sorted_times = sorted(first_times.items(), key=lambda kv: kv[1])

        certainty = self._safe_float(trial.get("certainty", 0.0))
        evidence_total = sum(max(0.0, self._safe_float(v)) for v in scores.values())
        readout_sources_fired = float(trial.get("readout_source_spikes", trial.get("readout_sources_fired_sum", 0)) or 0)
        readout_pool_spikes = trial.get("readout_pool_spikes", {}) or {}
        readout_pool_total = sum(max(0, int(v or 0)) for v in readout_pool_spikes.values())
        learned_readout_active = bool(readout_sources_fired > 0 or readout_pool_total > 0)
        valid_observation = bool(
            trial.get("frames", 0) > 0 and
            (learned_readout_active or evidence_total > 0.0)
        )
        prediction_match = bool(gt is not None and pred == gt)

        summary = {
            "trial_id": trial.get("id"),
            "ground_truth": gt,
            "prediction": pred,
            "certainty": certainty,
            "evidence_winner": winner,
            "valid_observation": valid_observation,
            "prediction_match": prediction_match,
            "learned_readout_active": learned_readout_active,
            "learned_readout_valid": bool(
                trial.get("prediction_source") != "learned_readout" or learned_readout_active
            ),
            "correct_prediction": bool(
                valid_observation and prediction_match
            ),
            "evidence_selectivity": self._selectivity(scores),
            "first_evidence_order": sorted_times,
            "candidate": None,
        }

        # Candidato observacional: solo si la observación fue válida.
        # Readout telemetría (FIX 1.4.9)
        summary["readout_sources_fired_avg"] = round(
            trial.get("readout_sources_fired_sum", 0) / max(1, trial.get("readout_frames", 1)), 2
        )
        summary["readout_pool_spikes"] = dict(trial.get("readout_pool_spikes", {}))
        summary["readout_weight_delta"] = dict(trial.get("readout_weight_delta_last", {}))
        detector_peak = dict(trial.get("detector_scores_peak", {}) or {})
        summary["detector_scores_peak"] = detector_peak
        summary["detector_prediction"] = max(detector_peak, key=detector_peak.get) if detector_peak else None
        summary["readout_source_spikes"] = int(readout_sources_fired)
        summary["readout_source_unique"] = int(trial.get("readout_source_unique", 0) or 0)
        summary["readout_pool_unique"] = dict(trial.get("readout_pool_unique", {}) or {})

        if (summary["valid_observation"] and pred and winner == pred
                and summary["evidence_selectivity"] >= 0.45):
            summary["candidate"] = {
                "type": "candidate_downstream_evidence",
                "symbol": pred,
                "reason": "evidencia dominante coincide con la predicción",
                "next_test": f"lesionar temporalmente evidencia_{pred} y repetir el mismo estímulo",
            }
        return summary

    @staticmethod
    def _selectivity(scores: Dict[str, float]) -> float:
        if not scores:
            return 0.0
        values = sorted([max(0.0, float(v)) for v in scores.values()], reverse=True)
        total = sum(values)
        if total <= 1e-9:
            return 0.0
        return values[0] / total

    def _curriculum_diagnostics(self, trials):
        session_counts = {sym: 0 for sym in (getattr(self.net, "symbols", None) or self.SYMBOLS)}
        for tr in trials:
            if tr.get("complete_frames") and tr.get("ground_truth") in session_counts:
                session_counts[str(tr.get("ground_truth"))] += 1
        persistent = getattr(self.net, "curriculum_state", {}) if self.net is not None else {}
        persistent_counts = dict((persistent or {}).get("total_trials", {}) or {})
        return {
            "session_counts": session_counts,
            "persistent_counts": {str(k): int(v) for k, v in persistent_counts.items()},
            "unlocked": list((persistent or {}).get("unlocked", []) or []),
            "semantics": "session_counts = trials completos de esta ejecución; persistent_counts = acumulado restaurado/guardado",
        }

    def summarize(self) -> dict:
        with self._lock:
            trials = list(self._trials)
            candidates = []
            _seen_candidates: set = set()
            correct = 0
            for tr in trials:
                s = self._trial_causal_summary(tr)
                if s.get("candidate"):
                    c = s["candidate"]
                    _key = (c.get("type"), c.get("symbol"))
                    if _key not in _seen_candidates:
                        _seen_candidates.add(_key)
                        candidates.append(c)
                correct += int(s.get("correct_prediction", False))

            now = time.time()
            accuracy = (correct / len(trials)) if trials else 0.0
            complete_trials = [tr for tr in trials if tr.get("complete_frames")]
            complete_correct = sum(
                1 for tr in complete_trials
                if self._trial_causal_summary(tr).get("correct_prediction", False)
            )

            # v1.8: métricas por clase + matriz de confusión. Esto separa un
            # accuracy global aparentemente bueno de un colapso X→O/O→X.
            labels = list(getattr(self.net, "symbols", []) or [])
            if not labels:
                labels = sorted({
                    str(tr.get("ground_truth")) for tr in complete_trials
                    if tr.get("ground_truth") is not None
                })
            termination_counts = {}
            for tr in trials:
                reason = str(tr.get("termination_reason") or ("completed" if tr.get("complete_frames") else "unknown"))
                termination_counts[reason] = int(termination_counts.get(reason, 0)) + 1

            per_class = {}
            confusion = {gt: {pred: 0 for pred in labels} for gt in labels}
            for gt in labels:
                rows = [tr for tr in complete_trials if tr.get("ground_truth") == gt]
                correct_cls = sum(1 for tr in rows if self._trial_causal_summary(tr).get("correct_prediction", False))
                certs = [self._safe_float(tr.get("certainty", 0.0)) for tr in rows]
                pred_counts = {}
                for tr in rows:
                    pred = tr.get("prediction")
                    if pred is not None:
                        pred_counts[str(pred)] = pred_counts.get(str(pred), 0) + 1
                        if str(pred) in confusion[gt]:
                            confusion[gt][str(pred)] += 1
                per_class[gt] = {
                    "total": len(rows),
                    "correct": correct_cls,
                    "accuracy": (correct_cls / len(rows)) if rows else 0.0,
                    "mean_certainty": (sum(certs) / len(certs)) if certs else 0.0,
                    "predictions": pred_counts,
                }

            # Recoger salud ANTES de construir conclusiones.
            # v0.9: evita UnboundLocalError cuando la telemetría se consulta
            # desde summarize() en una ruta de guardado temprana.
            queue_health: dict = {}
            queue_health_error = None
            if self.net is not None and hasattr(self.net, "get_event_queue_stats"):
                try:
                    queue_health = self.net.get_event_queue_stats() or {}
                except Exception as exc:
                    queue_health_error = f"{type(exc).__name__}: {exc}"
                    queue_health = {}

            energy_health: dict = {}
            energy_health_error = None
            if self.net is not None and hasattr(self.net, "energy"):
                try:
                    arr = np.asarray(self.net.energy[:self.net.n], dtype=float)
                    active = np.asarray(self.net.active[:self.net.n], dtype=bool)
                    vals = arr[active] if np.any(active) else arr
                    if vals.size:
                        energy_health = {
                            "active_mean": float(np.mean(vals)),
                            "active_min": float(np.min(vals)),
                            "negative_count": int(np.sum(vals < 0.0)),
                            "nonfinite_count": int(np.sum(~np.isfinite(vals))),
                        }
                except Exception as exc:
                    energy_health_error = f"{type(exc).__name__}: {exc}"
                    energy_health = {}

            region_energy_health: dict = {}
            region_energy_error = None
            if self.net is not None and hasattr(self.net, "regions"):
                try:
                    region_vals = [float(getattr(r, "energy", 0.0)) for r in self.net.regions.values()]
                    if region_vals:
                        arr_r = np.asarray(region_vals, dtype=float)
                        region_energy_health = {
                            "region_count": int(arr_r.size),
                            "mean": float(np.mean(arr_r)),
                            "min": float(np.min(arr_r)),
                            "max": float(np.max(arr_r)),
                            "negative_count": int(np.sum(arr_r < 0.0)),
                            "nonfinite_count": int(np.sum(~np.isfinite(arr_r))),
                        }
                except Exception as exc:
                    region_energy_error = f"{type(exc).__name__}: {exc}"
                    region_energy_health = {}

            # Estado explícito para consumidores externos/otras IAs.
            open_questions = []
            if not trials:
                open_questions.append("No hay trials suficientes para evaluar discriminación.")
            elif correct == 0:
                open_questions.append("La predicción correcta aún no está demostrada en los trials registrados.")
            if self._active_trial is not None:
                open_questions.append("Existe un trial activo; el archivo representa un estado intermedio.")
            if all(
                self._trial_causal_summary(tr).get("candidate") is None
                for tr in trials
            ):
                open_questions.append("No hay candidatos causales observacionales validados por el criterio actual.")
            qsize = queue_health.get("queue_size") if isinstance(queue_health, dict) else None
            if isinstance(qsize, (int, float)) and qsize >= 18000:
                open_questions.append("La cola de eventos presenta presión alta; medir producción frente a consumo antes de sesiones largas.")
            if isinstance(qsize, (int, float)) and qsize >= 22000:
                open_questions.append("La cola de eventos alcanzó/alcanzó el high watermark; la sesión requiere revisión del backpressure.")
            if energy_health.get("negative_count", 0) > 0:
                open_questions.append("Se observaron energías neuronales negativas; requieren atribución temporal antes de interpretar aprendizaje.")
            if region_energy_health.get("negative_count", 0) > 0:
                open_questions.append("Se observaron energías regionales negativas; verificar metabolismo regional antes de interpretar fitness.")

            confirmed = []
            if self.metrics["frames"] > 0:
                confirmed.append("La sonda está registrando frames y métricas de bloques.")
            if self.metrics["blocks_seen"] > 0:
                confirmed.append("Se observaron salidas de bloques visuales.")
            if any(v.get("evidence_active_frames", 0) > 0 for v in self._symbol_acc.values()):
                confirmed.append("Se observó evidencia lingüística por encima del umbral de actividad.")

            telemetry_status = "ok"
            telemetry_errors = {}
            if queue_health_error:
                telemetry_status = "degraded"
                telemetry_errors["event_queue"] = queue_health_error
            if energy_health_error:
                telemetry_status = "degraded"
                telemetry_errors["energy"] = energy_health_error
            if region_energy_error:
                telemetry_status = "degraded"
                telemetry_errors["region_energy"] = region_energy_error

            return {
                "schema_version": self.SCHEMA_VERSION,
                "producer": {
                    "project": self.PROJECT_VERSION,
                    "component": "CausalProbe",
                    "version": self.VERSION,
                },
                "session": {
                    "session_id": self.session_id,
                    "started_at": self.started_at,
                    "started_at_iso_utc": self._iso_utc(self.started_at),
                    "generated_at": now,
                    "generated_at_iso_utc": self._iso_utc(now),
                    "event_counter": self._event_counter,
                    "events_retained": len(self._events),
                    "events_total": self._events_total,
                    "events_dropped_from_ring_buffer": max(0, self._events_total - len(self._events)),
                },
                "runtime_contract": self._runtime_contract(self.net),
                "state": {
                    "enabled": bool(self.enabled),
                    "active_trial": deepcopy(self._active_trial),
                    "simulation_time": self._safe_float(getattr(self.net, "current_time", self.metrics.get("last_update", 0.0))) if self.net is not None else None,
                    "last_block_times": dict(self._last_block),
                    "last_evidence": dict(self._last_evidence),
                },
                "metrics": deepcopy(self.metrics),
                "health": {
                    "status": telemetry_status,
                    "errors": telemetry_errors,
                    "event_queue": queue_health,
                    "energy": energy_health,
                    "region_energy": region_energy_health,
                },
                "telemetry_integrity": {
                    "status": telemetry_status,
                    "errors": telemetry_errors,
                    "note": "Una falla de observabilidad no debe detener la simulación ni invalidar otros campos del snapshot."
                },
                "block_accumulators": deepcopy(dict(self._block_acc)),
                "symbol_accumulators": deepcopy(dict(self._symbol_acc)),
                "representation_diagnostics": self._representation_diagnostics(trials),
                "frame_timing_diagnostics": self._frame_timing_diagnostics(trials),
                "vision_diagnostics": self._vision_diagnostics(trials),
                "pool_health_diagnostics": self._pool_health_diagnostics(trials),
                "plasticity_diagnostics": self._plasticity_diagnostics(trials),
                "curriculum": self._curriculum_diagnostics(trials),
                "trials": {
                    "total_retained": len(trials),
                    "termination_counts": termination_counts,
                    "correct": correct,
                    "accuracy": accuracy,
                    "evaluated_total": len(complete_trials),
                    "evaluated_correct": complete_correct,
                    "evaluated_accuracy": (complete_correct / len(complete_trials)) if complete_trials else 0.0,
                    "complete_total": len(complete_trials),
                    "complete_correct": complete_correct,
                    "complete_accuracy": (complete_correct / len(complete_trials)) if complete_trials else 0.0,
                    "incomplete_total": len(trials) - len(complete_trials),
                    "per_class": per_class,
                    "confusion_matrix": confusion,
                    "recent": trials[-50:],
                },
                "causal_analysis": {
                    "status": "observational_only",
                    "candidates": candidates[-50:],
                    "limitations": [
                        "La precedencia temporal no demuestra causalidad por sí sola.",
                        "Los resultados dependen del protocolo y de la configuración vigente.",
                        "Una intervención controlada/repetición emparejada es necesaria para elevar el estatus causal.",
                    ],
                },
                "collaboration": {
                    "role": "shared_state_contract",
                    "read_rule": "Separar hechos observados, inferencias y acciones pendientes.",
                    "authoritative_fields": [
                        "schema_version", "session", "runtime_contract", "state",
                        "metrics", "block_accumulators", "symbol_accumulators",
                        "representation_diagnostics", "frame_timing_diagnostics",
                        "trials", "health", "telemetry_integrity", "causal_analysis"
                    ],
                    "confirmed_findings": confirmed,
                    "open_questions": open_questions,
                    "next_actions": [
                        "Repetir trials completos y balanceados por símbolo.",
                        "Comparar ensayos con intervención contra controles emparejados.",
                        "Actualizar el estado solo a partir de nuevas mediciones, no de inferencias textuales."
                    ],
                },
            }

    def flush(self, path: Optional[str] = None) -> bool:
        """Guarda telemetría compacta; una falla de observabilidad no interrumpe la simulación."""
        target = path or self.output_path
        tmp = f"{target}.tmp"
        try:
            payload = self.summarize()
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, default=list)
            os.replace(tmp, target)
            return True
        except Exception as exc:
            # Intento de último recurso: conservar lo máximo posible para que
            # otra IA pueda ver que el snapshot quedó degradado en vez de perderlo.
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            print(f"⚠️ [CausalProbe] No se pudo guardar telemetría: {type(exc).__name__}: {exc}", flush=True)
            return False

    def get_live_snapshot(self) -> dict:
        with self._lock:
            return {
                "metrics": deepcopy(self.metrics),
                "last_block": dict(self._last_block),
                "last_evidence": dict(self._last_evidence),
                "active_trial": deepcopy(self._active_trial),
            }
