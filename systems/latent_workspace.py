"""IA 1.6.29f - Latent Workspace v7.

Workspace latente post-percepción, deliberadamente aislado del Event Queue.

Cambios respecto de 1.6.28c y evidencia de 1.6.28b:
- S (memoria contextual) y H (workspace) son estados independientes.
- La codificación temporal usa bins canónicos y no depende de la duración observada.
- H tiene decoder persistente propio; ya no se reconstruye en cada trial desde los pesos del readout.
- Decoder y operador recurrente/contextual admiten aprendizaje supervisado local de tres factores.
- Las actualizaciones de decoder/recurrente/contexto se agregan primero y se aplican una sola vez con límites de norma.
- Se registran innovación, ganancia de margen, mejor etapa y criterio de parada por iteración.
- Se incluyen ablaciones C-only / S-only / C+S y una separación explícita train/eval del aprendizaje latente.
- Se conserva determinismo, sparsity y límites de energía/estado.
- La selección de crédito recurrente deja de depender del mayor margen y pasa a depender de mejora real de pérdida frente a H0.
- H0 queda como baseline operacional; el recurrente sólo recibe crédito cuando una etapa posterior demuestra utilidad sobre H0.
- Early-exit termina el refinamiento cuando no aporta utilidad observable, sin forzar H4.
- El decoder puede inicializarse de forma neutral y ortogonal; la siembra desde readout queda desactivada por defecto.
- Se incorpora auditoría de colapso temporal y un probe de prototipos de H0 puramente observacional.
- El workspace sigue siendo agnóstico a clases durante la inferencia; las etiquetas entran sólo por la función de aprendizaje/decoder y el probe de memoria.
- No programa spikes, no usa receive_spike y no toca Event Queue.
- H0 queda como referencia persistente de inferencia; las etapas posteriores son candidatas, no sustitutas automáticas.
- Se incorpora replay compacto de prototipos H0 para reducir olvido catastrófico del decoder.
- Se incorpora un decoder temporal shadow de 3 bins, siempre no autoritativo.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional

import numpy as np


class LatentWorkspace:
    """Memoria contextual S + workspace recurrente H aprendible y acotado."""

    VERSION = "1.6.29f-latent-workspace-v7"
    STATE_VERSION = 7

    def __init__(
        self,
        input_sources: int = 96,
        temporal_bins: int = 3,
        hidden_dim: int = 96,
        steps: int = 4,
        recurrent_density: float = 0.08,
        recurrent_gain: float = 0.62,
        context_gain: float = 0.35,
        state_mix: float = 0.50,
        context_update_rate: float = 0.65,
        top_k: int = 24,
        seed: int = 162800,
        decoder_learning_rate: float = 0.08,
        recurrent_learning_rate: float = 0.004,
        context_learning_rate: float = 0.006,
        min_step_mix: float = 0.20,
        halt_delta: float = 0.015,
        weight_clip: float = 0.40,
        decoder_max_update_norm: float = 0.35,
        recurrent_max_update_norm: float = 0.08,
        context_max_update_norm: float = 0.08,
        halt_margin_gain: float = 0.0015,
        halt_innovation: float = 0.005,
        halt_patience: int = 1,
        stage_min_loss_gain: float = 0.005,
        early_exit_enabled: bool = False,
        early_exit_min_margin_gain: float = 0.0015,
        early_exit_patience: int = 1,
        decoder_seed_from_readout: bool = False,
        neutral_decoder_scale: float = 0.75,
        prototype_probe_enabled: bool = True,
        prototype_update_rate: float = 0.08,
        replay_capacity: int = 5,
        replay_per_class: int = 1,
        replay_weight: float = 0.15,
        temporal_shadow_enabled: bool = True,
        temporal_shadow_learning_rate: float = 0.02,
        temporal_shadow_max_update_norm: float = 0.25,
        sleep_enabled: bool = True,
        sleep_interval: int = 25,
        sleep_nrem_passes: int = 2,
        sleep_replay_per_class: int = 2,
        sleep_replay_weight: float = 0.08,
        sleep_homeostasis_rate: float = 0.015,
        sleep_homeostasis_target_l1: float = 0.62,
        sleep_rem_like_enabled: bool = False,
        sleep_rem_like_passes: int = 1,
        sleep_rem_like_rate: float = 0.002,
        information_flow_audit: bool = True,
        information_loss_clip: float = 1.0,
        stage_reliability_decay: float = 0.08,
        stage_reliability_floor: float = 0.20,
        bridge_memory_capacity: int = 40,
    ):
        self.input_sources = int(max(1, input_sources))
        self.temporal_bins = int(max(1, temporal_bins))
        self.hidden_dim = int(max(8, hidden_dim))
        self.steps = int(max(1, steps))
        self.recurrent_density = float(np.clip(recurrent_density, 0.01, 1.0))
        self.recurrent_gain = float(np.clip(recurrent_gain, 0.0, 0.95))
        self.context_gain = float(np.clip(context_gain, 0.0, 0.95))
        self.state_mix = float(np.clip(state_mix, 0.05, 0.95))
        self.context_update_rate = float(np.clip(context_update_rate, 0.05, 1.0))
        self.top_k = int(max(1, min(top_k, self.hidden_dim)))
        self.seed = int(seed)
        self.decoder_learning_rate = float(max(0.0, decoder_learning_rate))
        self.recurrent_learning_rate = float(max(0.0, recurrent_learning_rate))
        self.context_learning_rate = float(max(0.0, context_learning_rate))
        self.min_step_mix = float(np.clip(min_step_mix, 0.05, 1.0))
        self.halt_delta = float(max(1e-5, halt_delta))
        self.weight_clip = float(max(0.05, weight_clip))
        self.decoder_max_update_norm = float(max(1e-6, decoder_max_update_norm))
        self.recurrent_max_update_norm = float(max(1e-6, recurrent_max_update_norm))
        self.context_max_update_norm = float(max(1e-6, context_max_update_norm))
        self.halt_margin_gain = float(max(0.0, halt_margin_gain))
        self.halt_innovation = float(max(0.0, halt_innovation))
        self.halt_patience = int(max(1, halt_patience))
        self.stage_min_loss_gain = float(max(0.0, stage_min_loss_gain))
        self.early_exit_enabled = bool(early_exit_enabled)
        self.early_exit_min_margin_gain = float(max(0.0, early_exit_min_margin_gain))
        self.early_exit_patience = int(max(1, early_exit_patience))
        self.decoder_seed_from_readout = bool(decoder_seed_from_readout)
        self.neutral_decoder_scale = float(max(0.05, neutral_decoder_scale))
        self.prototype_probe_enabled = bool(prototype_probe_enabled)
        self.prototype_update_rate = float(np.clip(prototype_update_rate, 0.0, 1.0))
        self.replay_capacity = int(max(1, replay_capacity))
        self.replay_per_class = int(max(0, replay_per_class))
        self.replay_weight = float(max(0.0, replay_weight))
        self.temporal_shadow_enabled = bool(temporal_shadow_enabled)
        self.temporal_shadow_learning_rate = float(max(0.0, temporal_shadow_learning_rate))
        self.temporal_shadow_max_update_norm = float(max(1e-6, temporal_shadow_max_update_norm))
        self.sleep_enabled = bool(sleep_enabled)
        self.sleep_interval = int(max(1, sleep_interval))
        self.sleep_nrem_passes = int(max(1, sleep_nrem_passes))
        self.sleep_replay_per_class = int(max(0, sleep_replay_per_class))
        self.sleep_replay_weight = float(max(0.0, sleep_replay_weight))
        self.sleep_homeostasis_rate = float(np.clip(sleep_homeostasis_rate, 0.0, 0.25))
        self.sleep_homeostasis_target_l1 = float(max(1e-4, sleep_homeostasis_target_l1))
        self.sleep_rem_like_enabled = bool(sleep_rem_like_enabled)
        self.sleep_rem_like_passes = int(max(0, sleep_rem_like_passes))
        self.sleep_rem_like_rate = float(max(0.0, sleep_rem_like_rate))
        self.information_flow_audit = bool(information_flow_audit)
        self.information_loss_clip = float(max(0.0, information_loss_clip))
        self.stage_reliability_decay = float(np.clip(stage_reliability_decay, 0.0, 1.0))
        self.stage_reliability_floor = float(np.clip(stage_reliability_floor, 0.0, 1.0))
        self.bridge_memory_capacity = int(max(1, bridge_memory_capacity))

        self._rng = np.random.default_rng(self.seed)
        self._input_projection = self._make_sparse_matrix(
            self.hidden_dim,
            self.input_sources * self.temporal_bins,
            density=min(0.20, max(0.02, 18.0 / max(1, self.input_sources * self.temporal_bins))),
            row_l1=0.95,
            positive=True,
        )
        self._recurrent = self._make_sparse_matrix(
            self.hidden_dim,
            self.hidden_dim,
            density=self.recurrent_density,
            row_l1=self.recurrent_gain,
            positive=True,
        )
        self._context_projection = self._make_sparse_matrix(
            self.hidden_dim,
            self.hidden_dim,
            density=min(self.recurrent_density, 0.10),
            row_l1=self.context_gain,
            positive=True,
        )

        # Decoder propio y persistente. Se inicializa de forma neutral y luego,
        # una sola vez, puede alinearse con el readout existente del sistema.
        self._decoder: Optional[np.ndarray] = None
        self._decoder_initialized = False
        self._decoder_seeded_from_readout = False
        self._decoder_bias: Optional[np.ndarray] = None

        self.context_state = np.zeros(self.hidden_dim, dtype=np.float32)
        self.workspace_state = np.zeros(self.hidden_dim, dtype=np.float32)
        self.last_states: list[np.ndarray] = []
        self.last_stage_scores: Dict[str, Dict[str, float]] = {}
        self.last_stage_predictions: list[str | None] = []
        self.last_stage_margins: Dict[str, float] = {}
        self.last_metrics: Dict[str, float | int | bool] = {}
        self.last_learning: Dict[str, object] = {}
        self._last_direct_state = np.zeros(self.hidden_dim, dtype=np.float32)
        self._last_context_only_state = np.zeros(self.hidden_dim, dtype=np.float32)
        self._last_stage_innovation: Dict[str, float] = {}
        self._last_stage_margin_gain: Dict[str, float] = {}
        self._last_halt_reason = ""
        self._last_temporal_audit: Dict[str, object] = {}
        self._last_prototype_probe: Dict[str, object] = {}
        self._last_learning_stage = "H0"
        self._last_learning_selection_reason = "baseline_h0"
        self._last_temporal_shadow = {}
        self._last_accepted_stage = "H0"
        self._last_raw_best_stage = "H0"
        self._prototypes: Dict[str, np.ndarray] = {}
        self._prototype_counts: Dict[str, int] = {}
        self._prototype_memory: Dict[str, list[np.ndarray]] = {}
        # Memoria episódica compacta: conserva la secuencia H0..Hn de trials
        # completos para replay, no sólo el último estado H0.
        self._episode_memory: Dict[str, list[list[np.ndarray]]] = {}
        # Fiabilidad histórica del stage, actualizada sólo con trials etiquetados.
        self._stage_reliability: Dict[str, float] = {}
        # Memoria compacta de fallos del puente para priorizar replay.
        self._bridge_failure_memory: list[dict[str, object]] = []
        self._sleep_trials_since_last = 0
        self._sleep_cycles = 0
        self._last_sleep: Dict[str, object] = {}
        self._last_information_flow: Dict[str, object] = {}
        self._temporal_shadow_decoder: Optional[np.ndarray] = None
        self._temporal_shadow_bias: Optional[np.ndarray] = None
        self._last_temporal_shadow: Dict[str, object] = {}
        self._last_accepted_stage = "H0"
        self._last_raw_best_stage = "H0"
        self.trial_runs = 0
        self.learning_steps = 0

    def _make_sparse_matrix(
        self,
        rows: int,
        cols: int,
        density: float,
        row_l1: float,
        positive: bool = True,
    ) -> np.ndarray:
        matrix = np.zeros((rows, cols), dtype=np.float32)
        nnz = max(1, int(round(cols * float(np.clip(density, 0.01, 1.0)))))
        for r in range(rows):
            picks = self._rng.choice(cols, size=min(nnz, cols), replace=False)
            vals = self._rng.random(len(picks), dtype=np.float32) + 0.15
            if not positive:
                vals *= np.where(self._rng.random(len(picks)) >= 0.5, 1.0, -1.0)
            vals /= max(1e-6, float(np.sum(np.abs(vals))))
            vals *= float(row_l1)
            matrix[r, picks] = vals
        return matrix

    @staticmethod
    def _safe_normalize(vec: np.ndarray) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if not math.isfinite(norm) or norm <= 1e-8:
            return np.zeros_like(arr, dtype=np.float32)
        return (arr / norm).astype(np.float32)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        aa = np.asarray(a, dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)
        na = float(np.linalg.norm(aa))
        nb = float(np.linalg.norm(bb))
        if na <= 1e-12 or nb <= 1e-12:
            return 0.0
        value = float(np.dot(aa, bb) / (na * nb))
        return float(np.clip(value, -1.0, 1.0))

    @staticmethod
    def _clip_update_norm(update: np.ndarray, max_norm: float) -> np.ndarray:
        arr = np.asarray(update, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        limit = float(max(1e-8, max_norm))
        if not math.isfinite(norm) or norm <= limit:
            return arr.copy()
        return (arr * float(limit / norm)).astype(np.float32)

    @staticmethod
    def _sparsify(vec: np.ndarray, top_k: int) -> np.ndarray:
        out = np.maximum(0.0, np.asarray(vec, dtype=np.float32).copy())
        k = min(max(1, int(top_k)), out.size)
        if k < out.size:
            keep = np.argpartition(out, -k)[-k:]
            mask = np.zeros(out.size, dtype=bool)
            mask[keep] = True
            out[~mask] = 0.0
        return out

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        x = np.asarray(logits, dtype=np.float32)
        if x.size == 0:
            return np.zeros_like(x)
        z = x - float(np.max(x))
        ex = np.exp(np.clip(z, -30.0, 30.0)).astype(np.float32)
        s = float(np.sum(ex))
        if not math.isfinite(s) or s <= 1e-8:
            return np.full(x.shape, 1.0 / max(1, x.size), dtype=np.float32)
        return (ex / s).astype(np.float32)

    @staticmethod
    def _entropy(prob: np.ndarray) -> float:
        p = np.clip(np.asarray(prob, dtype=np.float64), 1e-9, 1.0)
        return float(-np.sum(p * np.log(p)))

    @staticmethod
    def _top_indices(vec: np.ndarray, k: int) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float32)
        if arr.size <= 0:
            return np.asarray([], dtype=np.int64)
        k = min(max(1, int(k)), arr.size)
        idx = np.argpartition(np.abs(arr), -k)[-k:]
        return idx[np.argsort(-np.abs(arr[idx]))]

    def _ensure_temporal_shadow_decoder(self, n_classes: int) -> Optional[np.ndarray]:
        if not self.temporal_shadow_enabled or n_classes <= 0:
            return None
        shape = (n_classes, self.input_sources * self.temporal_bins)
        if self._temporal_shadow_decoder is not None and self._temporal_shadow_decoder.shape == shape:
            return self._temporal_shadow_decoder
        raw = self._rng.normal(size=shape)
        row_norm = np.linalg.norm(raw, axis=1, keepdims=True)
        raw = raw / np.maximum(row_norm, 1e-8)
        self._temporal_shadow_decoder = (raw * float(self.neutral_decoder_scale)).astype(np.float32)
        self._temporal_shadow_bias = np.zeros(n_classes, dtype=np.float32)
        return self._temporal_shadow_decoder

    def _decode_temporal_shadow(self, source_bin_counts: Dict[int, Iterable[float]], symbols: Iterable[str]) -> Dict[str, object]:
        syms = [str(x) for x in symbols]
        decoder = self._ensure_temporal_shadow_decoder(len(syms))
        feature = self._safe_normalize(self._build_temporal_tensor(source_bin_counts))
        if decoder is None or not np.any(feature):
            return {
                "enabled": bool(self.temporal_shadow_enabled),
                "available": False,
                "prediction": None,
                "margin": 0.0,
                "scores": {},
            }
        logits = decoder @ feature
        if self._temporal_shadow_bias is not None and self._temporal_shadow_bias.size == len(syms):
            logits = logits + self._temporal_shadow_bias
        probs = self._softmax(logits)
        scores = {sym: float(probs[i]) for i, sym in enumerate(syms)}
        pred, margin = self._winner(scores)
        return {
            "enabled": True,
            "available": True,
            "prediction": pred,
            "margin": float(margin),
            "scores": scores,
            "feature_nonzero": int(np.sum(feature > 0.0)),
        }

    def _update_temporal_shadow(self, target: str, source_feature: np.ndarray, symbols: list[str], apply: bool) -> Dict[str, object]:
        decoder = self._ensure_temporal_shadow_decoder(len(symbols))
        if decoder is None or not np.any(source_feature) or target not in symbols:
            return {"enabled": bool(self.temporal_shadow_enabled), "applied": False, "update_norm": 0.0}
        target_idx = symbols.index(str(target))
        logits = decoder @ source_feature
        if self._temporal_shadow_bias is not None and self._temporal_shadow_bias.size == len(symbols):
            logits = logits + self._temporal_shadow_bias
        probs = self._softmax(logits)
        target_vec = np.zeros(len(symbols), dtype=np.float32)
        target_vec[target_idx] = 1.0
        grad = (target_vec - probs)[:, None] * source_feature[None, :]
        delta = self._clip_update_norm(
            (float(self.temporal_shadow_learning_rate) * grad).astype(np.float32),
            self.temporal_shadow_max_update_norm,
        )
        if apply:
            decoder[:] = np.clip(decoder + delta, -2.0, 2.0)
        return {
            "enabled": True,
            "applied": bool(apply),
            "update_norm": float(np.linalg.norm(delta)),
            "target_probability": float(probs[target_idx]),
            "changed": int(np.sum(np.abs(delta) > 1e-10)) if apply else 0,
        }

    def reset_trial(self) -> None:
        # S y H son estados propios del trial de esta fase. La persistencia que
        # interesa a 1.6.28b está en los parámetros aprendibles, no en contaminar
        # un trial nuevo con el estado de la letra anterior.
        self.context_state.fill(0.0)
        self.workspace_state.fill(0.0)
        self.last_states = []
        self.last_stage_scores = {}
        self.last_stage_predictions = []
        self.last_stage_margins = {}
        self.last_metrics = {}
        self.last_learning = {}
        self._last_temporal_audit = {}
        self._last_information_flow = {}
        self._last_prototype_probe = {}
        self._last_learning_stage = "H0"
        self._last_learning_selection_reason = "baseline_h0"

    def _build_temporal_tensor(self, source_bin_counts: Dict[int, Iterable[float]]) -> np.ndarray:
        """Devuelve un vector canónico [bin * source + source]."""
        full = np.zeros(self.input_sources * self.temporal_bins, dtype=np.float32)
        for raw_src, raw_bins in (source_bin_counts or {}).items():
            try:
                src = int(raw_src)
            except Exception:
                continue
            if src < 0 or src >= self.input_sources:
                continue
            vals = list(raw_bins) if raw_bins is not None else []
            for b in range(self.temporal_bins):
                value = float(vals[b]) if b < len(vals) else 0.0
                if math.isfinite(value) and value > 0.0:
                    full[b * self.input_sources + src] = math.log1p(value)
        return full

    def _temporal_audit(self, source_bin_counts: Dict[int, Iterable[float]]) -> Dict[str, object]:
        """Mide concentración temporal sin alterar el binning canónico."""
        totals = np.zeros(self.temporal_bins, dtype=np.float64)
        source_count = 0
        for raw_vals in (source_bin_counts or {}).values():
            vals = list(raw_vals) if raw_vals is not None else []
            used = False
            for b in range(self.temporal_bins):
                try:
                    value = float(vals[b]) if b < len(vals) else 0.0
                except Exception:
                    value = 0.0
                if math.isfinite(value) and value > 0.0:
                    totals[b] += value
                    used = True
            if used:
                source_count += 1
        total = float(np.sum(totals))
        if total <= 1e-8:
            return {
                "bin_totals": [0.0] * self.temporal_bins,
                "active_bins": 0,
                "first_active_bin": None,
                "last_active_bin": None,
                "max_bin_share": 0.0,
                "entropy_normalized": 0.0,
                "temporal_collapse": False,
                "source_count": int(source_count),
            }
        probs = totals / total
        entropy = float(-np.sum(np.where(probs > 0.0, probs * np.log(np.maximum(probs, 1e-12)), 0.0)))
        max_entropy = math.log(max(2, self.temporal_bins))
        active = np.where(totals > 0.0)[0]
        max_share = float(np.max(probs))
        return {
            "bin_totals": [float(x) for x in totals],
            "active_bins": int(active.size),
            "first_active_bin": int(active[0]) if active.size else None,
            "last_active_bin": int(active[-1]) if active.size else None,
            "max_bin_share": max_share,
            "entropy_normalized": float(entropy / max_entropy) if max_entropy > 1e-9 else 0.0,
            "temporal_collapse": bool(max_share >= 0.80 or active.size <= 1),
            "source_count": int(source_count),
        }

    def _prototype_probe_for_state(self, h: np.ndarray, symbols: Iterable[str]) -> Dict[str, object]:
        syms = [str(x) for x in symbols]
        if not self.prototype_probe_enabled or not self._prototypes:
            return {
                "enabled": bool(self.prototype_probe_enabled),
                "available": False,
                "scores": {},
                "prediction": None,
                "margin": 0.0,
            }
        scores = {}
        for sym in syms:
            proto = self._prototypes.get(sym)
            if proto is None:
                continue
            scores[sym] = float((self._cosine_similarity(h, proto) + 1.0) * 0.5)
        pred, margin = self._winner(scores)
        return {
            "enabled": True,
            "available": bool(scores),
            "scores": scores,
            "prediction": pred,
            "margin": float(margin),
            "prototype_counts": {sym: int(self._prototype_counts.get(sym, 0)) for sym in syms},
        }

    def _encode_temporal_context(self, source_bin_counts: Dict[int, Iterable[float]]) -> tuple[np.ndarray, np.ndarray, int]:
        """Construye S por bins y H0 desde C+S; conserva ablaciones C-only/S-only."""
        full = self._build_temporal_tensor(source_bin_counts)
        bins = full.reshape(self.temporal_bins, self.input_sources)
        self.context_state.fill(0.0)
        context_updates = 0

        for b_index in range(self.temporal_bins):
            x_bin = bins[b_index].copy()
            norm = float(np.linalg.norm(x_bin))
            if norm <= 1e-8:
                continue
            x_bin /= norm
            col_start = int(b_index * self.input_sources)
            col_end = int(col_start + self.input_sources)
            projection_slice = self._input_projection[:, col_start:col_end]
            projected = self._safe_normalize(np.maximum(0.0, projection_slice @ x_bin))
            projected = self._sparsify(projected, self.top_k)
            a = float(self.context_update_rate)
            self.context_state = ((1.0 - a) * self.context_state + a * projected).astype(np.float32)
            self.context_state = self._sparsify(self.context_state, self.top_k)
            self.context_state = self._safe_normalize(self.context_state)
            context_updates += 1

        direct = self._safe_normalize(np.maximum(0.0, self._input_projection @ full))
        context_only = self._safe_normalize(self.context_state.copy())
        h0 = self._safe_normalize(
            (1.0 - 0.55) * direct + 0.55 * context_only
        )
        h0 = self._sparsify(h0, self.top_k)
        h0 = self._safe_normalize(h0)
        self._last_direct_state = direct.copy()
        self._last_context_only_state = context_only.copy()
        return h0, self.context_state.copy(), context_updates

    def _ensure_decoder(self, n_classes: int, seed_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        if n_classes <= 0:
            self._decoder = np.zeros((0, self.hidden_dim), dtype=np.float32)
            self._decoder_bias = np.zeros(0, dtype=np.float32)
            self._decoder_initialized = True
            return self._decoder
        if self._decoder is not None and self._decoder.shape == (n_classes, self.hidden_dim):
            return self._decoder

        # Inicialización determinista de pequeña amplitud para evitar que las cinco
        # clases nazcan con un sesgo arbitrario grande. Si existe el readout 1.6.27,
        # calculamos UNA alineación inicial ridge desde la proyección de entrada.
        decoder = np.zeros((n_classes, self.hidden_dim), dtype=np.float32)
        if seed_matrix is not None and np.asarray(seed_matrix).ndim == 2:
            seed = np.asarray(seed_matrix, dtype=np.float32)
            if seed.shape == (n_classes, self.input_sources):
                expanded = np.tile(seed / float(max(1, self.temporal_bins)), (1, self.temporal_bins))
                p = np.asarray(self._input_projection, dtype=np.float64)
                r = np.asarray(expanded, dtype=np.float64)
                try:
                    reg = 0.15
                    gram = p @ p.T + reg * np.eye(p.shape[0], dtype=np.float64)
                    decoder = (r @ p.T @ np.linalg.inv(gram)).astype(np.float32)
                    decoder -= np.mean(decoder, axis=1, keepdims=True)
                    decoder = np.clip(decoder, -0.5, 0.5)
                    scale = float(np.median(np.linalg.norm(decoder, axis=1)))
                    if not math.isfinite(scale) or scale <= 1e-6:
                        raise ValueError("degenerate decoder seed")
                    decoder *= float(0.75 / scale)
                    self._decoder_seeded_from_readout = True
                except Exception:
                    decoder = np.zeros((n_classes, self.hidden_dim), dtype=np.float32)

        if not np.any(np.abs(decoder) > 1e-8):
            # Scaffold neutral: filas ortogonales y de escala suficiente para no
            # arrancar pegadas a la distribución casi uniforme observada en 1.6.28c.
            try:
                raw = self._rng.normal(size=(self.hidden_dim, n_classes))
                q, _ = np.linalg.qr(raw)
                basis = q[:, :n_classes].T.astype(np.float32)
                decoder = (basis * self.neutral_decoder_scale).astype(np.float32)
            except Exception:
                for i in range(n_classes):
                    start = (i * max(1, self.hidden_dim // n_classes)) % self.hidden_dim
                    idx = (start + np.arange(min(12, self.hidden_dim))) % self.hidden_dim
                    signs = np.where((np.arange(idx.size) + i) % 2 == 0, 1.0, -1.0)
                    decoder[i, idx] = (0.12 * signs).astype(np.float32)

        self._decoder = decoder.astype(np.float32)
        self._decoder_bias = np.zeros(n_classes, dtype=np.float32)
        self._decoder_initialized = True
        return self._decoder

    def _decode_pool_scores(self, h: np.ndarray, symbols: Iterable[str]) -> Dict[str, float]:
        syms = list(symbols)
        decoder = self._ensure_decoder(len(syms))
        logits = decoder @ np.asarray(h, dtype=np.float32)
        if self._decoder_bias is not None and self._decoder_bias.size == len(syms):
            logits = logits + self._decoder_bias
        prob = self._softmax(logits)
        return {str(sym): float(np.clip(prob[i], 0.0, 1.0)) for i, sym in enumerate(syms)}

    @staticmethod
    def _winner(scores: Dict[str, float]) -> tuple[Optional[str], float]:
        if not scores:
            return None, 0.0
        ordered = sorted(((str(k), float(v)) for k, v in scores.items()), key=lambda item: (-item[1], item[0]))
        if not ordered:
            return None, 0.0
        top = max(0.0, ordered[0][1])
        second = max(0.0, ordered[1][1]) if len(ordered) > 1 else 0.0
        margin = (top - second) / max(1e-8, top + second)
        return ordered[0][0], float(max(0.0, margin))

    def _run_recurrent_step(self, prev: np.ndarray) -> np.ndarray:
        recurrent_drive = self._recurrent @ prev
        context_drive = self._context_projection @ self.context_state
        drive = np.maximum(0.0, recurrent_drive + context_drive)
        transformed = np.tanh(drive).astype(np.float32)
        # Residual mixing: H no se reemplaza por completo en cada iteración.
        mix = max(self.min_step_mix, self.state_mix)
        h = ((1.0 - mix) * prev + mix * transformed).astype(np.float32)
        h = self._sparsify(h, self.top_k)
        return self._safe_normalize(h)

    def _best_stage(self, stage_scores: Dict[str, Dict[str, float]], stage_margins: Dict[str, float]) -> str:
        """Elige la mayor separación; ante empate conserva la etapa más temprana."""
        keys = list(stage_scores.keys())
        if not keys:
            return "H0"
        best_i = max(range(len(keys)), key=lambda i: (float(stage_margins.get(keys[i], 0.0)), -i))
        return keys[best_i]

    @staticmethod
    def _support_jaccard(a: np.ndarray, b: np.ndarray) -> float:
        aa = set(int(i) for i in np.where(np.asarray(a) > 0.0)[0])
        bb = set(int(i) for i in np.where(np.asarray(b) > 0.0)[0])
        if not aa and not bb:
            return 1.0
        if not aa or not bb:
            return 0.0
        return float(len(aa & bb) / max(1, len(aa | bb)))

    def _build_information_flow_audit(
        self,
        source_bin_counts: Dict[int, Iterable[float]],
        states: list[np.ndarray],
        stage_scores: Dict[str, Dict[str, float]],
        stage_predictions: list[str | None],
        stage_margins: Dict[str, float],
    ) -> Dict[str, object]:
        """Proxy de conservación de información a través de C/S -> H0 -> Hn.

        No estima información mutua literal. Mide preservación de soporte,
        predicción, margen y selectividad, junto con cambios de entropía.
        """
        if not self.information_flow_audit or not states:
            return {"enabled": bool(self.information_flow_audit), "available": False}
        syms = list(stage_scores.get("H0", {}).keys())
        h0 = states[0]
        h0_pred = stage_predictions[0] if stage_predictions else None
        h0_margin = float(stage_margins.get("H0", 0.0))
        h0_scores = np.asarray([float(stage_scores.get("H0", {}).get(s, 0.0)) for s in syms], dtype=np.float32)
        h0_prob = self._softmax(h0_scores) if h0_scores.size else np.zeros(0, dtype=np.float32)
        h0_selectivity = float(np.max(h0_prob) - np.mean(h0_prob)) if h0_prob.size else 0.0

        stage_audit = {}
        for i, state in enumerate(states):
            key = "H0" if i == 0 else f"LATENT-{i}"
            scores = np.asarray([float(stage_scores.get(key, {}).get(s, 0.0)) for s in syms], dtype=np.float32)
            prob = self._softmax(scores) if scores.size else np.zeros(0, dtype=np.float32)
            entropy = self._entropy(prob) if prob.size else 0.0
            selectivity = float(np.max(prob) - np.mean(prob)) if prob.size else 0.0
            support = self._support_jaccard(h0, state)
            pred = stage_predictions[i] if i < len(stage_predictions) else None
            margin = float(stage_margins.get(key, 0.0))
            margin_ratio = (margin / max(1e-6, h0_margin)) if h0_margin > 1e-8 else (1.0 if i == 0 else 0.0)
            prediction_preserved = bool(pred == h0_pred)
            entropy_change = float(entropy - (self._entropy(h0_prob) if h0_prob.size else 0.0))
            selectivity_change = float(selectivity - h0_selectivity)
            margin_preserved = float(np.clip(margin_ratio, 0.0, 1.0))
            pred_penalty = 0.0 if prediction_preserved else 1.0
            support_penalty = 1.0 - support
            margin_penalty = 1.0 - margin_preserved
            entropy_penalty = min(1.0, max(0.0, entropy_change / max(1e-6, math.log(max(2, len(syms)))))) if entropy_change > 0.0 else 0.0
            # Proxy discriminativo separado: nos importa si el contraste útil
            # sobrevive, aunque el soporte estructural se mantenga intacto.
            discriminative_retention = float(np.clip(
                0.50 * margin_preserved + 0.30 * (1.0 - max(0.0, -selectivity_change) / max(1e-6, h0_selectivity + 1e-6)) +
                0.20 * (1.0 if prediction_preserved else 0.0),
                0.0, 1.0))
            discriminative_loss = float(np.clip(1.0 - discriminative_retention, 0.0, self.information_loss_clip))
            loss_proxy = float(np.clip(np.mean([support_penalty, pred_penalty, margin_penalty, entropy_penalty]), 0.0, self.information_loss_clip))
            stage_audit[key] = {
                "prediction": pred,
                "prediction_preserved": prediction_preserved,
                "margin": margin,
                "margin_preserved_ratio": margin_ratio,
                "support_jaccard_h0": support,
                "entropy": entropy,
                "entropy_change_vs_h0": entropy_change,
                "class_selectivity": selectivity,
                "class_selectivity_change_vs_h0": selectivity_change,
                "discriminative_retention": discriminative_retention,
                "discriminative_loss_proxy": discriminative_loss,
                "information_loss_proxy": loss_proxy,
            }

        active_input_sources = sum(1 for vals in (source_bin_counts or {}).values() if any(float(v) > 0.0 for v in (list(vals) if vals is not None else [])))
        return {
            "enabled": True,
            "available": True,
            "semantics": "proxy_not_mutual_information",
            "cortical_to_h0": {
                "active_source_count": int(active_input_sources),
                "h0_active_support": int(np.sum(h0 > 0.0)),
                "h0_support_fraction": float(np.mean(h0 > 0.0)),
            },
            "h0_to_latent": {
                "stages": stage_audit,
                "mean_support_preservation": float(np.mean([v["support_jaccard_h0"] for k, v in stage_audit.items() if k != "H0"])) if len(stage_audit) > 1 else 1.0,
                "mean_information_loss_proxy": float(np.mean([v["information_loss_proxy"] for v in stage_audit.values()])) if stage_audit else 0.0,
                "mean_discriminative_loss_proxy": float(np.mean([v["discriminative_loss_proxy"] for k, v in stage_audit.items() if k != "H0"])) if len(stage_audit) > 1 else 0.0,
                "mean_discriminative_retention": float(np.mean([v["discriminative_retention"] for v in stage_audit.values()])) if stage_audit else 1.0,
            },
            "baseline": {
                "prediction": h0_pred,
                "margin": h0_margin,
                "entropy": float(self._entropy(h0_prob)) if h0_prob.size else 0.0,
                "class_selectivity": h0_selectivity,
            },
            "trial_temporal_structure": {
                "active_bins": int(self._last_temporal_audit.get("active_bins", 0)),
                "temporal_collapse": bool(self._last_temporal_audit.get("temporal_collapse", False)),
                "entropy_normalized": float(self._last_temporal_audit.get("entropy_normalized", 0.0)),
            },
        }

    def run(
        self,
        source_bin_counts: Dict[int, Iterable[float]],
        symbols: Iterable[str],
        readout_matrix: Optional[np.ndarray] = None,
        steps: Optional[int] = None,
    ) -> Dict[str, object]:
        """Ejecuta H0 + refinamiento opcional, con H0 como baseline real."""
        self.reset_trial()
        syms = [str(x) for x in symbols]
        self._last_temporal_audit = self._temporal_audit(source_bin_counts)
        encoded_h0, context_state, context_updates = self._encode_temporal_context(source_bin_counts)

        if not np.any(encoded_h0 > 0.0):
            self.trial_runs += 1
            self._last_halt_reason = "no_input"
            self.last_metrics = {
                "version": self.VERSION, "input_nonzero": 0, "context_sparsity": 0.0,
                "workspace_sparsity": 0.0, "steps": 0, "converged": True, "state_delta_last": 0.0,
            }
            return {
                "version": self.VERSION, "active": False, "steps": 0,
                "stage_scores": {"H0": {s: 0.0 for s in syms}},
                "stage_predictions": [None], "stage_margins": {"H0": 0.0},
                "stage_diagnostics": {"H0": {"innovation": 0.0, "margin_gain": 0.0}},
                "ablation": {}, "context_nonzero": 0, "context_sparsity": 0.0,
                "context_updates": 0, "workspace_nonzero": 0, "workspace_sparsity": 0.0,
                "state_delta_last": 0.0, "state_delta_mean": 0.0, "converged": True,
                "halt_reason": "no_input", "authoritative": False, "best_stage": "H0",
                "decoder_initialized": bool(self._decoder_initialized),
                "temporal_audit": dict(self._last_temporal_audit),
                "prototype_probe": {"enabled": bool(self.prototype_probe_enabled), "available": False},
            }

        seed = readout_matrix if self.decoder_seed_from_readout else None
        self._ensure_decoder(len(syms), seed_matrix=seed)
        self._last_temporal_feature = self._safe_normalize(self._build_temporal_tensor(source_bin_counts)).copy()
        self._last_temporal_shadow = self._decode_temporal_shadow(source_bin_counts, syms)
        n_steps = int(max(0, steps if steps is not None else self.steps))
        n_steps = min(n_steps, 16)
        self.context_state = context_state.astype(np.float32).copy()
        h = encoded_h0.astype(np.float32).copy()
        self.workspace_state = h.copy()

        stage_scores: Dict[str, Dict[str, float]] = {}
        stage_predictions: list[str | None] = []
        stage_margins: Dict[str, float] = {}
        states = [h.copy()]
        stage_diagnostics: Dict[str, Dict[str, float | str | bool]] = {}

        scores0 = self._decode_pool_scores(h, syms)
        pred0, margin0 = self._winner(scores0)
        stage_scores["H0"] = scores0
        stage_predictions.append(pred0)
        stage_margins["H0"] = margin0
        stage_diagnostics["H0"] = {"innovation": 0.0, "margin_gain": 0.0, "prediction_changed": False}

        ablation_scores = {
            "C-only": self._decode_pool_scores(self._last_direct_state, syms),
            "S-only": self._decode_pool_scores(self._last_context_only_state, syms),
            "C+S": scores0,
        }
        ablation: Dict[str, object] = {}
        for name, scores in ablation_scores.items():
            p, m = self._winner(scores)
            ablation[name] = {"scores": dict(scores), "prediction": p, "margin": float(m)}

        prev = h.copy()
        deltas: list[float] = []
        innovations: list[float] = []
        prev_margin = margin0
        stable_counter = 0
        utility_counter = 0
        halt_reason = "max_steps" if n_steps > 0 else "h0_only"
        for step in range(1, n_steps + 1):
            h = self._run_recurrent_step(prev)
            delta = float(np.linalg.norm(h - prev))
            innovation = float(1.0 - self._cosine_similarity(h, prev))
            deltas.append(delta)
            innovations.append(innovation)
            self.workspace_state = h.copy()
            states.append(h.copy())
            key = f"LATENT-{step}"
            scores = self._decode_pool_scores(h, syms)
            pred, margin = self._winner(scores)
            stage_scores[key] = scores
            stage_predictions.append(pred)
            stage_margins[key] = margin
            margin_gain = float(margin - prev_margin)
            stage_diagnostics[key] = {
                "innovation": innovation,
                "margin_gain": margin_gain,
                "prediction_changed": bool(pred != stage_predictions[-2]),
                "target_free": True,
            }
            prev = h.copy()
            if delta <= self.halt_delta:
                stable_counter += 1
            else:
                stable_counter = 0
            useful_gain = margin_gain >= self.early_exit_min_margin_gain
            prediction_changed = bool(pred != stage_predictions[-2])
            if self.early_exit_enabled and (not useful_gain) and (not prediction_changed):
                utility_counter += 1
            else:
                utility_counter = 0
            if stable_counter >= self.halt_patience:
                halt_reason = "stable_state_or_utility"
                break
            if self.early_exit_enabled and utility_counter >= self.early_exit_patience:
                halt_reason = "early_exit_no_gain"
                break
            prev_margin = margin

        converged = bool(halt_reason == "stable_state_or_utility")
        self._last_stage_innovation = {k: float(v.get("innovation", 0.0)) for k, v in stage_diagnostics.items()}
        self._last_stage_margin_gain = {k: float(v.get("margin_gain", 0.0)) for k, v in stage_diagnostics.items()}
        self._last_halt_reason = halt_reason
        raw_best_stage = self._best_stage(stage_scores, stage_margins)
        stage_keys = list(stage_scores.keys())
        raw_best_state_index = stage_keys.index(raw_best_stage)
        raw_best_state = states[min(raw_best_state_index, len(states) - 1)].copy()
        # 1.6.29b: H0 es el estado operacional estable. Las etapas posteriores
        # son refinamientos candidatos y sólo se aceptan de forma explícita
        # durante aprendizaje por mejora real de loss.
        accepted_stage = "H0"
        accepted_state = states[0].copy()
        self._last_raw_best_stage = str(raw_best_stage)
        self._last_accepted_stage = accepted_stage
        self.workspace_state = accepted_state
        self.last_states = [x.copy() for x in states]
        self.last_stage_scores = dict(stage_scores)
        self.last_stage_predictions = list(stage_predictions)
        self.last_stage_margins = dict(stage_margins)
        self._last_prototype_probe = self._prototype_probe_for_state(states[0], syms)
        self.trial_runs += 1
        self.last_metrics = {
            "version": self.VERSION,
            "input_nonzero": int(np.sum(encoded_h0 > 0.0)),
            "context_nonzero": int(np.sum(context_state > 0.0)),
            "context_sparsity": float(np.mean(context_state > 0.0)),
            "context_updates": int(context_updates),
            "workspace_nonzero": int(np.sum(self.workspace_state > 0.0)),
            "workspace_sparsity": float(np.mean(self.workspace_state > 0.0)),
            "steps": int(len(deltas)),
            "state_delta_last": float(deltas[-1] if deltas else 0.0),
            "state_delta_mean": float(np.mean(deltas)) if deltas else 0.0,
            "innovation_last": float(innovations[-1] if innovations else 0.0),
            "innovation_mean": float(np.mean(innovations)) if innovations else 0.0,
            "converged": converged,
            "halt_reason": halt_reason,
        }
        raw_best_scores = dict(stage_scores.get(raw_best_stage, stage_scores["H0"]))
        h0_pred, h0_margin = self._winner(stage_scores["H0"])
        raw_best_pred, raw_best_margin = self._winner(raw_best_scores)
        accepted_scores = dict(stage_scores["H0"])
        accepted_pred, accepted_margin = self._winner(accepted_scores)
        ablation["recurrent"] = {
            "no_recurrence_stage": "H0", "no_recurrence_prediction": h0_pred,
            "no_recurrence_margin": float(h0_margin), "recurrent_stage": raw_best_stage,
            "recurrent_prediction": raw_best_pred, "recurrent_margin": float(raw_best_margin),
            "margin_gain": float(raw_best_margin - h0_margin),
            "prediction_changed": bool(raw_best_pred != h0_pred),
            "accepted_stage": accepted_stage,
            "accepted_prediction": accepted_pred,
        }
        c_only_margin = float(ablation["C-only"].get("margin", 0.0))
        s_only_margin = float(ablation["S-only"].get("margin", 0.0))
        self._last_information_flow = self._build_information_flow_audit(source_bin_counts, states, stage_scores, stage_predictions, stage_margins)
        return {
            "version": self.VERSION, "active": True, "steps": int(len(deltas)),
            "stage_scores": stage_scores, "stage_predictions": stage_predictions,
            "stage_margins": stage_margins, "stage_diagnostics": stage_diagnostics,
            "ablation": ablation, "best_stage": raw_best_stage, "best_scores": raw_best_scores,
            "best_prediction": raw_best_pred, "best_margin": float(raw_best_margin),
            "raw_best_stage": raw_best_stage, "raw_best_prediction": raw_best_pred,
            "accepted_stage": accepted_stage, "accepted_scores": accepted_scores,
            "accepted_prediction": accepted_pred, "accepted_margin": float(accepted_margin),
            "h0_prediction": h0_pred, "h0_margin": float(h0_margin),
            "h0_is_best_margin_stage": bool(raw_best_stage == "H0"),
            "accepted_stage_is_h0": True,
            "context_nonzero": int(np.sum(context_state > 0.0)),
            "context_sparsity": float(np.mean(context_state > 0.0)),
            "context_updates": int(context_updates),
            "workspace_nonzero": int(np.sum(self.workspace_state > 0.0)),
            "workspace_sparsity": float(np.mean(self.workspace_state > 0.0)),
            "state_delta_last": float(deltas[-1] if deltas else 0.0),
            "state_delta_mean": float(np.mean(deltas)) if deltas else 0.0,
            "state_delta_sequence": [float(x) for x in deltas],
            "innovation_sequence": [float(x) for x in innovations],
            "converged": converged, "halt_reason": halt_reason, "authoritative": False,
            "decoder_initialized": bool(self._decoder_initialized),
            "decoder_seeded_from_readout": bool(self._decoder_seeded_from_readout),
            "decoder_seed_policy": "readout" if self.decoder_seed_from_readout else "neutral_orthogonal",
            "decoder_entropy_best": self._entropy(np.asarray(list(raw_best_scores.values()), dtype=np.float32)),
            "decoder_entropy_h0": self._entropy(np.asarray(list(scores0.values()), dtype=np.float32)),
            "input_source_count_available": int(self.input_sources),
            "temporal_audit": dict(self._last_temporal_audit),
            "temporal_shadow": dict(self._last_temporal_shadow),
            "prototype_probe": dict(self._last_prototype_probe),
            "prototype_anchor": {
                "h0_prediction": h0_pred,
                "prototype_prediction": self._last_prototype_probe.get("prediction"),
                "consistent": bool(
                    not self._last_prototype_probe.get("available") or
                    self._last_prototype_probe.get("prediction") in (None, h0_pred)
                ),
                "soft_anchor_only": True,
            },
            "ablation_margin_delta": {"C+S_minus_S": float(stage_margins.get("H0", 0.0) - s_only_margin),
                                      "C+S_minus_C": float(stage_margins.get("H0", 0.0) - c_only_margin)},
            "operational_policy": "H0_baseline_with_utility_gated_refinement",
            "information_flow": dict(self._last_information_flow),
        }

    def _stage_reliability_snapshot(self, syms: Iterable[str]) -> Dict[str, float]:
        return {str(sym): float(self._stage_reliability.get(str(sym), self.stage_reliability_floor)) for sym in syms}

    def _update_stage_reliability(self, target_idx: int, syms: list[str], stage_predictions: list[str | None]) -> Dict[str, float]:
        """Actualiza fiabilidad histórica por stage usando el target real del trial."""
        target = syms[target_idx] if 0 <= int(target_idx) < len(syms) else None
        if target is None:
            return self._stage_reliability_snapshot(syms)
        alpha = float(self.stage_reliability_decay)
        for i, pred in enumerate(stage_predictions):
            key = "H0" if i == 0 else f"LATENT-{i}"
            observed = 1.0 if pred == target else 0.0
            old = float(self._stage_reliability.get(key, self.stage_reliability_floor))
            self._stage_reliability[key] = float(np.clip((1.0 - alpha) * old + alpha * observed, self.stage_reliability_floor, 1.0))
        return self._stage_reliability_snapshot(syms)

    def _record_bridge_failure(self, target: str, bridge_context: Optional[dict]) -> None:
        if not isinstance(bridge_context, dict):
            return
        target = str(target)
        h0_pred = bridge_context.get("h0_prediction")
        readout_pred = bridge_context.get("readout_prediction")
        latent_pred = bridge_context.get("latent_prediction")
        try:
            latent_target_prob = float(bridge_context.get("latent_target_probability", 0.0))
            readout_target_prob = float(bridge_context.get("readout_target_probability", 0.0))
        except Exception:
            latent_target_prob = 0.0
            readout_target_prob = 0.0
        severity = float(np.clip(latent_target_prob - readout_target_prob, 0.0, 1.0))
        failure = bool(h0_pred == target and readout_pred != target) or bool(latent_pred == target and readout_pred != target)
        if failure:
            self._bridge_failure_memory.append({
                "target": target,
                "h0_prediction": h0_pred,
                "latent_prediction": latent_pred,
                "readout_prediction": readout_pred,
                "latent_target_probability": latent_target_prob,
                "readout_target_probability": readout_target_prob,
                "severity": severity,
            })
            self._bridge_failure_memory = self._bridge_failure_memory[-int(self.bridge_memory_capacity):]

    def _select_learning_stage(self, target_idx: int, syms: list[str]) -> tuple[int, str, list[dict[str, float]]]:
        """Selecciona la etapa para crédito según loss real, nunca por margen."""
        decoder = self._decoder
        losses: list[dict[str, float]] = []
        if decoder is None:
            return 0, "no_decoder", losses
        target_vec = np.zeros(len(syms), dtype=np.float32)
        target_vec[target_idx] = 1.0
        decoder_bias = self._decoder_bias if self._decoder_bias is not None and self._decoder_bias.size == len(syms) else None
        for i, h in enumerate(self.last_states):
            logits = decoder @ h
            if decoder_bias is not None:
                logits = logits + decoder_bias
            probs = self._softmax(logits)
            loss = -float(np.log(max(1e-7, float(probs[target_idx]))))
            losses.append({
                "stage": "H0" if i == 0 else f"LATENT-{i}",
                "target_probability": float(probs[target_idx]),
                "loss": loss,
            })
        if not losses:
            return 0, "no_states", losses
        h0_loss = float(losses[0]["loss"])
        candidates = [(i, float(item["loss"])) for i, item in enumerate(losses[1:], start=1)
                      if float(item["loss"]) <= h0_loss - self.stage_min_loss_gain]
        if not candidates:
            return 0, "baseline_h0_no_later_loss_gain", losses
        # Utility = mejora de loss × fiabilidad histórica del stage. La fiabilidad
        # nunca puede crear crédito por sí sola: primero debe existir mejora real.
        def utility(item):
            idx, loss = item
            key = f"LATENT-{idx}"
            rel = float(self._stage_reliability.get(key, self.stage_reliability_floor))
            gain = max(0.0, h0_loss - loss)
            return (gain * rel, -loss, -idx)
        idx, _ = max(candidates, key=utility)
        return int(idx), "later_stage_improves_loss_over_h0_reliability_gated", losses

    def _update_prototype(self, target: str, h: np.ndarray) -> None:
        if not self.prototype_probe_enabled:
            return
        key = str(target)
        arr = self._safe_normalize(h)
        if not np.any(arr):
            return
        memory = self._prototype_memory.setdefault(key, [])
        memory.append(arr.copy())
        if len(memory) > self.replay_capacity:
            del memory[:-self.replay_capacity]
        old = self._prototypes.get(key)
        if old is None:
            self._prototypes[key] = arr.copy()
            self._prototype_counts[key] = 1
            return
        rate = float(self.prototype_update_rate)
        mixed = ((1.0 - rate) * old + rate * arr).astype(np.float32)
        self._prototypes[key] = self._safe_normalize(mixed)
        self._prototype_counts[key] = int(self._prototype_counts.get(key, 0)) + 1

    def learn_from_trial(self, target: str, symbols: Iterable[str], apply: bool = True, bridge_context: Optional[dict] = None) -> Dict[str, object]:
        """Actualiza decoder una sola vez; crédito recurrente sólo si Hn mejora loss sobre H0."""
        syms = [str(x) for x in symbols]
        target = str(target)
        if target not in syms or not self.last_states or self._decoder is None:
            self.last_learning = {"learning_applied": False, "skip_reason": "missing_target_or_state", "target": target}
            return dict(self.last_learning)

        target_idx = syms.index(target)
        decoder = self._ensure_decoder(len(syms))
        stage_reliability_before = self._stage_reliability_snapshot(syms)
        decoder_before = decoder.copy()
        target_vec = np.zeros(len(syms), dtype=np.float32)
        target_vec[target_idx] = 1.0
        n_states = len(self.last_states)

        # Recupera la masa de supervisión de 1.6.28b (~3.7 para 5 estados),
        # pero fija H0 como referencia fuerte para evitar que H4 degrade el baseline.
        base_stage_weights = np.asarray([1.00, 0.85, 0.75, 0.65, 0.45], dtype=np.float32)
        if n_states <= base_stage_weights.size:
            stage_weights = base_stage_weights[:n_states].copy()
        else:
            extra = np.full(n_states - base_stage_weights.size, 0.30, dtype=np.float32)
            stage_weights = np.concatenate([base_stage_weights, extra])

        decoder_grad = np.zeros_like(decoder, dtype=np.float32)
        stage_updates = []
        pre_update_probs = []
        for stage_i, h in enumerate(self.last_states):
            logits = decoder_before @ h
            if self._decoder_bias is not None and self._decoder_bias.size == len(syms):
                logits = logits + self._decoder_bias
            probs = self._softmax(logits)
            pre_update_probs.append(probs.copy())
            err = (target_vec - probs).astype(np.float32)
            weight = float(stage_weights[stage_i])
            decoder_grad += weight * (err[:, None] * h[None, :])
            loss = -float(np.log(max(1e-7, float(probs[target_idx]))))
            stage_updates.append({
                "stage": "H0" if stage_i == 0 else f"LATENT-{stage_i}",
                "target_probability": float(probs[target_idx]),
                "loss": loss,
                "weight": weight,
            })

        replay_examples = []
        replay_grad = np.zeros_like(decoder_before, dtype=np.float32)
        if self.replay_weight > 0.0 and self.replay_per_class > 0:
            for sym in syms:
                memories = list(self._prototype_memory.get(str(sym), []))
                if not memories:
                    proto = self._prototypes.get(str(sym))
                    if proto is not None:
                        memories = [proto]
                if not memories:
                    continue
                take = min(self.replay_per_class, len(memories))
                selected = memories[-take:]
                replay_target_idx = syms.index(str(sym))
                replay_target_vec = np.zeros(len(syms), dtype=np.float32)
                replay_target_vec[replay_target_idx] = 1.0
                for mem in selected:
                    r_logits = decoder_before @ mem
                    if self._decoder_bias is not None and self._decoder_bias.size == len(syms):
                        r_logits = r_logits + self._decoder_bias
                    r_probs = self._softmax(r_logits)
                    r_err = (replay_target_vec - r_probs).astype(np.float32)
                    replay_grad += float(self.replay_weight) * (r_err[:, None] * mem[None, :])
                    replay_examples.append({
                        "symbol": str(sym),
                        "target_probability": float(r_probs[replay_target_idx]),
                        "loss": float(-np.log(max(1e-7, float(r_probs[replay_target_idx])))),
                    })
        decoder_grad += replay_grad
        decoder_delta = (self.decoder_learning_rate * decoder_grad).astype(np.float32)
        decoder_delta = self._clip_update_norm(decoder_delta, self.decoder_max_update_norm)

        temporal_shadow_learning = self._update_temporal_shadow(
            target, getattr(self, "_last_temporal_feature", np.zeros(self.input_sources * self.temporal_bins, dtype=np.float32)),
            syms, apply=apply,
        )

        learning_idx, learning_reason, _ = self._select_learning_stage(target_idx, syms)
        learning_stage = "H0" if learning_idx == 0 else f"LATENT-{learning_idx}"
        self._last_learning_stage = learning_stage
        self._last_learning_selection_reason = learning_reason
        credit_state = self.last_states[min(learning_idx, len(self.last_states) - 1)]
        final_logits = decoder_before @ credit_state
        final_probs = self._softmax(final_logits)
        final_error = (target_vec - final_probs).astype(np.float32)
        hidden_credit = decoder_before.T @ final_error
        max_credit = float(np.max(np.abs(hidden_credit))) if hidden_credit.size else 0.0
        if max_credit > 1e-8:
            hidden_credit = np.clip(hidden_credit / max_credit, -1.0, 1.0)

        rec_delta = np.zeros_like(self._recurrent, dtype=np.float32)
        ctx_delta = np.zeros_like(self._context_projection, dtype=np.float32)
        recurrent_credit_applied = bool(learning_idx > 0 and learning_reason.startswith("later_stage_improves_loss_over_h0"))
        if recurrent_credit_applied:
            for transition_i in range(1, min(len(self.last_states), learning_idx + 1)):
                prev = self.last_states[transition_i - 1]
                cur = self.last_states[transition_i]
                pre_idx = np.where(prev > 0.0)[0]
                post_idx = np.where(cur > 0.0)[0]
                if pre_idx.size == 0 or post_idx.size == 0:
                    continue
                scale = 1.0 / max(1.0, float(learning_idx))
                for row in post_idx:
                    rc = float(hidden_credit[int(row)])
                    if abs(rc) < 1e-4:
                        continue
                    for col in pre_idx:
                        old = float(self._recurrent[int(row), int(col)])
                        if abs(old) <= 1e-12:
                            continue
                        rec_delta[int(row), int(col)] += float(self.recurrent_learning_rate * rc * float(prev[int(col)]) * scale)

            ctx_idx = np.where(self.context_state > 0.0)[0]
            post_idx = np.where(credit_state > 0.0)[0]
            if ctx_idx.size and post_idx.size:
                for row in post_idx:
                    rc = float(hidden_credit[int(row)])
                    if abs(rc) < 1e-4:
                        continue
                    for col in ctx_idx:
                        old = float(self._context_projection[int(row), int(col)])
                        if abs(old) <= 1e-12:
                            continue
                        ctx_delta[int(row), int(col)] += float(self.context_learning_rate * rc * float(self.context_state[int(col)]))

        rec_delta = self._clip_update_norm(rec_delta, self.recurrent_max_update_norm)
        ctx_delta = self._clip_update_norm(ctx_delta, self.context_max_update_norm)

        if apply:
            decoder[:] = np.clip(decoder_before + decoder_delta, -2.0, 2.0)
            self._recurrent[:] = np.clip(self._recurrent + rec_delta, -self.weight_clip, self.weight_clip)
            self._context_projection[:] = np.clip(self._context_projection + ctx_delta, -self.weight_clip, self.weight_clip)
            self._renormalize_recurrent_rows()
            self._renormalize_context_rows()
            self._update_prototype(target, self.last_states[0])
            self._store_episode(target)
            self._update_stage_reliability(target_idx, syms, self.last_stage_predictions)
            self._record_bridge_failure(target, bridge_context)
            self.learning_steps += 1
            sleep_result = self._maybe_sleep_after_trial(syms)
        else:
            sleep_result = {"enabled": bool(self.sleep_enabled), "applied": False, "reason": "eval_or_no_apply"}

        predicted_h0 = syms[int(np.argmax(pre_update_probs[0]))] if pre_update_probs and pre_update_probs[0].size else None
        predicted_credit = syms[int(np.argmax(final_probs))] if final_probs.size else None
        h0_loss = float(stage_updates[0]["loss"]) if stage_updates else 0.0
        selected_loss = float(stage_updates[learning_idx]["loss"]) if stage_updates else h0_loss
        self.last_learning = {
            "learning_applied": bool(apply),
            "mode": "train" if apply else "eval",
            "target": target,
            "baseline_stage": "H0",
            "prediction_before_update": predicted_h0,
            "target_probability_before_update": float(pre_update_probs[0][target_idx]) if pre_update_probs else 0.0,
            "decoder_changed": int(np.sum(np.abs(decoder_delta) > 1e-10)) if apply else 0,
            "decoder_sum_abs_delta": float(np.sum(np.abs(decoder_delta))) if apply else 0.0,
            "decoder_update_norm": float(np.linalg.norm(decoder_delta)),
            "decoder_max_update_norm": float(self.decoder_max_update_norm),
            "decoder_stage_weight_sum": float(np.sum(stage_weights)),
            "decoder_replay_weight": float(self.replay_weight),
            "decoder_replay_examples": int(len(replay_examples)),
            "decoder_replay_loss_mean": float(np.mean([x["loss"] for x in replay_examples])) if replay_examples else 0.0,
            "temporal_shadow": dict(temporal_shadow_learning),
            "recurrent_changed": int(np.sum(np.abs(rec_delta) > 1e-10)) if apply else 0,
            "recurrent_sum_abs_delta": float(np.sum(np.abs(rec_delta))) if apply else 0.0,
            "recurrent_update_norm": float(np.linalg.norm(rec_delta)),
            "context_changed": int(np.sum(np.abs(ctx_delta) > 1e-10)) if apply else 0,
            "context_sum_abs_delta": float(np.sum(np.abs(ctx_delta))) if apply else 0.0,
            "context_update_norm": float(np.linalg.norm(ctx_delta)),
            "stage_losses": stage_updates,
            "learning_stage": learning_stage,
            "learning_selection_reason": learning_reason,
            "h0_loss": h0_loss,
            "selected_stage_loss": selected_loss,
            "selected_stage_loss_gain_over_h0": float(h0_loss - selected_loss),
            "recurrent_credit_applied": recurrent_credit_applied,
            "prototype_update_applied": bool(apply and self.prototype_probe_enabled),
            "prediction_credit_stage": predicted_credit,
            "learning_rule": "h0-baseline-loss-gated-recursion-plus-aggregated-decoder-v5",
            "decoder_update_single_pass": True,
            "stage_reliability_before": stage_reliability_before,
            "stage_reliability_after": self._stage_reliability_snapshot(syms),
            "bridge_context": dict(bridge_context) if isinstance(bridge_context, dict) else {},
            "bridge_failure_memory_size": int(len(self._bridge_failure_memory)),
            "recurrent_credit_uses_pre_update_decoder": True,
            "sleep": dict(sleep_result),
            "sleep_cycles": int(self._sleep_cycles),
            "episodes_per_class": {str(sym): int(len(self._episode_memory.get(str(sym), []))) for sym in syms},
            "information_flow_loss_proxy": float((self._last_information_flow.get("h0_to_latent", {}) or {}).get("mean_information_loss_proxy", 0.0)) if self._last_information_flow else 0.0,
        }
        return dict(self.last_learning)

    def _store_episode(self, target: str) -> None:
        if not self.last_states:
            return
        key = str(target)
        bucket = self._episode_memory.setdefault(key, [])
        episode = [self._safe_normalize(np.asarray(x, dtype=np.float32)).copy() for x in self.last_states if np.any(np.asarray(x) > 0.0)]
        if not episode:
            return
        bucket.append(episode)
        # La capacidad es moderada para que la memoria siga siendo CPU/RAM barata.
        max_episodes = max(2, int(self.replay_capacity) * 2)
        if len(bucket) > max_episodes:
            del bucket[:-max_episodes]

    def _apply_homeostatic_downscale(self) -> Dict[str, float]:
        """Normalización lenta y agnóstica a clase para el operador recurrente/contextual."""
        rate = float(self.sleep_homeostasis_rate)
        target = float(self.sleep_homeostasis_target_l1)
        rec_before = []
        ctx_before = []
        for mat in (self._recurrent, self._context_projection):
            for i in range(mat.shape[0]):
                row = mat[i]
                norm = float(np.sum(np.abs(row)))
                if norm <= 1e-8:
                    continue
                factor = 1.0 + rate * (target / max(target, norm) - 1.0)
                factor = float(np.clip(factor, 0.985, 1.015))
                before = norm
                row *= factor
                row[:] = np.clip(row, -self.weight_clip, self.weight_clip)
                if mat is self._recurrent:
                    rec_before.append(before)
                else:
                    ctx_before.append(before)
        self._renormalize_recurrent_rows()
        self._renormalize_context_rows()
        return {
            "recurrent_mean_l1_before": float(np.mean(rec_before)) if rec_before else 0.0,
            "context_mean_l1_before": float(np.mean(ctx_before)) if ctx_before else 0.0,
            "rate": rate,
            "target_l1": target,
        }

    def _sleep_replay(self, syms: list[str], passes: int, replay_per_class: int, weight: float) -> Dict[str, object]:
        if self._decoder is None or not self._episode_memory or weight <= 0.0 or replay_per_class <= 0:
            return {"available": False, "examples": 0, "update_norm": 0.0, "passes": 0}
        decoder_before = self._decoder.copy()
        total_update = np.zeros_like(decoder_before, dtype=np.float32)
        examples = 0
        for _ in range(max(1, int(passes))):
            for sym in syms:
                memories = list(self._episode_memory.get(str(sym), []))
                if not memories:
                    continue
                selected = memories[-min(int(replay_per_class), len(memories)):]
                target_idx = syms.index(str(sym))
                target_vec = np.zeros(len(syms), dtype=np.float32)
                target_vec[target_idx] = 1.0
                for episode in selected:
                    for state in episode:
                        logits = decoder_before @ state
                        if self._decoder_bias is not None and self._decoder_bias.size == len(syms):
                            logits = logits + self._decoder_bias
                        probs = self._softmax(logits)
                        err = (target_vec - probs).astype(np.float32)
                        total_update += float(weight) * (err[:, None] * state[None, :])
                        examples += 1
        delta = self._clip_update_norm((self.decoder_learning_rate * total_update).astype(np.float32), self.decoder_max_update_norm * 0.50)
        self._decoder[:] = np.clip(self._decoder + delta, -2.0, 2.0)
        return {
            "available": True,
            "examples": int(examples),
            "update_norm": float(np.linalg.norm(delta)),
            "changed": int(np.sum(np.abs(delta) > 1e-10)),
            "passes": int(max(1, int(passes))),
        }

    def _apply_rem_like_reorganization(self) -> Dict[str, object]:
        """Fase REM-like opcional, muy débil y sin etiquetas: diversidad del operador recurrente."""
        if not self.sleep_rem_like_enabled or self.sleep_rem_like_passes <= 0 or self.sleep_rem_like_rate <= 0.0:
            return {"enabled": bool(self.sleep_rem_like_enabled), "applied": False, "perturbation_norm": 0.0}
        before = self._recurrent.copy()
        scale = float(self.sleep_rem_like_rate)
        for _ in range(self.sleep_rem_like_passes):
            noise = self._rng.normal(0.0, scale, size=self._recurrent.shape).astype(np.float32)
            # Mantener el operador acotado y con sparsity aproximada: sólo toca
            # conexiones ya existentes; no crea topología nueva.
            mask = np.abs(self._recurrent) > 1e-12
            self._recurrent[mask] += noise[mask]
            self._recurrent[:] = np.clip(self._recurrent, -self.weight_clip, self.weight_clip)
        self._renormalize_recurrent_rows()
        return {
            "enabled": True,
            "applied": True,
            "passes": int(self.sleep_rem_like_passes),
            "perturbation_norm": float(np.linalg.norm(self._recurrent - before)),
        }

    def consolidate_sleep(self, symbols: Iterable[str], force: bool = False) -> Dict[str, object]:
        """Ciclo sueño ligero: NREM replay -> homeostasis -> REM-like opcional.

        Nunca genera spikes ni toca Event Queue. Sólo opera sobre parámetros
        persistentes del workspace latente y su memoria episódica compacta.
        """
        syms = [str(x) for x in symbols]
        if not self.sleep_enabled and not force:
            return {"enabled": False, "applied": False, "reason": "disabled"}
        nrem = self._sleep_replay(syms, self.sleep_nrem_passes, self.sleep_replay_per_class, self.sleep_replay_weight)
        homeo = self._apply_homeostatic_downscale()
        rem = self._apply_rem_like_reorganization()
        self._sleep_cycles += 1
        self._sleep_trials_since_last = 0
        result = {
            "enabled": True,
            "applied": True,
            "cycle": int(self._sleep_cycles),
            "mode": "nrem_replay" if not rem.get("applied") else "nrem_replay_plus_rem_like",
            "nrem_replay": nrem,
            "homeostatic": homeo,
            "rem_like": rem,
            "episode_counts": {sym: int(len(self._episode_memory.get(sym, []))) for sym in syms},
            "external_input_suppressed": True,
            "event_queue_touched": False,
        }
        self._last_sleep = result
        return dict(result)

    def _maybe_sleep_after_trial(self, syms: list[str]) -> Dict[str, object]:
        if not self.sleep_enabled:
            return {"enabled": False, "applied": False, "reason": "disabled"}
        self._sleep_trials_since_last += 1
        if self._sleep_trials_since_last < self.sleep_interval:
            return {"enabled": True, "applied": False, "reason": "interval_not_reached", "trials_since_last_sleep": int(self._sleep_trials_since_last)}
        return self.consolidate_sleep(syms, force=True)

    def _renormalize_recurrent_rows(self) -> None:
        target = float(self.recurrent_gain)
        for i in range(self.hidden_dim):
            row = self._recurrent[i]
            norm = float(np.sum(np.abs(row)))
            if norm <= 1e-8:
                continue
            row *= float(target / norm)
            self._recurrent[i] = np.clip(row, -self.weight_clip, self.weight_clip)

    def _renormalize_context_rows(self) -> None:
        target = float(self.context_gain)
        for i in range(self.hidden_dim):
            row = self._context_projection[i]
            norm = float(np.sum(np.abs(row)))
            if norm <= 1e-8:
                continue
            row *= float(target / norm)
            self._context_projection[i] = np.clip(row, -self.weight_clip, self.weight_clip)

    def state_dict(self) -> Dict[str, object]:
        return {
            "version": self.VERSION,
            "state_version": int(self.STATE_VERSION),
            "seed": int(self.seed),
            "input_sources": int(self.input_sources),
            "temporal_bins": int(self.temporal_bins),
            "hidden_dim": int(self.hidden_dim),
            "steps": int(self.steps),
            "top_k": int(self.top_k),
            "context_update_rate": float(self.context_update_rate),
            "decoder_learning_rate": float(self.decoder_learning_rate),
            "recurrent_learning_rate": float(self.recurrent_learning_rate),
            "context_learning_rate": float(self.context_learning_rate),
            "min_step_mix": float(self.min_step_mix),
            "halt_delta": float(self.halt_delta),
            "weight_clip": float(self.weight_clip),
            "decoder_max_update_norm": float(self.decoder_max_update_norm),
            "recurrent_max_update_norm": float(self.recurrent_max_update_norm),
            "context_max_update_norm": float(self.context_max_update_norm),
            "halt_margin_gain": float(self.halt_margin_gain),
            "halt_innovation": float(self.halt_innovation),
            "halt_patience": int(self.halt_patience),
            "stage_min_loss_gain": float(self.stage_min_loss_gain),
            "early_exit_enabled": bool(self.early_exit_enabled),
            "early_exit_min_margin_gain": float(self.early_exit_min_margin_gain),
            "early_exit_patience": int(self.early_exit_patience),
            "decoder_seed_from_readout": bool(self.decoder_seed_from_readout),
            "neutral_decoder_scale": float(self.neutral_decoder_scale),
            "prototype_probe_enabled": bool(self.prototype_probe_enabled),
            "prototype_update_rate": float(self.prototype_update_rate),
            "replay_capacity": int(self.replay_capacity),
            "replay_per_class": int(self.replay_per_class),
            "replay_weight": float(self.replay_weight),
            "sleep_enabled": bool(self.sleep_enabled),
            "sleep_interval": int(self.sleep_interval),
            "sleep_nrem_passes": int(self.sleep_nrem_passes),
            "sleep_replay_per_class": int(self.sleep_replay_per_class),
            "sleep_replay_weight": float(self.sleep_replay_weight),
            "sleep_homeostasis_rate": float(self.sleep_homeostasis_rate),
            "sleep_homeostasis_target_l1": float(self.sleep_homeostasis_target_l1),
            "sleep_rem_like_enabled": bool(self.sleep_rem_like_enabled),
            "sleep_rem_like_passes": int(self.sleep_rem_like_passes),
            "sleep_rem_like_rate": float(self.sleep_rem_like_rate),
            "information_flow_audit": bool(self.information_flow_audit),
            "information_loss_clip": float(self.information_loss_clip),
            "sleep_trials_since_last": int(self._sleep_trials_since_last),
            "sleep_cycles": int(self._sleep_cycles),
            "last_sleep": dict(self._last_sleep),
            "stage_reliability": {str(k): float(v) for k, v in self._stage_reliability.items()},
            "stage_reliability_decay": float(self.stage_reliability_decay),
            "stage_reliability_floor": float(self.stage_reliability_floor),
            "bridge_memory_capacity": int(self.bridge_memory_capacity),
            "bridge_failure_memory": [dict(x) for x in self._bridge_failure_memory[-int(self.bridge_memory_capacity):]],
            "episode_memory": {str(k): [[np.asarray(state, dtype=np.float32).tolist() for state in episode] for episode in v[-max(2, int(self.replay_capacity)*2):]] for k, v in self._episode_memory.items()},
            "temporal_shadow_enabled": bool(self.temporal_shadow_enabled),
            "temporal_shadow_learning_rate": float(self.temporal_shadow_learning_rate),
            "temporal_shadow_max_update_norm": float(self.temporal_shadow_max_update_norm),
            "prototypes": {str(k): np.asarray(v, dtype=np.float32).tolist() for k, v in self._prototypes.items()},
            "prototype_memory": {str(k): [np.asarray(x, dtype=np.float32).tolist() for x in v[-self.replay_capacity:]] for k, v in self._prototype_memory.items()},
            "prototype_counts": {str(k): int(v) for k, v in self._prototype_counts.items()},
            "last_learning_stage": str(self._last_learning_stage),
            "last_learning_selection_reason": str(self._last_learning_selection_reason),
            "trial_runs": int(self.trial_runs),
            "learning_steps": int(self.learning_steps),
            "context_state": self.context_state.tolist(),
            "workspace_state": self.workspace_state.tolist(),
            "input_projection": self._input_projection.tolist(),
            "recurrent": self._recurrent.tolist(),
            "context_projection": self._context_projection.tolist(),
            "decoder": self._decoder.tolist() if self._decoder is not None else [],
            "decoder_bias": self._decoder_bias.tolist() if self._decoder_bias is not None else [],
            "decoder_initialized": bool(self._decoder_initialized),
            "decoder_seeded_from_readout": bool(self._decoder_seeded_from_readout),
            "temporal_shadow_decoder": self._temporal_shadow_decoder.tolist() if self._temporal_shadow_decoder is not None else [],
            "temporal_shadow_bias": self._temporal_shadow_bias.tolist() if self._temporal_shadow_bias is not None else [],
            "last_learning": dict(self.last_learning),
        }

    def restore_state(self, state: Optional[dict]) -> None:
        if not isinstance(state, dict):
            return
        try:
            c = np.asarray(state.get("context_state", []), dtype=np.float32)
            h = np.asarray(state.get("workspace_state", []), dtype=np.float32)
            p = np.asarray(state.get("input_projection", []), dtype=np.float32)
            r = np.asarray(state.get("recurrent", []), dtype=np.float32)
            cp = np.asarray(state.get("context_projection", []), dtype=np.float32)
            dec = np.asarray(state.get("decoder", []), dtype=np.float32)
            db = np.asarray(state.get("decoder_bias", []), dtype=np.float32)
            if c.size == self.hidden_dim:
                self.context_state = c.copy()
            if h.size == self.hidden_dim:
                self.workspace_state = h.copy()
            if p.shape == self._input_projection.shape:
                self._input_projection = p.copy()
            if r.shape == self._recurrent.shape:
                self._recurrent = r.copy()
            if cp.shape == self._context_projection.shape:
                self._context_projection = cp.copy()
            if dec.ndim == 2 and dec.shape[1] == self.hidden_dim:
                self._decoder = dec.copy()
                self._decoder_initialized = True
            if db.ndim == 1 and self._decoder is not None and db.size == self._decoder.shape[0]:
                self._decoder_bias = db.copy()
            self._decoder_seeded_from_readout = bool(state.get("decoder_seeded_from_readout", self._decoder_seeded_from_readout))
            self.decoder_max_update_norm = float(state.get("decoder_max_update_norm", self.decoder_max_update_norm))
            self.recurrent_max_update_norm = float(state.get("recurrent_max_update_norm", self.recurrent_max_update_norm))
            self.context_max_update_norm = float(state.get("context_max_update_norm", self.context_max_update_norm))
            self.halt_margin_gain = float(state.get("halt_margin_gain", self.halt_margin_gain))
            self.halt_innovation = float(state.get("halt_innovation", self.halt_innovation))
            self.halt_patience = int(max(1, state.get("halt_patience", self.halt_patience)))
            self.stage_min_loss_gain = float(max(0.0, state.get("stage_min_loss_gain", self.stage_min_loss_gain)))
            self.early_exit_enabled = bool(state.get("early_exit_enabled", self.early_exit_enabled))
            self.early_exit_min_margin_gain = float(max(0.0, state.get("early_exit_min_margin_gain", self.early_exit_min_margin_gain)))
            self.early_exit_patience = int(max(1, state.get("early_exit_patience", self.early_exit_patience)))
            self.decoder_seed_from_readout = bool(state.get("decoder_seed_from_readout", self.decoder_seed_from_readout))
            self.neutral_decoder_scale = float(max(0.05, state.get("neutral_decoder_scale", self.neutral_decoder_scale)))
            self.prototype_probe_enabled = bool(state.get("prototype_probe_enabled", self.prototype_probe_enabled))
            self.prototype_update_rate = float(np.clip(state.get("prototype_update_rate", self.prototype_update_rate), 0.0, 1.0))
            self.replay_capacity = int(max(1, state.get("replay_capacity", self.replay_capacity)))
            self.replay_per_class = int(max(0, state.get("replay_per_class", self.replay_per_class)))
            self.replay_weight = float(max(0.0, state.get("replay_weight", self.replay_weight)))
            self.sleep_enabled = bool(state.get("sleep_enabled", self.sleep_enabled))
            self.sleep_interval = int(max(1, state.get("sleep_interval", self.sleep_interval)))
            self.sleep_nrem_passes = int(max(1, state.get("sleep_nrem_passes", self.sleep_nrem_passes)))
            self.sleep_replay_per_class = int(max(0, state.get("sleep_replay_per_class", self.sleep_replay_per_class)))
            self.sleep_replay_weight = float(max(0.0, state.get("sleep_replay_weight", self.sleep_replay_weight)))
            self.sleep_homeostasis_rate = float(np.clip(state.get("sleep_homeostasis_rate", self.sleep_homeostasis_rate), 0.0, 0.25))
            self.sleep_homeostasis_target_l1 = float(max(1e-4, state.get("sleep_homeostasis_target_l1", self.sleep_homeostasis_target_l1)))
            self.sleep_rem_like_enabled = bool(state.get("sleep_rem_like_enabled", self.sleep_rem_like_enabled))
            self.sleep_rem_like_passes = int(max(0, state.get("sleep_rem_like_passes", self.sleep_rem_like_passes)))
            self.sleep_rem_like_rate = float(max(0.0, state.get("sleep_rem_like_rate", self.sleep_rem_like_rate)))
            self.information_flow_audit = bool(state.get("information_flow_audit", self.information_flow_audit))
            self.information_loss_clip = float(max(0.0, state.get("information_loss_clip", self.information_loss_clip)))
            self._sleep_trials_since_last = int(max(0, state.get("sleep_trials_since_last", self._sleep_trials_since_last)))
            self._sleep_cycles = int(max(0, state.get("sleep_cycles", self._sleep_cycles)))
            saved_rel = state.get("stage_reliability") or {}
            if isinstance(saved_rel, dict):
                self._stage_reliability = {str(k): float(np.clip(v, self.stage_reliability_floor, 1.0)) for k, v in saved_rel.items()}
            self.stage_reliability_decay = float(np.clip(state.get("stage_reliability_decay", self.stage_reliability_decay), 0.0, 1.0))
            self.stage_reliability_floor = float(np.clip(state.get("stage_reliability_floor", self.stage_reliability_floor), 0.0, 1.0))
            self.bridge_memory_capacity = int(max(1, state.get("bridge_memory_capacity", self.bridge_memory_capacity)))
            saved_bridge = state.get("bridge_failure_memory") or []
            if isinstance(saved_bridge, list):
                self._bridge_failure_memory = [dict(x) for x in saved_bridge if isinstance(x, dict)][-self.bridge_memory_capacity:]
            saved_sleep = state.get("last_sleep")
            if isinstance(saved_sleep, dict):
                self._last_sleep = dict(saved_sleep)
            self._episode_memory = {}
            for key, episodes in (state.get("episode_memory") or {}).items():
                bucket = []
                for episode in episodes[-max(2, int(self.replay_capacity)*2):]:
                    seq = []
                    for value in episode:
                        arr = np.asarray(value, dtype=np.float32)
                        if arr.size == self.hidden_dim:
                            seq.append(self._safe_normalize(arr))
                    if seq:
                        bucket.append(seq)
                if bucket:
                    self._episode_memory[str(key)] = bucket
            self.temporal_shadow_enabled = bool(state.get("temporal_shadow_enabled", self.temporal_shadow_enabled))
            self.temporal_shadow_learning_rate = float(max(0.0, state.get("temporal_shadow_learning_rate", self.temporal_shadow_learning_rate)))
            self.temporal_shadow_max_update_norm = float(max(1e-6, state.get("temporal_shadow_max_update_norm", self.temporal_shadow_max_update_norm)))
            self._prototypes = {}
            for key, value in (state.get("prototypes") or {}).items():
                arr = np.asarray(value, dtype=np.float32)
                if arr.size == self.hidden_dim:
                    self._prototypes[str(key)] = self._safe_normalize(arr)
            self._prototype_counts = {str(k): int(v) for k, v in (state.get("prototype_counts") or {}).items()}
            self._prototype_memory = {}
            for key, values in (state.get("prototype_memory") or {}).items():
                mem = []
                for value in values[-self.replay_capacity:]:
                    arr = np.asarray(value, dtype=np.float32)
                    if arr.size == self.hidden_dim:
                        mem.append(self._safe_normalize(arr))
                if mem:
                    self._prototype_memory[str(key)] = mem
            tsd = np.asarray(state.get("temporal_shadow_decoder", []), dtype=np.float32)
            tsb = np.asarray(state.get("temporal_shadow_bias", []), dtype=np.float32)
            expected_shape = (len(self._prototypes) if False else (tsd.shape[0] if tsd.ndim == 2 else 0), self.input_sources * self.temporal_bins)
            if tsd.ndim == 2 and tsd.shape[1] == self.input_sources * self.temporal_bins:
                self._temporal_shadow_decoder = tsd.copy()
                self._temporal_shadow_bias = tsb.copy() if tsb.ndim == 1 and tsb.size == tsd.shape[0] else np.zeros(tsd.shape[0], dtype=np.float32)
            self._last_learning_stage = str(state.get("last_learning_stage", self._last_learning_stage))
            self._last_learning_selection_reason = str(state.get("last_learning_selection_reason", self._last_learning_selection_reason))
            self.trial_runs = int(max(0, state.get("trial_runs", self.trial_runs)))
            self.learning_steps = int(max(0, state.get("learning_steps", self.learning_steps)))
            saved_learning = state.get("last_learning")
            if isinstance(saved_learning, dict):
                self.last_learning = dict(saved_learning)
        except Exception:
            # La carga de estado no debe romper el proceso principal.
            return
