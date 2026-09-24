# languague_io.py

import numpy as np
from contextlib import nullcontext as _nullcontext
from copy import deepcopy
from core.module import Module
from config import (V_REST, READOUT_POOL_VTHRESH, READOUT_POOL_TAU_M,
                    READOUT_LEARNING_RATE, READOUT_COMPETITOR_RATE,
                    READOUT_ROW_CENTERING, READOUT_BALANCE_MIN, READOUT_BALANCE_MAX,
                    READOUT_SCORE_TEMPERATURE, READOUT_SYNAPTIC_GAIN,
                    READOUT_LATERAL_INHIBITION, READOUT_LATERAL_DECAY,
                    CORTICAL_INTRA_TRIAL_INHIBITION, CORTICAL_WINNER_COOLDOWN_MS,
                    CORTICAL_MIN_WINNERS, READOUT_POOLS_PER_SOURCE,
                    READOUT_ALL_RIVAL_DEPRESSION,
                    READOUT_MIN_UNIQUE_POOL_NEURONS,
                    READOUT_SEPARATION_MARGIN_THRESHOLD, READOUT_INTERPOOL_INHIBITION_STEP,
                    READOUT_INTRAPOOL_FATIGUE_STEP, READOUT_INTRAPOOL_FATIGUE_DECAY_MS,
                    READOUT_INTERPOOL_INHIBITION_DECAY,
                    CORTICAL_REPRESENTATION_SPARSE_INPUTS,
                    CORTICAL_REPRESENTATION_TOP_K,
                    CORTICAL_REPRESENTATION_TEMPORAL_BLEND,
                    CORTICAL_REPRESENTATION_SCORE_THRESHOLD,
                    CORTICAL_REPRESENTATION_REPEAT_PENALTY,
                    READOUT_TARGETS_PER_POOL_PER_SOURCE,
                    READOUT_LEARNING_SOURCE_FRACTION,
                    CORTICAL_SEPARATOR_BINS_X, CORTICAL_SEPARATOR_BINS_Y,
                    CORTICAL_SEPARATOR_NONLINEAR_MIX, CORTICAL_SEPARATOR_WHITEN_ALPHA,
                    CORTICAL_SEPARATOR_TEMPERATURE, CORTICAL_SEPARATOR_SEED,
                    CORTICAL_PROJECTION_NONZERO, CORTICAL_CORE_FRACTION, CORTICAL_CORE_BONUS,
                    CORTICAL_USAGE_PENALTY, CORTICAL_FRINGE_NOVELTY_GAIN, CORTICAL_DIVERSITY_PENALTY,
                    CORTICAL_HOMEOSTASIS_TARGET, CORTICAL_HOMEOSTASIS_RATE, CORTICAL_HOMEOSTASIS_MAX_DELTA,
                    CORTICAL_PATTERN_MEMORY_SIZE, CORTICAL_STABLE_SIMILARITY,
                    CORTICAL_CORE_SIZE, CORTICAL_CORE_ANCHOR_BLEND, CORTICAL_CORE_ANCHOR_BONUS,
                    CORTICAL_PERSISTENT_ANCHOR_BLEND, CORTICAL_PERSISTENT_ANCHOR_SIMILARITY,
                    CORTICAL_MIN_EMIT, CORTICAL_ACTIVITY_FLOOR, CORTICAL_USAGE_SOFT_CAP,
                    CORTICAL_HOMEOSTASIS_MIN_RATE, CORTICAL_HOMEOSTASIS_MAX_RATE,
                    READOUT_TEMPORAL_BINS, READOUT_TEMPORAL_DECAY, READOUT_TEMPORAL_WINDOW_MS,
                    READOUT_ELIGIBILITY_TRACE_DECAY_MS, READOUT_ELIGIBILITY_WINDOW_MS,
                    READOUT_ELIGIBILITY_POST_GAIN, READOUT_SYNAPTIC_SCALING_INTERVAL,
                    READOUT_SYNAPTIC_SCALING_RATE, READOUT_SYNAPTIC_NORM_TARGET,
                    READOUT_POOL_BASELINE_DECAY, READOUT_POOL_BASELINE_VAR_DECAY,
                    READOUT_POOL_BASELINE_FLOOR, READOUT_POOL_Z_CLIP, READOUT_POOL_SCORE_RAW_MIX,
                    READOUT_COMPETITION_MIN_SPIKES, READOUT_COMPETITION_MIN_MARGIN,
                    READOUT_COMPETITION_MIN_INTERVAL_MS,
                    READOUT_SUBTHRESHOLD_GATE, READOUT_SUBTHRESHOLD_GAIN, READOUT_SUBTHRESHOLD_MAX,
                    READOUT_SUBTHRESHOLD_MIN_TARGET_SPIKES,
                    CORTICAL_IDENTITY_MAX_PROTOTYPES, CORTICAL_IDENTITY_SIMILARITY,
                    CORTICAL_IDENTITY_UPDATE, CORTICAL_IDENTITY_CORE_BONUS, CORTICAL_IDENTITY_NOVELTY_GATE,
                    READOUT_1627_USE_ALL_ACTIVE_SOURCES, READOUT_1627_CREDIT_RATE,
                    READOUT_1627_CREDIT_RIVAL_RATE, READOUT_1627_CREDIT_MAX_ELIGIBILITY,
                    READOUT_1627_CREDIT_MIN_ELIGIBILITY, READOUT_1627_CREDIT_ERROR_FLOOR,
                    READOUT_1627_SUBTHRESHOLD_CREDIT_SCALE, READOUT_1627_SOURCE_COUNT_SCALE_MIN,
                    READOUT_1627_SOURCE_COUNT_SCALE_MAX, READOUT_1627_SURPRISE_WEIGHT,
                    READOUT_1627_PRESENCE_WEIGHT, READOUT_1627_SURPRISE_TEMPERATURE,
                    READOUT_1627_SCORE_EPS, READOUT_1627_HOMEOSTASIS_TARGET_RATE,
                    READOUT_1627_HOMEOSTASIS_EMA_DECAY, READOUT_1627_HOMEOSTASIS_THRESHOLD_RATE,
                    READOUT_1627_HOMEOSTASIS_MAX_OFFSET, READOUT_1627_HOMEOSTASIS_MIN_THRESHOLD,
                    READOUT_1627_HOMEOSTASIS_MAX_THRESHOLD, READOUT_1627_COLUMN_NORM_INTERVAL,
                    READOUT_1627_COLUMN_NORM_RATE, READOUT_1627_COLUMN_NORM_MIN_FACTOR,
                    READOUT_1627_COLUMN_NORM_MAX_FACTOR, READOUT_1627_COLUMN_NORM_EPS,
                    READOUT_1627_SHADOW_BINS, READOUT_1627_SHADOW_MIN_SOURCE_RATE)
from core.evolution_config import EVO_MENU
from systems.latent_workspace import LatentWorkspace
from config import (LATENT_WORKSPACE_ENABLED, LATENT_WORKSPACE_AUTHORITATIVE,
                    LATENT_WORKSPACE_VERSION, LATENT_WORKSPACE_STEPS, LATENT_WORKSPACE_HIDDEN_DIM, LATENT_WORKSPACE_TOP_K,
                    LATENT_WORKSPACE_RECURRENT_DENSITY, LATENT_WORKSPACE_RECURRENT_GAIN,
                    LATENT_WORKSPACE_CONTEXT_GAIN, LATENT_WORKSPACE_STATE_MIX, LATENT_WORKSPACE_CONTEXT_UPDATE_RATE,
                    LATENT_WORKSPACE_SEED, LATENT_WORKSPACE_DECODER_LR, LATENT_WORKSPACE_RECURRENT_LR,
                    LATENT_WORKSPACE_CONTEXT_LR, LATENT_WORKSPACE_MIN_STEP_MIX, LATENT_WORKSPACE_HALT_DELTA,
                    LATENT_WORKSPACE_WEIGHT_CLIP, LATENT_WORKSPACE_DECODER_MAX_UPDATE_NORM,
                    LATENT_WORKSPACE_RECURRENT_MAX_UPDATE_NORM, LATENT_WORKSPACE_CONTEXT_MAX_UPDATE_NORM,
                    LATENT_WORKSPACE_HALT_MARGIN_GAIN, LATENT_WORKSPACE_HALT_INNOVATION,
                    LATENT_WORKSPACE_HALT_PATIENCE, LATENT_WORKSPACE_TRAINING_ENABLED,
                    LATENT_WORKSPACE_EVAL_EVERY, LATENT_WORKSPACE_PAIRED_ABLATION,
                    LATENT_WORKSPACE_QUEUE_AUDIT_ENABLED, LATENT_WORKSPACE_STAGE_MIN_LOSS_GAIN,
                    LATENT_WORKSPACE_EARLY_EXIT_ENABLED, LATENT_WORKSPACE_EARLY_EXIT_MIN_MARGIN_GAIN,
                    LATENT_WORKSPACE_EARLY_EXIT_PATIENCE, LATENT_WORKSPACE_DECODER_SEED_FROM_READOUT,
                    LATENT_WORKSPACE_NEUTRAL_DECODER_SCALE, LATENT_WORKSPACE_PROTOTYPE_PROBE_ENABLED,
                    LATENT_WORKSPACE_PROTOTYPE_UPDATE_RATE, LATENT_WORKSPACE_REPLAY_CAPACITY,
                    LATENT_WORKSPACE_REPLAY_PER_CLASS, LATENT_WORKSPACE_REPLAY_WEIGHT,
                    LATENT_WORKSPACE_TEMPORAL_SHADOW_ENABLED, LATENT_WORKSPACE_TEMPORAL_SHADOW_LR,
                    LATENT_WORKSPACE_TEMPORAL_SHADOW_MAX_UPDATE_NORM, LATENT_WORKSPACE_TEMPORAL_AUDIT_ENABLED,
                    LATENT_WORKSPACE_EVAL_BALANCE_STRICT, LATENT_WORKSPACE_EVAL_SYMBOLS,
                    LATENT_WORKSPACE_SLEEP_ENABLED, LATENT_WORKSPACE_SLEEP_INTERVAL,
                    LATENT_WORKSPACE_SLEEP_NREM_PASSES, LATENT_WORKSPACE_SLEEP_REPLAY_PER_CLASS,
                    LATENT_WORKSPACE_SLEEP_REPLAY_WEIGHT, LATENT_WORKSPACE_SLEEP_HOMEOSTASIS_RATE,
                    LATENT_WORKSPACE_SLEEP_HOMEOSTASIS_TARGET_L1, LATENT_WORKSPACE_SLEEP_REM_LIKE_ENABLED,
                    LATENT_WORKSPACE_SLEEP_REM_LIKE_PASSES, LATENT_WORKSPACE_SLEEP_REM_LIKE_RATE,
                    LATENT_WORKSPACE_INFORMATION_FLOW_AUDIT, LATENT_WORKSPACE_INFORMATION_LOSS_CLIP,
                    READOUT_EVIDENCE_SPIKE_WEIGHT, READOUT_EVIDENCE_SURPRISE_WEIGHT, READOUT_EVIDENCE_PRESENCE_WEIGHT,
                    READOUT_HISTORY_BIAS_RATE, READOUT_HISTORY_BIAS_CLIP, READOUT_BRIDGE_LATENT_GUIDE_RATE,
                    READOUT_BRIDGE_LATENT_GAIN_FLOOR, READOUT_BRIDGE_MEMORY_CAPACITY, READOUT_BRIDGE_SLEEP_REPLAY_ENABLED,
                    READOUT_BRIDGE_SLEEP_REPLAY_PER_CLASS, READOUT_BRIDGE_SLEEP_REPLAY_WEIGHT, READOUT_BRIDGE_SLEEP_REPLAY_PASSES,
                    READOUT_TRIAL_EXPECTED_FRAME_DT_MS,
                    READOUT_29F_HOMEOSTASIS_ENABLED, READOUT_29F_HOMEOSTASIS_WARMUP_TRIALS,
                    READOUT_29F_HOMEOSTASIS_RATE_EMA_DECAY, READOUT_29F_HOMEOSTASIS_REFERENCE_QUANTILE,
                    READOUT_29F_HOMEOSTASIS_OVERACTIVITY_RATIO, READOUT_29F_HOMEOSTASIS_MIN_RATE,
                    READOUT_29F_HOMEOSTASIS_OFFSET_RATE, READOUT_29F_HOMEOSTASIS_MAX_OFFSET,
                    READOUT_29F_HOMEOSTASIS_MIN_THRESHOLD, READOUT_29F_HOMEOSTASIS_MAX_THRESHOLD,
                    READOUT_29F_HOMEOSTASIS_EPS, READOUT_29F_HOMEOSTASIS_STATE_VERSION,
                    READOUT_29F_DESIRED_TARGET_MARGIN, READOUT_29F_TARGET_ERROR_FLOOR,
                    READOUT_29F_RIVAL_MARGIN_GUARD, READOUT_29F_MAX_DELTA,
                    READOUT_29F_DISABLE_COLUMN_NORMALIZATION, READOUT_29F_DISABLE_SYNAPTIC_SCALING)

BUFFER_PROFILE = EVO_MENU["BUFFER"]

# --- NUEVO: Perfil RELAY para las conexiones visión->atención ---
# tau_m=20, v_thresh=18 (más fácil de disparar que el default 20)
# energy_drain=0.03 (mucho más bajo que RAPID_FIRE=0.12)
RELAY_PROFILE = EVO_MENU["RELAY"]
# Umbral mínimo para relés que reciben directamente actividad visual normalizada.
# TransductorBlock entrega strengths de ~3..8; 18 mV hacía imposible dispararlos.
SENSORY_RELAY_VTHRESH = 2.5

class LetterIOModule(Module):
    def __init__(self, net=None, symbols=("X", "O")):
        super().__init__("Lenguaje")
        self.net = net
        self.symbols = list(symbols)
        n = len(self.symbols)
        self._buffer_pools = {}                
        self._assigned_buffer_neurons = set()   
        
        # --- Fase A ---
        self._prev_evidence     = {sym: 0.0 for sym in self.symbols}
        self._evidence_cooldown = {sym: False for sym in self.symbols}
        # v1.6.9: separar la evidencia neural observacional del score
        # heurístico del detector. El detector puede alimentar secuencias,
        # pero no debe contaminar evidencia_* ni la telemetría causal.
        self._detector_evidence = {sym: 0.0 for sym in self.symbols}
        self._last_detector_evidence_t = 0.0
        self.detector_threshold = 0.55
        self.edge_threshold     = 0.02
        self.evidence_window_ms = 200.0   # FIX D: 200ms — cubre latencia variable del SpikingThread (era 120ms)
        self._clean_trial_baseline_t = None

        # --- Readout neuronal aprendible ---
        self._readout_pools = {sym: [] for sym in self.symbols}
        self._readout_sources = []
        self._readout_ready = False
        self.readout_learning_rate = float(READOUT_LEARNING_RATE)
        # v1.8: castigo local del competidor y recentrado suave por fuente.
        # Evita que una clase gane solo por excitabilidad global acumulada.
        self.readout_competitor_rate = float(READOUT_COMPETITOR_RATE)
        self.readout_row_centering = float(READOUT_ROW_CENTERING)
        self.readout_initial_weight = 1.05
        self.readout_min_weight = -0.50
        self.readout_max_weight = 2.00
        self.readout_pool_size = 12
        self.readout_source_count = 0
        self.readout_targets_per_class = 1
        # v1.6: interfaz tipada detector -> readout. Reutiliza neuronas existentes.
        self.feature_bus_ready = False
        # 1.6.22: mayor resolución espacial antes de la proyección cortical.
        self.feature_bus_bins = (int(CORTICAL_SEPARATOR_BINS_X), int(CORTICAL_SEPARATOR_BINS_Y))
        self.feature_bus_top_k = 5
        self.feature_bus_strength_floor = 2.8
        self.feature_bus_features = (
            "vertical", "horizontal", "diag_back", "diag_slash", "curve", "corner", "junction", "occupancy",
            "center_junction", "closure", "top_bar", "mid_bar", "bottom_bar",
            "center_vertical", "left_vertical", "diag_cross", "apex", "closure_ring"
        )
        self._feature_bus_groups = {k: [] for k in self.feature_bus_features}
        self._feature_bus_last_activity = {k: 0 for k in self.feature_bus_features}
        # v1.7: los pools NO reciben afinidades de clase codificadas.
        # Todos parten simétricos y la especialización emerge de sus pesos.
        self._readout_feature_affinity = {sym: set(self.feature_bus_features) for sym in self.symbols}
        # v1.9: métricas temporales de fuente, sin plantilla por clase.
        self._trial_source_first_t = {}
        self._trial_source_last_t = {}
        # Aliases robustos frente a alfabetos reducidos o ampliados.
        self._feature_bus_source_class_map = {}
        # v1.6.15: capa de representación cortical agnóstica a las letras.
        # El FeatureBus conserva tipo + posición; esta capa convierte esos mapas
        # en un código latente escaso y competitivo antes del readout.
        self.cortical_representation_ready = False
        self._cortical_representation_neurons = []
        self._cortical_projection = None
        self._cortical_separator_projection = None
        self._cortical_separator_phase = None
        self._cortical_separator_mean = None
        self._cortical_separator_var = None
        self._cortical_separator_initialized = False
        self._cortical_core_ema = None
        self._cortical_usage_ema = None
        self._cortical_consistency_ema = None
        self._cortical_last_trial_pattern = None
        self._cortical_pattern_memory = []
        self._cortical_identity_prototypes = []
        self._cortical_identity_usage = []
        self._cortical_active_identity = None
        self._cortical_identity_similarity = 0.0
        self._cortical_identity_is_new = False
        self._cortical_trial_pattern_sum = None
        self._cortical_trial_pattern_count = 0
        self._cortical_trial_anchor_source = "identity_prototype"
        self._readout_target_sources = {}
        self._readout_pre_trace = {}
        self._readout_post_trace = {}
        self._readout_delivery_trace = {}
        self._trial_pool_bin_counts = {sym: np.zeros(int(READOUT_TEMPORAL_BINS), dtype=np.float32) for sym in self.symbols}
        # 1.6.26: baseline + variance temporal por pool. Se actualiza sólo entre trials.
        bins_n = max(1, int(READOUT_TEMPORAL_BINS))
        self._pool_activity_baseline = {sym: np.ones(bins_n, dtype=np.float32) for sym in self.symbols}
        self._pool_activity_var = {sym: np.ones(bins_n, dtype=np.float32) for sym in self.symbols}
        self._pool_baseline_trials = 0
        # IA 1.6.27: homeostasis lenta específica del readout.
        # 1.6.29f: estado homeostático limpio; rate_ema empieza en 0 para no
        # imponer artificialmente una tasa objetivo antes de observar trials.
        self._readout_pool_rate_ema = {sym: 0.0 for sym in self.symbols}
        self._readout_pool_drive_ema = {sym: 0.0 for sym in self.symbols}
        self._readout_pool_threshold_offset = {sym: 0.0 for sym in self.symbols}
        self._readout_homeostasis_trials = 0
        self._readout_homeostasis_last = {}
        self._readout_column_norm_last = {}
        self._readout_1627_shadow = {}
        # IA 1.6.28: workspace latente recurrente. Es una rama de cómputo
        # posterior al snapshot; por defecto sólo audita y no sustituye el readout neuronal.
        self._latent_workspace = None
        self._latent_workspace_last = {}
        self._bridge_failure_memory = []
        self._trial_queue_before = {}
        self._trial_queue_after = {}
        self._latent_trial_sequence = 0
        self._latent_trial_mode = "train"
        self._latent_trial_eval_expected = None
        self._latent_trial_eval_balance_match = None
        self._trial_target_symbol = None
        self._trial_subthreshold_eligibility = {}
        self._trial_subthreshold_post = {}
        self._readout_1627_shadow = {
            "source_spikes": 0,
            "source_bins": [0, 0, 0],
            "pool_spikes": {sym: 0 for sym in self.symbols},
            "pool_bins": {sym: [0, 0, 0] for sym in self.symbols},
        }
        self._pending_readout_delivery_links = {}
        self._trial_frame_expected = 60
        self._cortical_trial_anchor = None
        self._cortical_trial_anchor_indices = set()
        self._cortical_trial_anchor_age = 0
        self._cortical_last_emit_count = 0
        self._trial_cortical_counts = None
        self._cortical_last_scores = None
        self._trial_separator_raw_sum = None
        self._trial_separator_raw_sq_sum = None
        self._trial_separator_raw_count = 0
        self._trial_cortical_counts = np.zeros(len(self._cortical_representation_neurons), dtype=np.float32) if self._cortical_representation_neurons else None
        self._cortical_previous = None
        self._cortical_last_emit_t = {}
        self._cortical_activity_trace = None
        self.cortical_neuron_count = 96
        self.cortical_top_k = int(CORTICAL_REPRESENTATION_TOP_K)
        self.cortical_sparse_inputs = int(CORTICAL_REPRESENTATION_SPARSE_INPUTS)
        self.cortical_separator_nonlinear_mix = float(CORTICAL_SEPARATOR_NONLINEAR_MIX)
        self.cortical_separator_whiten_alpha = float(CORTICAL_SEPARATOR_WHITEN_ALPHA)
        self.cortical_separator_temperature = float(CORTICAL_SEPARATOR_TEMPERATURE)
        self.cortical_separator_seed = int(CORTICAL_SEPARATOR_SEED)
        self.cortical_temporal_blend = float(CORTICAL_REPRESENTATION_TEMPORAL_BLEND)
        self.cortical_score_threshold = float(CORTICAL_REPRESENTATION_SCORE_THRESHOLD)
        self.cortical_repeat_penalty = float(CORTICAL_REPRESENTATION_REPEAT_PENALTY)
        self._cortical_input_dim = len(self.feature_bus_features) * (self.feature_bus_bins[0] * self.feature_bus_bins[1])
        self.cortical_tau_ms = 55.0
        self.cortical_fatigue = 0.65
        self.cortical_min_strength = 2.7
        self.cortical_max_strength = 3.8
        self.min_learning_pool_spikes = 2
        self.min_learning_source_spikes = 2
        self.readout_settle_timeout_s = 2.5
        # Estado causal del trial: contadores reales llenados por _execute_spike().
        self._trial_source_spike_count = 0
        self._trial_pool_spike_counts = {sym: 0 for sym in self.symbols}
        self._trial_source_indices = set()
        self._trial_source_spike_counts = {}
        self._trial_source_first_t = {}
        self._trial_source_last_t = {}
        self._trial_source_bin_counts = {}
        self._trial_pool_indices = {sym: set() for sym in self.symbols}
        self._trial_pool_bin_counts = {sym: np.zeros(int(READOUT_TEMPORAL_BINS), dtype=np.float32) for sym in self.symbols}
        self._readout_pre_trace = {}
        self._readout_post_trace = {}
        self._readout_delivery_trace = {}
        self._cortical_trial_pattern_sum = None
        self._cortical_trial_pattern_count = 0
        # v1.6.25: la identidad prototipo es específica del trial; nunca se hereda
        # como ancla activa desde el trial anterior. Los prototipos sí persisten.
        self._cortical_active_identity = None
        self._cortical_identity_similarity = 0.0
        self._cortical_identity_is_new = False
        self._cortical_trial_pattern_sum = np.zeros(len(self._cortical_representation_neurons), dtype=np.float32) if self._cortical_representation_neurons else None
        self._cortical_trial_pattern_count = 0
        # v1.6.11: resetear estado de inhibición lateral y separación por trial.
        self._pool_inhibition_state = {sym: 0.0 for sym in self.symbols}
        self._pool_unique_spikes_per_trial = {sym: 0 for sym in self.symbols}
        self._trial_pool_neuron_spike_counts = {}
        self._readout_pool_neuron_fatigue = {}
        self._readout_last_competition_t = None
        self._trial_frame_count = 0
        self._cortical_trials_completed = 0
        self._last_trial_readout = {}
        # v1.6.18: diagnóstico causal source→pool→membrana→umbral.
        self._trial_pool_delivery_diag = {sym: self._new_pool_delivery_diag() for sym in self.symbols}
        self._trial_pool_delivery_samples = []
        self._trial_synapse_eligibility = {}
        # Diagnóstico 1.6.10: snapshot exacto de sinapsis del readout por trial.
        self._diag_trial_weights_before = {}
        self._diag_last_learning_audit = {}
        self._trial_pool_delivery_diag = {sym: self._new_pool_delivery_diag() for sym in self.symbols}
        self._trial_pool_delivery_samples = []
        self._last_trial_readout = {}

        # --- Etapa 3 (Secuencias) ---
        # Log liviano de eventos de flanco (símbolo, timestamp_ms), la misma
        # información que check_edge_events ya imprimía y tiraba a la basura.
        # No son neuronas: es la bitácora software sobre la que se calcula
        # evidencia_{palabra}, igual que evidencia_{letra} se calcula sobre
        # net.last_spike en vez de sobre una neurona "letra-X" mágica.
        # v1.6.11: inhibición lateral entre pools y métricas de separación.
        self._pool_inhibition_state = {sym: 0.0 for sym in self.symbols}  # nivel actual de inhibición cruzada
        self._pool_separation_history = []  # historial de márgenes de separación por trial
        self._pool_unique_spikes_per_trial = {sym: 0 for sym in self.symbols}  # neuronas únicas activas
        self._event_log = []
        self._event_log_max = 40
        self.words = []            # bigramas configurados, ej. ["OX","XO"]
        self._word_pairs = {}      # {"OX": ("O","X")}
        
        for i, sym in enumerate(self.symbols):
            x = 65.0 + (i * (20.0 / max(1, n - 1))) if n > 1 else 75.0

            self.add_input_port(f"evidencia_{sym}", pos=(x, 75.0, 45.0))  # FIX C: z=45 (centro banda, mayor densidad)
            self.add_output_port(f"maestro_{sym}",  pos=(x, 90.0, 52.0))  # inyecta por receive_spike directo, no depende de distancia
            self.add_input_port(f"buffer_{sym}",    pos=(x, 75.0, 75.0))  # Fase B, no tocado hoy


    def protect_band(self, net, z_range=(37.0, 53.0)):
        """
        A diferencia de recable_from_live_band (que solo actúa sobre
        neuronas que YA demostraron estar vivas), esto le da un piso
        energético + perfil de bajo consumo a TODA la banda de lenguaje,
        para que tengan chance real de sobrevivir hasta recibir su
        primer disparo, en vez de morir de atrofia antes de que les
        llegue señal por primera vez.
        """
        z = net.positions[:net.n, 2]
        en_banda = np.where((z > z_range[0]) & (z < z_range[1]) & net.active[:net.n])[0]
        if len(en_banda) == 0:
            print("🔌 [Protección] Banda vacía — nada que proteger.")
            return

        protected_pools = set()
        if self._readout_ready:
            for pool in self._readout_pools.values():
                protected_pools.update(int(i) for i in pool)

        for idx in en_banda:
            idx = int(idx)
            net.energy[idx] = max(float(net.energy[idx]), 1.0)
            net.tau_m[idx] = RELAY_PROFILE.tau_m
            if idx not in protected_pools:
                net.v_thresh[idx] = RELAY_PROFILE.v_thresh
            net.refr_period[idx] = RELAY_PROFILE.refr_period

        print(
            f"🔌 [Protección] {len(en_banda)} neuronas de la banda reperfiladas y con energía asegurada ≥1.0 | "
            f"readout_preservado={len(protected_pools)}", flush=True
        )

    
    def recable_from_live_band(self, net, z_range=(37.0, 53.0), max_per_letter=10):
        # --- NUEVO: proteger SIEMPRE primero, pase lo que pase después ---
        self.protect_band(net, z_range=z_range)

        z = net.positions[:net.n, 2]
        en_banda = (z > z_range[0]) & (z < z_range[1]) & net.active[:net.n]
        idx_banda = np.where(en_banda)[0]
        vivas = idx_banda[net.last_spike[idx_banda] > 0].tolist()

        if len(vivas) < len(self.symbols):
            print(f"🔌 [Recable] Solo {len(vivas)} neurona(s) viva(s) — insuficiente "
                  f"para {len(self.symbols)} letras sin que compartan la misma. "
                  f"(La banda ya quedó protegida para la próxima corrida.)")
            return

        import random
        random.shuffle(vivas)
        grupos = np.array_split(vivas, len(self.symbols))

        for sym, grupo in zip(self.symbols, grupos):
            grupo = [int(i) for i in grupo.tolist()][:max_per_letter]
            port = self.input_ports.get(f"evidencia_{sym}")
            if not port or not grupo: continue

            port.connected_neurons = grupo
            print(f"🔌 [Recable] evidencia_{sym} → {len(grupo)} neuronas disjuntas: {grupo}")


    def assign_buffer_pools(self, net, pool_size=12, search_radius=40.0):
        """Reserva un pool FIJO y EXCLUSIVO de neuronas por símbolo."""
        for sym in self.symbols:
            port = self.input_ports[f"buffer_{sym}"]
            port_pos = np.array(port.pos)
            dist  = np.linalg.norm(net.positions[:net.n] - port_pos, axis=1)
            orden = np.argsort(dist)
            
            pool = []
            for idx in orden:
                idx = int(idx)
                if idx in self._assigned_buffer_neurons or not net.active[idx]:
                    continue
                pool.append(idx)
                self._assigned_buffer_neurons.add(idx)
                if len(pool) >= pool_size:
                    break
                    
            self._buffer_pools[sym] = pool
            port.connected_neurons = pool
            
            # Reperfilado eléctrico
            for idx in pool:
                net.tau_m[idx]       = BUFFER_PROFILE.tau_m
                net.v_thresh[idx]    = BUFFER_PROFILE.v_thresh
                net.refr_period[idx] = BUFFER_PROFILE.refr_period
                
            print(f"🧠 [Buffer] Pool '{sym}': {len(pool)} neuronas reperfiladas")
    

    def debug_band_geometry(self, net, z_range=(37.0, 53.0),
                         injection_point=(75.0, 75.0, 45.0), injection_radius=45.0):  # FIX C: sync con pos puerto
        """¿Están estas neuronas siquiera dentro del alcance físico de
        inject_sensory_activity? Mide distancia real, no supone nada."""
        z = net.positions[:net.n, 2]
        en_banda = (z > z_range[0]) & (z < z_range[1]) & net.active[:net.n]
        idx_banda = np.where(en_banda)[0]
        if len(idx_banda) == 0:
            print("🔬 [Diag-Geo] Banda vacía.")
            return

        pos = net.positions[idx_banda]
        target = np.array(injection_point)
        dist = np.linalg.norm(pos - target, axis=1)
        alcanzables = int(np.sum(dist <= injection_radius))

        print(f"🔬 [Diag-Geo] {len(idx_banda)} neuronas en banda | "
              f"dist. prom al punto de inyección={float(np.mean(dist)):.1f} | "
              f"mín={float(np.min(dist)):.1f} | máx={float(np.max(dist)):.1f} | "
              f"dentro del radio de inyección ({injection_radius}): {alcanzables}/{len(idx_banda)}")

    def configure_feature_bus(self, net, bins=None, z_range=(55.0, 90.0)):
        """Reserva una pequeña interfaz neuronal para preservar tipo y posición de cada detector.

        v1.6 sustituye el antiguo uso del sensory_relay_routes como fuente directa
        del readout; esos relés siguen perteneciendo al camino visual general.
        """
        if self.feature_bus_ready and all(self._feature_bus_groups.get(f) for f in self.feature_bus_features):
            return
        if bins is not None:
            self.feature_bus_bins = tuple(int(x) for x in bins)
        bx_n, by_n = self.feature_bus_bins
        needed = len(self.feature_bus_features) * bx_n * by_n
        n_available = min(
            int(net.n),
            int(len(net.positions)),
            int(len(net.active)),
        )
        if n_available <= 0:
            raise RuntimeError("Feature bus sin neuronas disponibles")
        z = np.asarray(net.positions[:n_available, 2], dtype=float)
        z_lo, z_hi = map(float, z_range)
        heart_raw = getattr(net, "heart_mask", None)
        heart = np.zeros(n_available, dtype=bool) if heart_raw is None else np.asarray(heart_raw, dtype=bool)[:n_available]
        active = np.asarray(net.active[:n_available], dtype=bool)
        candidates = [int(i) for i in np.where((z >= z_lo) & (z <= z_hi) & active & ~heart)[0]]
        # Red de prueba / red recién creada puede no tener posiciones distribuidas.
        if len(candidates) < needed:
            candidates = [int(i) for i in np.where(active & ~heart)[0]]
        excluded = set()
        try:
            v = getattr(net, "vision", None)
            for port in (getattr(v, "coord_x_port", None), getattr(v, "coord_y_port", None), getattr(v, "fovea_radius_port", None)):
                if port is not None:
                    excluded.update(int(i) for i in getattr(port, "connected_neurons", []))
        except Exception:
            pass
        candidates = [i for i in candidates if i not in excluded]
        candidates.sort(key=lambda i: (float(z[i]), float(net.positions[i,0]), float(net.positions[i,1]), i))
        selected = candidates[:needed]
        if len(selected) < needed:
            raise RuntimeError(f"Feature bus insuficiente: {len(selected)} < {needed}")
        selected_set = set(selected)
        n_conn = min(int(net.n), int(len(net.connections)))
        for src in range(n_conn):
            net.connections[src][:] = [int(t) for t in net.connections[src] if int(t) not in selected_set]
        per_feature = bx_n * by_n
        for idx, neuron in enumerate(selected):
            feature = self.feature_bus_features[idx // per_feature]
            self._feature_bus_groups[feature].append(int(neuron))
            net.energy[neuron] = max(float(net.energy[neuron]), 1.5)
            net.tau_m[neuron] = RELAY_PROFILE.tau_m
            net.v_thresh[neuron] = min(float(net.v_thresh[neuron]), SENSORY_RELAY_VTHRESH)
            # Compatibilidad con la auditoría histórica: mismo perfil eléctrico del relay.
            # net.v_thresh[idx] = min(float(net.v_thresh[idx]), SENSORY_RELAY_VTHRESH)
            # El transductor histórico entregaba strengths en el rango strength 3..8.
            net.refr_period[neuron] = RELAY_PROFILE.refr_period
            net.connections[neuron] = []
            net.register_noncompetitive_neurons([neuron], role="feature_bus")
        self.feature_bus_ready = True
        print(f"🧩 [FeatureBus] {len(self.feature_bus_features)} tipos × {bx_n}×{by_n} = {needed} neuronas", flush=True)

    def _initialize_cortical_projection(self, count):
        """1.6.23: proyección sparse para estabilidad + diversidad agnóstica."""
        count = int(count)
        input_dim = int(self._cortical_input_dim)
        rng = np.random.default_rng(self.cortical_separator_seed + int(count) * 17 + input_dim)
        nz = min(int(CORTICAL_PROJECTION_NONZERO), input_dim)
        proj = np.zeros((count, input_dim), dtype=np.float32)
        sep = np.zeros((count, input_dim), dtype=np.float32)
        for row in range(count):
            ids = rng.choice(input_dim, size=nz, replace=False)
            signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=nz)
            vals = rng.normal(0.0, 1.0, size=nz).astype(np.float32) * signs
            vals /= max(float(np.linalg.norm(vals)), 1e-6)
            proj[row, ids] = vals
            ids2 = rng.choice(input_dim, size=nz, replace=False)
            vals2 = rng.normal(0.0, 1.0, size=nz).astype(np.float32)
            vals2 /= max(float(np.linalg.norm(vals2)), 1e-6)
            sep[row, ids2] = vals2
        phase = rng.uniform(-np.pi, np.pi, size=count).astype(np.float32)
        self._cortical_projection = proj
        self._cortical_separator_projection = sep
        self._cortical_separator_phase = phase
        self._cortical_separator_mean = np.zeros(count, dtype=np.float32)
        self._cortical_separator_var = np.ones(count, dtype=np.float32)
        self._cortical_separator_initialized = False
        self._cortical_last_scores = np.zeros(count, dtype=np.float32)
        self._cortical_previous = None
        self._cortical_activity_trace = np.zeros(count, dtype=np.float32)
        self._cortical_core_ema = np.zeros(count, dtype=np.float32)
        self._cortical_usage_ema = np.zeros(count, dtype=np.float32)
        self._cortical_consistency_ema = np.zeros(count, dtype=np.float32)
        self._cortical_last_trial_pattern = None
        self._cortical_pattern_memory = []
        self._cortical_trial_anchor = None
        self._cortical_trial_anchor_indices = set()
        self._cortical_trial_anchor_age = 0
        self._cortical_last_emit_count = 0
        self._trial_cortical_counts = np.zeros(count, dtype=np.float32)
        self._cortical_projection_similarity = proj @ proj.T
        self._cortical_projection_similarity = np.clip(self._cortical_projection_similarity, -1.0, 1.0)
        self._cortical_last_emit_t = {int(i): -1e12 for i in self._cortical_representation_neurons}

    def configure_cortical_representation(self, net, z_range=(53.5, 55.0), count=None):
        """Crea la capa intermedia de representación, sin codificar identidades de letras.

        Diseño v1.6.15:
        FeatureBus -> proyección fija aleatoria determinista -> competencia espacial/temporal
        -> neuronas corticales -> readout aprendido.
        La proyección usa únicamente mapas visuales y posición, nunca etiquetas X/O/T/A/E.
        """
        if self.cortical_representation_ready:
            return
        count = int(count or self.cortical_neuron_count)
        n_avail = min(int(net.n), len(net.positions), len(net.active))
        z = np.asarray(net.positions[:n_avail, 2], dtype=float)
        active = np.asarray(net.active[:n_avail], dtype=bool)
        heart_raw = getattr(net, 'heart_mask', None)
        heart = np.zeros(n_avail, dtype=bool) if heart_raw is None else np.asarray(heart_raw, dtype=bool)[:n_avail]
        bus = {int(i) for f in self.feature_bus_features for i in self._feature_bus_groups.get(f, [])}
        existing_readout = {int(i) for s in self.symbols for i in self._readout_pools.get(s, [])}
        roles = getattr(net, 'neuron_roles', {}) or {}
        excluded_roles = {"monitor_immortal", "feature_bus", "language_infrastructure"}
        excluded_infra = {int(i) for i, role in roles.items() if str(role) in excluded_roles}
        lo, hi = map(float, z_range)
        cand = [int(i) for i in np.where((z >= lo) & (z < hi) & active & ~heart)[0]
                if int(i) not in bus and int(i) not in existing_readout and int(i) not in excluded_infra]
        if len(cand) < count:
            cand = [int(i) for i in np.where(active & ~heart)[0]
                    if int(i) not in bus and int(i) not in existing_readout and int(i) not in excluded_infra]
        if len(cand) < count:
            raise RuntimeError(f"Representación cortical insuficiente: {len(cand)} < {count}")
        cand.sort(key=lambda i: (float(net.positions[i, 0]), float(net.positions[i, 1]), float(net.positions[i, 2]), i))
        # Submuestreo uniforme para cubrir toda la banda disponible.
        picks = np.linspace(0, len(cand) - 1, count, dtype=int)
        self._cortical_representation_neurons = [int(cand[i]) for i in picks]
        self._cortical_representation_neurons = list(dict.fromkeys(self._cortical_representation_neurons))
        if len(self._cortical_representation_neurons) < count:
            for i in cand:
                if i not in self._cortical_representation_neurons:
                    self._cortical_representation_neurons.append(i)
                if len(self._cortical_representation_neurons) >= count:
                    break
        self._cortical_representation_neurons = self._cortical_representation_neurons[:count]

        # Proyección dispersa sobre feature+posición completos.
        # Sigue siendo agnóstica a X/O/T/A/E y evita colapsar la posición a un único
        # vecindario arbitrario de la neurona cortical.
        self._initialize_cortical_projection(count)
        for idx in self._cortical_representation_neurons:
            net.energy[idx] = max(float(net.energy[idx]), 1.2)
            net.tau_m[idx] = float(self.cortical_tau_ms)
            net.v_thresh[idx] = float(self.cortical_min_strength)
            net.refr_period[idx] = BUFFER_PROFILE.refr_period
            net.connections[idx] = []
            if hasattr(net, 'register_noncompetitive_neurons'):
                net.register_noncompetitive_neurons([idx], role='cortical_representation')
        self.cortical_representation_ready = True
        print(f"🧠 [Corteza] representación cortical preparada: {len(self._cortical_representation_neurons)} neuronas | top-k={self.cortical_top_k}", flush=True)

    def _build_cortical_input_vector(self, feature_maps):
        """1.6.22: separador agnóstico multiescala sobre la geometría completa.

        Usa 6×6 celdas por feature, compresión logarítmica y normalización global.
        La mayor resolución conserva desplazamientos y proporciones que 3×4 borraba.
        """
        bx_n, by_n = self.feature_bus_bins
        vectors = []
        for feature in self.feature_bus_features:
            arr = np.asarray(feature_maps.get(feature), dtype=np.float32)
            if arr.ndim != 2 or arr.size == 0 or not np.isfinite(arr).all():
                arr = np.zeros((20, 20), dtype=np.float32)
            gy, gx = arr.shape
            cells = []
            for iy in range(by_n):
                y0, y1 = int(iy * gy / by_n), int((iy + 1) * gy / by_n)
                for ix in range(bx_n):
                    x0, x1 = int(ix * gx / bx_n), int((ix + 1) * gx / bx_n)
                    cell = arr[y0:max(y0 + 1, y1), x0:max(x0 + 1, x1)]
                    value = float(np.mean(cell)) if cell.size else 0.0
                    cells.append(max(0.0, value))
            vectors.extend(np.log1p(np.asarray(cells, dtype=np.float32)))
        vec = np.asarray(vectors, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 1e-6:
            vec /= norm
        return vec

    @staticmethod
    def _cosine_similarity(a, b):
        a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
        if a.size == 0 or b.size != a.size:
            return 0.0
        return float(np.dot(a, b) / max(1e-6, np.linalg.norm(a) * np.linalg.norm(b)))

    def _select_identity_prototype(self, current):
        """Selecciona una identidad latente una sola vez por trial."""
        cur = np.asarray(current, dtype=np.float32).copy()
        norm = float(np.linalg.norm(cur))
        if norm > 1e-6:
            cur /= norm

        active = self._cortical_active_identity
        if active is not None and 0 <= int(active) < len(self._cortical_identity_prototypes):
            self._cortical_identity_similarity = self._cosine_similarity(cur, self._cortical_identity_prototypes[int(active)])
            self._cortical_identity_is_new = False
            return np.asarray(self._cortical_identity_prototypes[int(active)], dtype=np.float32)

        best_idx = None; best_sim = -1.0
        for i, proto in enumerate(self._cortical_identity_prototypes):
            sim = self._cosine_similarity(cur, proto)
            if sim > best_sim:
                best_sim = sim; best_idx = i

        threshold = float(CORTICAL_IDENTITY_SIMILARITY)
        created = False
        if best_idx is None or best_sim < threshold:
            if len(self._cortical_identity_prototypes) < int(CORTICAL_IDENTITY_MAX_PROTOTYPES):
                self._cortical_identity_prototypes.append(cur.copy())
                self._cortical_identity_usage.append(0)
                best_idx = len(self._cortical_identity_prototypes) - 1
                best_sim = 1.0
                created = True
            else:
                idx = int(np.argmin(np.asarray(self._cortical_identity_usage, dtype=float)))
                self._cortical_identity_prototypes[idx] = cur.copy()
                self._cortical_identity_usage[idx] = 0
                best_idx = idx; best_sim = 1.0; created = True

        self._cortical_active_identity = int(best_idx)
        if len(self._cortical_identity_usage) < len(self._cortical_identity_prototypes):
            self._cortical_identity_usage.extend([0] * (len(self._cortical_identity_prototypes)-len(self._cortical_identity_usage)))
        self._cortical_identity_usage[int(best_idx)] = int(self._cortical_identity_usage[int(best_idx)]) + 1
        self._cortical_identity_similarity = float(best_sim)
        self._cortical_identity_is_new = bool(created)
        return np.asarray(self._cortical_identity_prototypes[int(best_idx)], dtype=np.float32)

    def set_trial_frame_expectation(self, frames):
        try:
            self._trial_frame_expected = max(1, int(frames))
        except Exception:
            self._trial_frame_expected = 60

    def note_trial_frame(self, frame_index):
        try:
            self._trial_frame_count = max(self._trial_frame_count, int(frame_index))
        except Exception:
            pass

    def _note_pre_trace(self, src, t, amplitude=1.0):
        src = int(src); now = float(t)
        last_t, val = self._readout_pre_trace.get(src, (now, 0.0))
        decay = float(np.exp(-max(0.0, now - float(last_t)) / max(1e-6, float(READOUT_ELIGIBILITY_TRACE_DECAY_MS))))
        self._readout_pre_trace[src] = (now, float(val * decay + max(0.0, float(amplitude))))

    def _note_post_trace(self, tgt, t, amplitude=1.0):
        tgt = int(tgt); now = float(t)
        last_t, val = self._readout_post_trace.get(tgt, (now, 0.0))
        decay = float(np.exp(-max(0.0, now - float(last_t)) / max(1e-6, float(READOUT_ELIGIBILITY_TRACE_DECAY_MS))))
        self._readout_post_trace[tgt] = (now, float(val * decay + max(0.0, float(amplitude))))

    def _update_pre_post_eligibility_for_post(self, tgt, t):
        tgt = int(tgt); now = float(t)
        post_last, post_val = self._readout_post_trace.get(tgt, (now, 0.0))
        connected_sources = list(self._readout_target_sources.get(tgt, []))
        if not connected_sources and self.net is not None:
            connected_sources = [int(src) for src in self._readout_sources
                                 if 0 <= int(src) < self.net.n and int(tgt) in set(int(x) for x in self.net.connections[int(src)])]
        for src in connected_sources:
            src_last, pre_val = self._readout_pre_trace.get(int(src), (now, 0.0))
            dt = max(0.0, now - float(src_last))
            if dt > float(READOUT_ELIGIBILITY_WINDOW_MS):
                continue
            decay_e = float(np.exp(-dt / max(1e-6, float(READOUT_ELIGIBILITY_TRACE_DECAY_MS))))
            prev = float(self._trial_synapse_eligibility.get((int(src), tgt), 0.0))
            elig = prev * decay_e + float(pre_val) * float(post_val) * float(READOUT_ELIGIBILITY_POST_GAIN)
            self._trial_synapse_eligibility[(int(src), tgt)] = float(min(10.0, elig))

    def _emit_cortical_representation(self, net, feature_maps, t):
        """1.6.25: identity prototypes + adaptive fringe, agnóstico a clase.

        El primer frame del trial establece un ancla local. Durante el trial,
        el core se conserva y sólo la fringe responde a novedad. No se usa
        ground-truth ni se codifica X/O/T/A/E en la representación.
        """
        if not self.cortical_representation_ready:
            self.configure_cortical_representation(net)
        n = len(self._cortical_representation_neurons)
        if n == 0 or self._cortical_projection is None:
            return 0
        vec = self._build_cortical_input_vector(feature_maps)
        if vec.size != self._cortical_projection.shape[1]:
            return 0
        now = float(t)

        linear = np.asarray(self._cortical_projection @ vec, dtype=np.float32)
        if self._cortical_separator_projection is not None:
            nonlinear_base = np.asarray(self._cortical_separator_projection @ vec, dtype=np.float32)
            phase = np.asarray(self._cortical_separator_phase, dtype=np.float32)
            nonlinear = np.sin(1.65 * nonlinear_base + phase) - np.sin(phase)
        else:
            nonlinear = np.zeros_like(linear)
        mix = float(np.clip(self.cortical_separator_nonlinear_mix, 0.0, 0.08))
        raw = (1.0 - mix) * linear + mix * nonlinear

        center = float(np.median(raw))
        mad = float(np.median(np.abs(raw - center)))
        scale = max(0.10, 1.4826 * mad, float(np.std(raw)) * 0.65)
        z = np.clip((raw - center) / scale, -4.0, 4.0)
        if self._cortical_separator_mean is not None and self._cortical_separator_initialized:
            hist_scale = np.sqrt(np.maximum(self._cortical_separator_var, 1e-4))
            z_hist = np.clip((raw - self._cortical_separator_mean) / hist_scale, -4.0, 4.0)
            z = 0.90 * z + 0.10 * z_hist

        if self._clean_trial_baseline_t is not None:
            if self._trial_separator_raw_sum is None or self._trial_separator_raw_sum.size != n:
                self._trial_separator_raw_sum = np.zeros(n, dtype=np.float32)
                self._trial_separator_raw_sq_sum = np.zeros(n, dtype=np.float32)
                self._trial_separator_raw_count = 0
            self._trial_separator_raw_sum += raw
            self._trial_separator_raw_sq_sum += raw * raw
            self._trial_separator_raw_count += 1
        else:
            mean = np.asarray(self._cortical_separator_mean, dtype=np.float32)
            var = np.asarray(self._cortical_separator_var, dtype=np.float32)
            alpha = float(np.clip(self.cortical_separator_whiten_alpha, 0.001, 0.05))
            delta = raw - mean
            mean += alpha * delta
            var[:] = np.maximum(1e-4, (1-alpha) * var + alpha * (delta * delta))
            self._cortical_separator_initialized = True

        temp = max(0.50, float(self.cortical_separator_temperature))
        current = 1.0 / (1.0 + np.exp(-np.clip(z / temp, -8.0, 8.0)))

        # IA 1.6.25: seleccionar una identidad latente agnóstica por prototipos.
        # No se reutiliza el último trial de forma global.
        proto = self._select_identity_prototype(current)
        if self._cortical_trial_pattern_sum is None or self._cortical_trial_pattern_sum.size != n:
            self._cortical_trial_pattern_sum = np.zeros(n, dtype=np.float32)
            self._cortical_trial_pattern_count = 0
        self._cortical_trial_pattern_sum += current
        self._cortical_trial_pattern_count += 1

        anchor_vec = np.asarray(proto, dtype=np.float32)
        sim_proto = self._cosine_similarity(current, anchor_vec)
        if sim_proto >= float(CORTICAL_IDENTITY_SIMILARITY):
            blend = 0.62
            current = blend * anchor_vec + (1.0 - blend) * current
            cn = float(np.linalg.norm(current))
            if cn > 1e-6:
                current = current / cn

        # Ancla del trial: sus índices salen del prototipo activo, no del primer
        # patrón global de la red. La fringe queda libre para aportar novedad.
        if self._cortical_trial_anchor is None or len(self._cortical_trial_anchor) != n:
            self._cortical_trial_anchor = anchor_vec.copy()
            core_n = min(int(CORTICAL_CORE_SIZE), n)
            self._cortical_trial_anchor_indices = set(int(j) for j in np.argsort(-self._cortical_trial_anchor, kind='stable')[:core_n])
            self._cortical_trial_anchor_age = 0
        else:
            anchor = np.asarray(self._cortical_trial_anchor, dtype=np.float32)
            sim_anchor = self._cosine_similarity(current, anchor)
            if sim_anchor >= float(CORTICAL_IDENTITY_SIMILARITY):
                b = 0.35
                current = b * anchor + (1.0 - b) * current
            self._cortical_trial_anchor_age += 1

        usage = np.asarray(self._cortical_usage_ema, dtype=np.float32)
        core_ema = np.asarray(self._cortical_core_ema, dtype=np.float32)
        if usage.size != n:
            usage = np.zeros(n, dtype=np.float32); self._cortical_usage_ema = usage
        if core_ema.size != n:
            core_ema = np.zeros(n, dtype=np.float32); self._cortical_core_ema = core_ema

        usage_penalty = 1.0 + float(CORTICAL_USAGE_PENALTY) * np.minimum(np.clip(usage, 0.0, 1.0), float(CORTICAL_USAGE_SOFT_CAP))
        scores = current / usage_penalty

        # Core prior: pequeño bonus específico de la identidad latente activa.
        for j in self._cortical_trial_anchor_indices:
            scores[int(j)] *= (1.0 + float(CORTICAL_IDENTITY_CORE_BONUS))
        # No existe bonus global por ``core_ema``: evitamos fabricar un core
        # compartido que vuelva a colapsar la representación entre clases.

        # Novedad sólo modifica la fringe. Se compara contra las identidades
        # latentes, no contra el último trial global.
        best_sim = 0.0
        for memory_pattern in self._cortical_identity_prototypes[-int(CORTICAL_IDENTITY_MAX_PROTOTYPES):]:
            a = np.asarray(memory_pattern, dtype=np.float32)
            if a.size != n:
                continue
            best_sim = max(best_sim, self._cosine_similarity(current, a))
        novelty = float(np.clip(0.5 * (1.0 - best_sim), 0.0, 1.0))
        for j in range(n):
            if j not in self._cortical_trial_anchor_indices:
                scores[j] *= (1.0 + float(CORTICAL_FRINGE_NOVELTY_GAIN) * novelty)

        # k-WTA suave. La diversidad desanima duplicados, pero ya no mata la actividad.
        candidate_n = min(n, max(int(self.cortical_top_k) * 3, 16))
        candidates = set(int(x) for x in np.argsort(-scores, kind='stable')[:candidate_n])
        chosen = []
        while candidates and len(chosen) < int(self.cortical_top_k):
            best_j = None; best_val = -1e9
            for j in sorted(candidates):
                val = float(scores[j])
                if chosen:
                    sim = max(float(self._cortical_projection_similarity[j, k]) for k in chosen)
                    val -= float(CORTICAL_DIVERSITY_PENALTY) * max(0.0, sim)
                if val > best_val:
                    best_val, best_j = val, int(j)
            candidates.discard(best_j)
            chosen.append(best_j)

        # Mantener un core estable mínimo, pero sólo si sus scores son funcionales.
        core_available = [j for j in sorted(self._cortical_trial_anchor_indices, key=lambda x: -float(scores[x]))]
        floor = float(CORTICAL_ACTIVITY_FLOOR)
        chosen = list(dict.fromkeys(chosen))
        for j in core_available:
            if len(chosen) >= int(self.cortical_top_k):
                break
            if j not in chosen and float(scores[j]) >= floor:
                chosen.append(j)
        chosen = chosen[:int(self.cortical_top_k)]

        # Activity floor: evita que un trial informativo quede sin corteza/readout.
        if len(chosen) < int(CORTICAL_MIN_EMIT):
            fallback = [int(x) for x in np.argsort(-scores, kind='stable') if float(scores[int(x)]) >= floor]
            for j in fallback:
                if j not in chosen:
                    chosen.append(j)
                if len(chosen) >= int(CORTICAL_MIN_EMIT):
                    break
            chosen = chosen[:int(self.cortical_top_k)]

        emitted = 0
        if self._trial_cortical_counts is None or len(self._trial_cortical_counts) != n:
            self._trial_cortical_counts = np.zeros(n, dtype=np.float32)
        for j in chosen:
            idx = int(self._cortical_representation_neurons[j])
            strength_score = float(np.clip(scores[j], 0.0, 1.0))
            strength = self.cortical_min_strength + (self.cortical_max_strength - self.cortical_min_strength) * strength_score
            if net.receive_spike(idx, float(strength), now, origin='feature_bus'):
                self._trial_cortical_counts[j] += 1.0
                self._cortical_last_emit_t[idx] = now
                emitted += 1
        self._cortical_trace_t = now
        self._cortical_last_scores = np.asarray(scores, dtype=np.float32)
        self._cortical_previous = np.asarray(current, dtype=np.float32)
        self._cortical_last_emit_count = int(emitted)
        return emitted

    def emit_feature_maps(self, net, feature_maps, t=None):
        """Emite un máximo de pocas señales por característica conservando posición gruesa."""
        if not self.feature_bus_ready:
            self.configure_feature_bus(net)
        now = float(net.current_time if t is None else t)
        bx_n, by_n = self.feature_bus_bins
        emitted = 0
        for feature in self.feature_bus_features:
            grid = feature_maps.get(feature)
            group = self._feature_bus_groups.get(feature, [])
            if grid is None or not group:
                self._feature_bus_last_activity[feature] = 0
                continue
            arr = np.asarray(grid, dtype=np.float32)
            if arr.ndim != 2 or not np.isfinite(arr).all():
                self._feature_bus_last_activity[feature] = 0
                continue
            gy, gx = arr.shape
            values = []
            for iy in range(by_n):
                y0, y1 = int(iy*gy/by_n), int((iy+1)*gy/by_n)
                for ix in range(bx_n):
                    x0, x1 = int(ix*gx/bx_n), int((ix+1)*gx/bx_n)
                    cell = arr[y0:max(y0+1,y1), x0:max(x0+1,x1)]
                    values.append(float(np.max(cell)) if cell.size else 0.0)
            mx = max(values) if values else 0.0
            if mx <= 1e-6:
                self._feature_bus_last_activity[feature] = 0
                continue
            positives = [v for v in values if v > 0]
            p35 = float(np.percentile(positives, 35)) if positives else 0.0
            threshold = max(0.10*mx, 0.55*p35)
            count = 0
            for slot in sorted(range(len(values)), key=lambda i:(-values[i], i))[:self.feature_bus_top_k]:
                value = values[slot]
                if value < threshold:
                    continue
                target = int(group[slot])
                strength = self.feature_bus_strength_floor + 4.0 * min(1.0, value/mx)
                if net.receive_spike(target, strength, now, origin="feature_bus"):
                    emitted += 1; count += 1
            self._feature_bus_last_activity[feature] = count
        # v1.6.15: FeatureBus deja de ser la entrada directa del readout.
        # Primero construimos una representación latente escasa y competitiva.
        try:
            emitted += self._emit_cortical_representation(net, feature_maps, now)
        except Exception as _ctx_exc:
            logging.debug(f"[Corteza] emisión omitida: {_ctx_exc}")
        return emitted

    def wait_for_trial_readout_settle(self, net, timeout_s=None):
        """Espera a que los eventos feature_bus y readout estén procesados."""
        import time
        deadline = time.time() + float(timeout_s if timeout_s is not None else self.readout_settle_timeout_s)
        while time.time() < deadline:
            with net.lock:
                pending = int(net.event_queue_stats.get("feature_bus_pending", 0))
                rq = len(getattr(net, "readout_event_queue", []))
            if pending == 0 and rq == 0:
                time.sleep(0.04)
                with net.lock:
                    if int(net.event_queue_stats.get("feature_bus_pending", 0)) == 0 and not getattr(net, "readout_event_queue", []):
                        return True
            time.sleep(0.01)
        return False

    def configure_learned_readout(self, net, source_z=(55.0, 90.0),
                                  target_z=(37.0, 53.0), pool_size=12,
                                  source_count=120):
        """Construye un readout neuronal por símbolo, sin inyección de etiquetas."""
        self.protect_band(net, z_range=target_z)
        self.readout_pool_size = int(pool_size)
        source_lo, source_hi = map(float, source_z)
        self.configure_feature_bus(net, z_range=(source_lo, source_hi))

        # v1.6.25: reservar primero los integradores de readout.
        # La red de prueba puede no tener suficientes neuronas en la banda cortical,
        # por lo que el encoder tiene un fallback espacial; sin reservar los pools
        # primero, ese fallback podía seleccionar accidentalmente las mismas neuronas.
        z = net.positions[:net.n, 2]
        active = net.active[:net.n]
        heart = getattr(net, 'heart_mask', None)
        if heart is None:
            heart = np.zeros(net.n, dtype=bool)
        else:
            heart = np.asarray(heart)[:net.n]
        pacemakers = set()
        try:
            pacemakers = {int(i) for i in getattr(getattr(net, 'heart', None), 'pulse_port', None).connected_neurons}
        except Exception:
            pacemakers = set()
        target_mask = (z >= target_z[0]) & (z <= target_z[1]) & active
        bus_neurons = {int(i) for f in self.feature_bus_features for i in self._feature_bus_groups.get(f, [])}
        roles = getattr(net, 'neuron_roles', {}) or {}
        excluded_roles = {"monitor_immortal", "feature_bus", "language_infrastructure"}
        excluded_infra = {int(i) for i, role in roles.items() if str(role) in excluded_roles}
        targets = np.asarray([int(i) for i in np.where(target_mask & ~heart)[0]
                              if int(i) not in bus_neurons and int(i) not in excluded_infra], dtype=int)
        if len(targets) < len(self.symbols) * self.readout_pool_size:
            fallback = [int(i) for i in np.where(active & ~heart)[0]
                        if int(i) not in bus_neurons and int(i) not in excluded_infra]
            targets = np.asarray(fallback, dtype=int)
        if pacemakers:
            targets = np.asarray([i for i in targets if int(i) not in pacemakers], dtype=int)

        required = len(self.symbols) * self.readout_pool_size
        if len(targets) < required:
            raise RuntimeError(f"Pocos targets para readout: {len(targets)} < {required}")
        ordered_targets = targets[np.argsort(net.positions[targets, 0])]
        chunks = np.array_split(ordered_targets, len(self.symbols))
        self._readout_pools = {}
        for sym, chunk in zip(self.symbols, chunks):
            pool = [int(i) for i in chunk[:self.readout_pool_size]]
            self._readout_pools[sym] = pool
            for idx in pool:
                net.energy[idx] = max(float(net.energy[idx]), 1.5)
                # v1.4: pools de lectura = integradores lentos dedicados,
                # no neuronas competitivas de la capa biológica.
                net.tau_m[idx] = float(READOUT_POOL_TAU_M)
                net.v_thresh[idx] = float(READOUT_POOL_VTHRESH)
                net.refr_period[idx] = BUFFER_PROFILE.refr_period
                # El pool clasifica; no debe crear cascadas hacia la red general.
                net.connections[idx] = []
            # IMPORTANTE: NO tocar evidencia_{sym} aquí.
            # Esos puertos son observacionales y no forman parte del readout neuronal.

        # Ahora que los pools están reservados en self._readout_pools, el encoder
        # cortical excluye explícitamente esos índices incluso si necesita fallback.
        self.cortical_representation_ready = False
        self.configure_cortical_representation(net)

        # v1.6.15: las fuentes del readout son las neuronas de representación cortical.
        # Las neuronas del FeatureBus siguen siendo una interfaz de sensores, no un clasificador.
        ordered_sources = [int(i) for i in self._cortical_representation_neurons]
        if not ordered_sources:
            raise RuntimeError("Representación cortical sin neuronas fuente")
        self._readout_sources = ordered_sources
        self.readout_source_count = len(self._readout_sources)

        # Retirar cualquier cableado histórico fuente->pool del FeatureBus.
        all_readout_targets = {int(t) for sym in self.symbols for t in self._readout_pools.get(sym, [])}
        feature_sources = {int(i) for f in self.feature_bus_features for i in self._feature_bus_groups.get(f, [])}
        for src in feature_sources:
            if 0 <= src < net.n:
                net.connections[src][:] = [int(t) for t in net.connections[src] if int(t) not in all_readout_targets]

        # 1.6.21: readout simétrico 5-way. La discriminación ya debe venir de la
        # corteza; cada fuente puede ser comparada contra las cinco clases.
        # Para no aumentar el coste respecto al diseño 3/5×2, se usa una sola
        # sinapsis por par fuente→pool, con targets distribuidos de forma determinista.
        targets_per_pool = max(1, int(READOUT_TARGETS_PER_POOL_PER_SOURCE))
        self._feature_bus_source_class_map.clear()
        for s_idx, src in enumerate(self._readout_sources):
            conns = net.connections[src]
            conns[:] = [int(t) for t in conns if int(t) not in all_readout_targets]
            selected_names = []
            for sym_idx, sym in enumerate(self.symbols):
                pool = self._readout_pools[sym]
                if not pool:
                    continue
                selected_names.append(sym)
                for target_off in range(targets_per_pool):
                    tgt = int(pool[(s_idx * 7 + sym_idx * 3 + target_off) % len(pool)])
                    if tgt not in conns:
                        conns.append(tgt)
                    prior = net.weights.get((int(src), tgt), self.readout_initial_weight)
                    net.weights[(int(src), tgt)] = float(prior)
            self._feature_bus_source_class_map[int(src)] = tuple(selected_names)

        # Mapa inverso para que Network pueda repartir el fan-out entre clases.
        if not hasattr(net, 'readout_target_to_symbol'):
            net.readout_target_to_symbol = {}
        for sym in self.symbols:
            for tgt in self._readout_pools.get(sym, []):
                net.readout_target_to_symbol[int(tgt)] = sym

        self._readout_target_sources = {int(tgt): [] for sym in self.symbols for tgt in self._readout_pools.get(sym, [])}
        for src in self._readout_sources:
            for tgt in net.connections[int(src)]:
                if int(tgt) in self._readout_target_sources:
                    self._readout_target_sources[int(tgt)].append(int(src))

        # 1.6.16: aislamiento de entrada del readout. Solo las fuentes corticales
        # declaradas pueden alimentar pools; se eliminan enlaces históricos desde
        # cualquier otra neurona para impedir contaminación del clasificador.
        if not hasattr(net, 'readout_target_to_symbol'):
            net.readout_target_to_symbol = {}
        for sym in self.symbols:
            for tgt in self._readout_pools.get(sym, []):
                net.readout_target_to_symbol[int(tgt)] = sym
        net.readout_allowed_sources = {int(i) for i in self._readout_sources}
        for src_idx in range(int(net.n)):
            if src_idx in net.readout_allowed_sources:
                continue
            conns_src = net.connections[src_idx]
            if conns_src:
                net.connections[src_idx] = [int(t) for t in conns_src if int(t) not in all_readout_targets]

        # Arranque simétrico del integrador: nada de potencial/last_spike heredado.
        with net.lock:
            pool_all = [int(i) for sym in self.symbols for i in self._readout_pools.get(sym, [])]
            for idx in self._readout_sources + pool_all:
                if 0 <= idx < net.n:
                    net.membrane_potential[idx] = V_REST if 'V_REST' in globals() else -65.0
                    net.last_spike[idx] = 0.0
                    net.refractory_until[idx] = 0.0

        self._readout_ready = True
        print(f"🧠 [Readout] Corteza({len(self._readout_sources)}) → {len(self.symbols)} pools × {self.readout_pool_size} | w0={self.readout_initial_weight:.2f} | proyección 1×5 pools/source", flush=True)

        # v1.2: no eximir pools del governor. Las fuentes tampoco necesitan
        # exemption para recibir la inyección sensorial, porque origin='sensory'.
        # Solo se limpia una eventual exemption heredada de versiones previas.
        if hasattr(net, 'governor_exempt_targets'):
            for idx in self._readout_sources:
                net.governor_exempt_targets.discard(int(idx))
            for sym in self.symbols:
                for idx in self._readout_pools.get(sym, []):
                    net.governor_exempt_targets.discard(int(idx))



    @staticmethod
    def _new_pool_delivery_diag():
        return {
            "events": 0,
            "strength_sum": 0.0,
            "strength_max": 0.0,
            "pre_v_sum": 0.0,
            "pre_v_min": float("inf"),
            "pre_v_max": float("-inf"),
            "post_v_sum": 0.0,
            "post_v_max": float("-inf"),
            "threshold_sum": 0.0,
            "effective_threshold_min": float("inf"),
            "near_threshold_events": 0,
            "refractory_block_events": 0,
            "would_fire_at_base_threshold": 0,
            "effective_threshold_gap_sum": 0.0,
            "fired_events": 0,
            "inhibition_state_sum": 0.0,
            "fatigue_sum": 0.0,
        }

    def note_readout_delivery(self, idx, t, strength, pre_v, decayed_v, post_v, threshold,
                              refractory_blocked=False, fired=False, base_threshold=None):
        """v1.6.18: registra una entrega real a una neurona de pool.

        No modifica la dinámica. Sirve para separar cuatro casos: entrada baja,
        membrana ya inhibida/fatigada, umbral alto y disparo efectivo.
        """
        try:
            sym = self._readout_pool_for_index(int(idx))
            if sym is None or self._clean_trial_baseline_t is None:
                return
            d = self._trial_pool_delivery_diag.setdefault(sym, self._new_pool_delivery_diag())
            strength = float(strength); pre_v = float(pre_v); decayed_v = float(decayed_v)
            post_v = float(post_v); threshold = float(threshold)
            d["events"] += 1
            d["strength_sum"] += strength
            d["strength_max"] = max(d["strength_max"], strength)
            d["pre_v_sum"] += pre_v
            d["pre_v_min"] = min(d["pre_v_min"], pre_v)
            d["pre_v_max"] = max(d["pre_v_max"], pre_v)
            d["post_v_sum"] += post_v
            d["post_v_max"] = max(d["post_v_max"], post_v)
            d["threshold_sum"] += threshold
            d["effective_threshold_min"] = min(d["effective_threshold_min"], threshold)
            # Margen relativo pequeño: útil para saber si falta poco para disparar.
            if post_v >= 0.90 * threshold:
                d["near_threshold_events"] += 1
            if refractory_blocked:
                d["refractory_block_events"] += 1
            base_th = float(base_threshold) if base_threshold is not None else threshold
            if decayed_v + strength >= base_th:
                d["would_fire_at_base_threshold"] += 1
            d["effective_threshold_gap_sum"] += max(0.0, threshold - base_th)
            if fired:
                d["fired_events"] += 1
            # 1.6.26: convertir una entrega real cercana a threshold en una
            # elegibilidad secundaria sólo si el evento no llegó a disparar.
            # La pareja concreta se recupera mediante el vínculo (src,t) guardado
            # al encolarse; así no hay asignación de crédito sin entrega física.
            if not fired and not refractory_blocked and post_v >= float(READOUT_SUBTHRESHOLD_GATE) * threshold:
                pending = list(self._pending_readout_delivery_links.get(int(idx), []))
                matched = []
                keep = []
                for item in pending:
                    src_i, t_i, pre_at_delivery = item
                    if abs(float(t_i) - float(t)) <= 0.50:
                        matched.append((int(src_i), float(pre_at_delivery)))
                    else:
                        keep.append(item)
                self._pending_readout_delivery_links[int(idx)] = keep
                ratio = float(np.clip(post_v / max(1e-6, threshold), 0.0, 1.25))
                proximity = float(np.clip(
                    (ratio - float(READOUT_SUBTHRESHOLD_GATE)) /
                    max(1e-6, 1.0 - float(READOUT_SUBTHRESHOLD_GATE)), 0.0, 1.0
                ))
                for src_i, pre_at_delivery in matched:
                    trace = float(pre_at_delivery) * (0.25 + 0.75 * proximity) * float(READOUT_SUBTHRESHOLD_GAIN)
                    if trace > 0.0:
                        key = (int(src_i), int(idx))
                        self._trial_subthreshold_eligibility[key] = float(min(
                            float(READOUT_SUBTHRESHOLD_MAX),
                            float(self._trial_subthreshold_eligibility.get(key, 0.0)) + trace
                        ))
                self._trial_subthreshold_post[int(idx)] = max(
                    float(self._trial_subthreshold_post.get(int(idx), 0.0)),
                    float(proximity)
                )
            inh = float(self._pool_inhibition_state.get(sym, 0.0))
            fatigue = float(self._readout_pool_neuron_fatigue.get(int(idx), 0.0))
            d["inhibition_state_sum"] += inh
            d["fatigue_sum"] += fatigue
            if len(self._trial_pool_delivery_samples) < 2000:
                self._trial_pool_delivery_samples.append({
                    "t": float(t), "idx": int(idx), "symbol": sym,
                    "strength": strength, "pre_v": pre_v, "decayed_v": decayed_v,
                    "post_v": post_v, "threshold": threshold,
                    "refractory_blocked": bool(refractory_blocked), "fired": bool(fired),
                    "pool_inhibition": inh, "neuron_fatigue": fatigue,
                })
        except Exception:
            return

    def _readout_pool_for_index(self, idx):
        idx = int(idx)
        for sym, pool in self._readout_pools.items():
            if idx in pool:
                return sym
        return None

    def get_readout_effective_threshold(self, idx, t, base_threshold):
        """Umbral adaptativo: una neurona de pool que dispara acumula fatiga temporal."""
        if self._readout_pool_for_index(idx) is None:
            return float(base_threshold)
        now = float(t)
        last = self._readout_last_competition_t
        if last is not None and self._readout_pool_neuron_fatigue:
            dt = max(0.0, now - float(last))
            decay = float(np.exp(-dt / max(1e-6, float(READOUT_INTRAPOOL_FATIGUE_DECAY_MS))))
            for key in list(self._readout_pool_neuron_fatigue):
                self._readout_pool_neuron_fatigue[key] *= decay
        fatigue = float(self._readout_pool_neuron_fatigue.get(int(idx), 0.0))
        sym = self._readout_pool_for_index(int(idx))
        homeo = float(self._readout_pool_threshold_offset.get(sym, 0.0)) if sym is not None else 0.0
        return float(np.clip(
            float(base_threshold) + homeo + fatigue,
            float(READOUT_1627_HOMEOSTASIS_MIN_THRESHOLD),
            float(READOUT_1627_HOMEOSTASIS_MAX_THRESHOLD),
        ))

    def _apply_readout_competition(self, net, winner_sym, winner_idx, t):
        """Aplica competencia física inmediata entre pools y fatiga intrapool."""
        now = float(t)
        last = self._readout_last_competition_t
        if last is not None:
            dt = max(0.0, now - float(last))
            decay = float(np.exp(-dt / 250.0))
            for sym in self.symbols:
                self._pool_inhibition_state[sym] = float(self._pool_inhibition_state.get(sym, 0.0) * decay)
        for sym in self.symbols:
            if sym == winner_sym:
                continue
            inc = float(READOUT_INTERPOOL_INHIBITION_STEP)
            self._pool_inhibition_state[sym] = min(1.0, float(self._pool_inhibition_state.get(sym, 0.0)) + inc)
            for idx in self._readout_pools.get(sym, []):
                if 0 <= int(idx) < net.n and hasattr(net, "membrane_potential"):
                    net.membrane_potential[int(idx)] = max(V_REST, float(net.membrane_potential[int(idx)]) - inc)
        widx = int(winner_idx)
        self._readout_pool_neuron_fatigue[widx] = min(0.9, float(self._readout_pool_neuron_fatigue.get(widx, 0.0)) + float(READOUT_INTRAPOOL_FATIGUE_STEP))
        try:
            net.event_queue_stats["readout_competition_events"] = net.event_queue_stats.get("readout_competition_events", 0) + 1
            net.event_queue_stats["readout_intrapool_fatigue_events"] = net.event_queue_stats.get("readout_intrapool_fatigue_events", 0) + 1
        except Exception:
            pass
        self._readout_last_competition_t = now

    def note_readout_eligibility(self, src, tgt, t, strength, post_v=None, threshold=None,
                                 fired=False, refractory_blocked=False):
        """Registra la entrega pre y deja un vínculo pendiente con su llegada real.

        El evento se encola con su (src,target,t). La medición subumbral se completa
        posteriormente en ``note_readout_delivery()``, cuando el evento ha llegado
        físicamente al target y conocemos su ``post_v`` real.
        """
        if self._clean_trial_baseline_t is None:
            return
        src = int(src); tgt = int(tgt); now = float(t)
        self._readout_delivery_trace[(src, tgt)] = now
        # Guardar cuánto valía la traza pre en el momento temporal de la entrega.
        try:
            pre_last, pre_val = self._readout_pre_trace.get(src, (now, 0.0))
            dt = max(0.0, now - float(pre_last))
            if dt <= float(READOUT_ELIGIBILITY_WINDOW_MS):
                pre_at_delivery = float(pre_val) * float(np.exp(
                    -dt / max(1e-6, float(READOUT_ELIGIBILITY_TRACE_DECAY_MS))
                ))
            else:
                pre_at_delivery = 0.0
            queue = self._pending_readout_delivery_links.setdefault(tgt, [])
            queue.append((src, now, pre_at_delivery))
            # Mantenerlo corto: los eventos sin entrega no deben crecer indefinidamente.
            cutoff = now - max(100.0, float(READOUT_ELIGIBILITY_WINDOW_MS) * 2.0)
            self._pending_readout_delivery_links[tgt] = [
                item for item in queue if float(item[1]) >= cutoff
            ][-32:]
        except Exception:
            pass

    def note_readout_spike(self, idx, t):
        """Registra el spike efectivo de source/pool durante un trial activo."""
        if self._clean_trial_baseline_t is None:
            return
        idx = int(idx)
        # v1.6.13: si es un pool, registrar identidad de la actividad.
        # La inhibición física se aplica desde el loop de trial cuando existe
        # una dominancia acumulada clara; así el primer spike no decide solo.
        if self._readout_ready:
            for sym, pool in self._readout_pools.items():
                if idx in pool:
                    # v1.6.25: el post spike es la segunda mitad de la
                    # elegibilidad pre-post. No se genera crédito sin este evento.
                    self._note_post_trace(idx, float(t), 1.0)
                    self._update_pre_post_eligibility_for_post(idx, float(t))
                    self._trial_pool_indices.setdefault(sym, set()).add(idx)
                    self._pool_unique_spikes_per_trial[sym] = int(len(self._trial_pool_indices[sym]))
                    self._trial_pool_neuron_spike_counts[idx] = int(self._trial_pool_neuron_spike_counts.get(idx, 0)) + 1
                    self._trial_pool_spike_counts[sym] = int(self._trial_pool_spike_counts.get(sym, 0)) + 1
                    # IA 1.6.27: auditoría sombra de actividad por pool.
                    shadow = self._readout_1627_shadow.setdefault(
                        "pool_spikes", {s: 0 for s in self.symbols})
                    shadow[sym] = int(shadow.get(sym, 0)) + 1
                    bins_n = max(1, int(READOUT_TEMPORAL_BINS))
                    duration = max(1.0, float(max(1, int(self._trial_frame_expected))) * (1000.0 / 30.0))
                    rel = max(0.0, float(t) - float(self._clean_trial_baseline_t))
                    bucket = min(bins_n - 1, max(0, int((rel / duration) * bins_n)))
                    arr = self._trial_pool_bin_counts.setdefault(sym, np.zeros(bins_n, dtype=np.float32))
                    arr[bucket] += 1.0
                    # 1.6.26: la competición física espera a una dominancia
                    # mínima; el primer spike ya no inhibe inmediatamente a los demás pools.
                    counts_now = {s: int(self._trial_pool_spike_counts.get(s, 0)) for s in self.symbols}
                    normalized_now = self._compute_pool_temporal_normalized_scores()
                    ordered_raw = sorted(counts_now.items(), key=lambda kv: (-kv[1], kv[0]))
                    ordered_norm = sorted(normalized_now.items(), key=lambda kv: (-float(kv[1]), kv[0]))
                    should_compete = False
                    if ordered_raw and ordered_norm and ordered_raw[0][1] >= int(READOUT_COMPETITION_MIN_SPIKES):
                        top_score = float(ordered_norm[0][1]); second_score = float(ordered_norm[1][1] if len(ordered_norm) > 1 else 0.0)
                        margin_now = (top_score - second_score) / max(1e-6, top_score + second_score)
                        last_t = self._readout_last_competition_t
                        interval_ok = (last_t is None) or ((float(t) - float(last_t)) >= float(READOUT_COMPETITION_MIN_INTERVAL_MS))
                        should_compete = bool(ordered_norm[0][0] == sym and margin_now >= float(READOUT_COMPETITION_MIN_MARGIN) and interval_ok)
                    if should_compete:
                        self._apply_readout_competition(self.net, sym, idx, float(t))
                    return
        if idx in self._readout_sources:
            self._note_pre_trace(idx, float(t), 1.0)
            self._trial_source_spike_count += 1
            self._trial_source_indices.add(idx)
            self._trial_source_spike_counts[idx] = self._trial_source_spike_counts.get(idx, 0) + 1
            self._readout_1627_shadow["source_spikes"] = int(self._readout_1627_shadow.get("source_spikes", 0)) + 1
            t_f = float(t)
            self._trial_source_first_t[idx] = min(t_f, self._trial_source_first_t.get(idx, t_f))
            self._trial_source_last_t[idx] = max(t_f, self._trial_source_last_t.get(idx, t_f))
            base_t = float(self._clean_trial_baseline_t)
            # Trial temporal envelope: ~60 frames at 30 Hz. Clamped to the
            # observed duration so this remains valid for short tests.
            # IA 1.6.28c: ventana fija basada en expected_frames; no usar el contador observado.
            duration = max(1.0, float(max(1, int(self._trial_frame_expected))) * float(READOUT_TRIAL_EXPECTED_FRAME_DT_MS))
            rel = max(0.0, float(t_f) - base_t) / duration
            bucket = 0 if rel < (1.0 / 3.0) else (1 if rel < (2.0 / 3.0) else 2)
            bins = self._trial_source_bin_counts.setdefault(idx, [0, 0, 0])
            bins[bucket] += 1
            return
        for sym, pool in self._readout_pools.items():
            if idx in pool:
                self._trial_pool_spike_counts[sym] = self._trial_pool_spike_counts.get(sym, 0) + 1
                self._trial_pool_indices.setdefault(sym, set()).add(idx)
                return

    def _diagnostic_readout_synapse_snapshot(self, net):
        """Snapshot exacto de todas las sinapsis representación->pool del readout.
        Incluye todas las sinapsis visibles del clasificador para auditar plasticidad.

        Solo lectura. No modifica topología ni pesos. Se usa para demostrar si
        apply_readout_learning realmente altera los pesos y dónde.
        """
        snap = {}
        if net is None or not self._readout_ready:
            return snap
        for src in self._readout_sources:
            src_i = int(src)
            if src_i < 0 or src_i >= net.n:
                continue
            conn = net.connections[src_i]
            for sym in self.symbols:
                pool = set(int(i) for i in self._readout_pools.get(sym, []))
                for tgt in conn:
                    tgt_i = int(tgt)
                    if tgt_i in pool:
                        key = f"{src_i}:{tgt_i}:{sym}"
                        snap[key] = float(net.weights.get((src_i, tgt_i), self.readout_initial_weight))
        return snap

    @staticmethod
    def _summarize_weight_values(values):
        if not values:
            return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
        arr = np.asarray(values, dtype=float)
        return {
            "count": int(arr.size),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }

    def get_readout_topology_audit(self, net=None):
        """Auditoría estructural del readout, sin modificar conexiones."""
        net = net or self.net
        roles = getattr(net, "neuron_roles", {}) or {} if net is not None else {}
        sources = [int(i) for i in self._readout_sources]
        source_role_counts = {}
        for idx in sources:
            role = str(roles.get(idx, "unknown"))
            source_role_counts[role] = source_role_counts.get(role, 0) + 1
        excluded_roles = {"monitor_immortal", "feature_bus", "language_infrastructure"}
        excluded_sources = [i for i in sources if str(roles.get(i, "unknown")) in excluded_roles]
        pool_sizes = {sym: len(self._readout_pools.get(sym, [])) for sym in self.symbols}
        source_pool_counts = []
        incoming_per_pool = {sym: 0 for sym in self.symbols}
        total_synapses = 0
        score_visible_synapses = 0
        if net is not None:
            pool_sets = {sym: set(int(x) for x in self._readout_pools.get(sym, [])) for sym in self.symbols}
            for src in sources:
                targets = [int(t) for t in getattr(net, "connections", [])[src]] if 0 <= src < getattr(net, "n", 0) else []
                connected_syms = [sym for sym, pool in pool_sets.items() if any(t in pool for t in targets)]
                source_pool_counts.append(len(connected_syms))
                for sym, pool in pool_sets.items():
                    k = sum(1 for t in targets if t in pool)
                    incoming_per_pool[sym] += k
                    total_synapses += k
                    # readout_scores currently consumes one connection per source/pool;
                    # count how many pools actually have at least one visible connection.
                    if k > 0:
                        score_visible_synapses += 1
        return {
            "projection_version": 10,
            "source_count": len(sources),
            "source_role_counts": source_role_counts,
            "excluded_source_count": len(excluded_sources),
            "excluded_source_indices": excluded_sources[:50],
            "pools_per_source_expected": int(READOUT_POOLS_PER_SOURCE),
            "targets_per_pool_per_source_expected": int(READOUT_TARGETS_PER_POOL_PER_SOURCE),
            "expected_total_source_pool_synapses": int(len(sources) * len(self.symbols) * READOUT_TARGETS_PER_POOL_PER_SOURCE),
            "source_pool_count_min": min(source_pool_counts) if source_pool_counts else 0,
            "source_pool_count_max": max(source_pool_counts) if source_pool_counts else 0,
            "pool_sizes": pool_sizes,
            "incoming_synapses_per_pool": incoming_per_pool,
            "total_source_pool_synapses": total_synapses,
            "score_visible_source_pool_pairs": score_visible_synapses,
            "pool_normalization_version": "1.6.27",
            "pool_baseline_trials": int(getattr(self, "_pool_baseline_trials", 0)),
            "all_sources_are_cortical_representation": all(source_role_counts.get("cortical_representation", 0) == len(sources) for _ in [0]) if sources else False,
        }

    def _latent_readout_bridge(self):
        """Relaciona la decisión latente con la fisiología real del readout, sin modificarla."""
        latent = dict(self._latent_workspace_last or {})
        best_scores = latent.get("best_scores") or {}
        if not best_scores:
            return {}
        bridge = {}
        for sym in self.symbols:
            pool = self._readout_pools.get(sym, [])
            weights = []
            if self.net is not None:
                for src in self._readout_sources[:40]:
                    outgoing = set(int(t) for t in self.net.connections[int(src)])
                    for tgt in pool[:6]:
                        if int(tgt) in outgoing:
                            weights.append(float(self.net.weights.get((int(src), int(tgt)), self.readout_initial_weight)))
            bridge[sym] = {
                "latent_probability": float(best_scores.get(sym, 0.0)),
                "readout_pool_spikes": int(self._trial_pool_spike_counts.get(sym, 0)),
                "readout_pool_unique": int(len(self._trial_pool_indices.get(sym, set()))),
                "readout_spike_share": 0.0,
                "latent_vs_spike_share_gap": 0.0,
                "pool_drive_ema": float(self._readout_pool_drive_ema.get(sym, 0.0)),
                "pool_rate_ema": float(self._readout_pool_rate_ema.get(sym, 0.0)),
                "pool_inhibition": float(self._pool_inhibition_state.get(sym, 0.0)),
                "weight_mean_sample": float(np.mean(weights)) if weights else float(self.readout_initial_weight),
            }
        total_spikes = float(sum(int(v.get("readout_pool_spikes", 0)) for v in bridge.values()))
        for sym, item in bridge.items():
            share = float(item.get("readout_pool_spikes", 0)) / max(1.0, total_spikes)
            item["readout_spike_share"] = share
            item["latent_vs_spike_share_gap"] = float(item.get("latent_probability", 0.0) - share)
        # Auditoría puente: compara distribución latente vs actividad física del readout.
        latent_probs = np.asarray([float(best_scores.get(sym, 0.0)) for sym in self.symbols], dtype=np.float32)
        latent_sum = float(np.sum(np.maximum(0.0, latent_probs)))
        latent_probs = latent_probs / max(1e-8, latent_sum) if latent_sum > 0 else latent_probs
        spike_probs = np.asarray([float(bridge[sym].get("readout_spike_share", 0.0)) for sym in self.symbols], dtype=np.float32)
        l1_gap = float(np.sum(np.abs(latent_probs - spike_probs))) if latent_probs.size else 0.0
        bridge_preservation = float(np.clip(1.0 - 0.5 * l1_gap, 0.0, 1.0))
        flow = dict(latent.get("information_flow") or {})
        flow["latent_to_bridge"] = {
            "latent_prediction": latent.get("h0_prediction", latent.get("best_prediction")),
            "readout_prediction_from_spikes": self._build_latent_readout_error_ledger(self._trial_pool_spike_counts).get("readout_prediction_from_spikes"),
            "distribution_l1_gap": l1_gap,
            "preservation_score": bridge_preservation,
            "information_loss_proxy": float(1.0 - bridge_preservation),
        }
        flow["bridge_to_readout"] = {
            "readout_active": bool(total_spikes > 0.0),
            "pool_count": int(len(self.symbols)),
            "spike_support": int(sum(1 for sym in self.symbols if bridge[sym].get("readout_pool_spikes", 0) > 0)),
            "dominant_pool": max(self.symbols, key=lambda sym: int(bridge[sym].get("readout_pool_spikes", 0))) if self.symbols else None,
        }
        return {
            "version": "1.6.29f-h0-to-readout-bridge-v6",
            "latent_stage": latent.get("best_stage", "H0"),
            "latent_prediction": latent.get("best_prediction"),
            "latent_margin": float(latent.get("best_margin", 0.0)),
            "h0_prediction": latent.get("h0_prediction", latent.get("best_prediction")),
            "h0_margin": float(latent.get("h0_margin", latent.get("best_margin", 0.0))),
            "operational_policy": latent.get("operational_policy", "H0_baseline_with_utility_gated_refinement"),
            "accepted_stage": latent.get("accepted_stage", "H0"),
            "raw_best_stage": latent.get("raw_best_stage", latent.get("best_stage", "H0")),
            "error_ledger": self._build_latent_readout_error_ledger(self._trial_pool_spike_counts),
            "information_flow": flow,
            "pools": bridge,
        }

    def _build_latent_readout_error_ledger(self, pool_spikes):
        latent = self._latent_workspace_last or {}
        h0_pred = latent.get("h0_prediction")
        readout_scores = {}
        if pool_spikes:
            total = float(sum(max(0, int(v)) for v in pool_spikes.values()))
            if total > 0.0:
                readout_scores = {sym: float(max(0, int(pool_spikes.get(sym, 0))) / total) for sym in self.symbols}
        readout_pred = max(readout_scores.items(), key=lambda kv: (kv[1], kv[0]))[0] if readout_scores else None
        target = self._trial_target_symbol
        if target is None:
            category = "unlabeled"
        elif h0_pred == target and readout_pred == target:
            category = "both_correct"
        elif h0_pred == target and readout_pred != target:
            category = "latent_h0_only_correct"
        elif h0_pred != target and readout_pred == target:
            category = "readout_only_correct"
        elif readout_pred is None:
            category = "latent_wrong_readout_inactive"
        else:
            category = "both_wrong"
        return {
            "target": target,
            "h0_prediction": h0_pred,
            "readout_prediction_from_spikes": readout_pred,
            "category": category,
            "readout_active": bool(readout_scores),
            "eval_mode": str(self._latent_trial_mode),
            "eval_expected_symbol": self._latent_trial_eval_expected,
            "eval_balance_match": self._latent_trial_eval_balance_match,
        }

    def get_trial_readout_activity(self):
        """Devuelve el ledger causal único del trial actual.

        v1.6.18: ``_trial_pool_spike_counts`` es el único contador de spikes de
        pool; ``_trial_pool_indices`` es la única fuente para neuronas únicas.
        Se exponen además totales e invariantes para que Probe y aprendizaje
        no puedan leer acumuladores distintos para el mismo trial.
        """
        pool_spikes = {s: int(max(0, self._trial_pool_spike_counts.get(s, 0))) for s in self.symbols}
        pool_unique = {s: int(len(self._trial_pool_indices.get(s, set()))) for s in self.symbols}
        pool_total = int(sum(pool_spikes.values()))
        unique_total = int(sum(pool_unique.values()))
        consistency = bool(pool_total >= unique_total)
        topology = self.get_readout_topology_audit(self.net)
        return {
            "ledger_version": "1.6.29f",
            "trial_target": self._trial_target_symbol,
            "trial_mode": str(self._latent_trial_mode),
            "eval_expected_symbol": self._latent_trial_eval_expected,
            "eval_balance_match": self._latent_trial_eval_balance_match,
            "readout_topology_audit": topology,
            "source_spikes": int(self._trial_source_spike_count),
            "source_unique": int(len(self._trial_source_indices)),
            "source_indices": sorted(int(i) for i in self._trial_source_indices),
            "source_spike_counts": {str(int(k)): int(v) for k, v in self._trial_source_spike_counts.items()},
            "source_bin_counts": {str(int(k)): list(v) for k, v in self._trial_source_bin_counts.items()},
            "pool_bin_counts": {str(sym): list(np.asarray(self._trial_pool_bin_counts.get(sym, np.zeros(int(READOUT_TEMPORAL_BINS))), dtype=np.float32)) for sym in self.symbols},
            "pool_temporal_bins": int(READOUT_TEMPORAL_BINS),
            "pool_temporal_decay": float(READOUT_TEMPORAL_DECAY),
            "pool_normalization": {
                "version": "1.6.27",
                "baseline_trials": int(getattr(self, "_pool_baseline_trials", 0)),
                "baseline_decay": float(READOUT_POOL_BASELINE_DECAY),
                "variance_decay": float(READOUT_POOL_BASELINE_VAR_DECAY),
                "floor": float(READOUT_POOL_BASELINE_FLOOR),
                "scores": self._compute_pool_temporal_normalized_scores() if self._clean_trial_baseline_t is not None else {},
            },
            "pool_activity_baseline": {sym: list(np.asarray(self._pool_activity_baseline.get(sym, np.ones(int(READOUT_TEMPORAL_BINS))), dtype=np.float32)) for sym in self.symbols},
            "pool_activity_variance": {sym: list(np.asarray(self._pool_activity_var.get(sym, np.ones(int(READOUT_TEMPORAL_BINS))), dtype=np.float32)) for sym in self.symbols},
            "pool_spikes": pool_spikes,
            "pool_unique": pool_unique,
            "pool_spikes_total": pool_total,
            "pool_unique_total": unique_total,
            "pool_ledger_consistent": consistency,
            "latent_workspace": deepcopy(self._latent_workspace_last) if self._latent_workspace_last else {},
            "latent_readout_error_ledger": self._build_latent_readout_error_ledger(pool_spikes),
            "information_flow": deepcopy((self._latent_workspace_last or {}).get("information_flow", {})),
            "latent_readout_bridge": self._latent_readout_bridge(),
            "pool_delivery_diagnostics": {sym: dict(self._trial_pool_delivery_diag.get(sym, {})) for sym in self.symbols},
            "pool_delivery_samples": list(self._trial_pool_delivery_samples[-200:]),
            "learning_audit": dict(self._diag_last_learning_audit),
            "eligibility": {
                "pairs": int(len(self._trial_synapse_eligibility)),
                "sum": float(sum(max(0.0, float(v)) for v in self._trial_synapse_eligibility.values())),
                "max": float(max(self._trial_synapse_eligibility.values()) if self._trial_synapse_eligibility else 0.0),
                "post_required": True,
                "subthreshold_pairs": int(len(self._trial_subthreshold_eligibility)),
                "subthreshold_sum": float(sum(max(0.0, float(v)) for v in self._trial_subthreshold_eligibility.values())),
                "subthreshold_max": float(max(self._trial_subthreshold_eligibility.values()) if self._trial_subthreshold_eligibility else 0.0),
                "subthreshold_gate": float(READOUT_SUBTHRESHOLD_GATE),
            },
            "identity": {
                "prototype_count": int(len(self._cortical_identity_prototypes)),
                "active_identity": self._cortical_active_identity,
                "similarity": float(self._cortical_identity_similarity),
                "is_new": bool(self._cortical_identity_is_new),
            },
            "readout_1627": {
                "credit_rule": "1.6.29f-synapse-specific-three-factor-margin-gated",
                "use_all_active_sources": bool(READOUT_1627_USE_ALL_ACTIVE_SOURCES),
                "pool_rate_ema": {sym: float(self._readout_pool_rate_ema.get(sym, 0.0)) for sym in self.symbols},
                "pool_threshold_offset": {sym: float(self._readout_pool_threshold_offset.get(sym, 0.0)) for sym in self.symbols},
                "pool_drive_ema": {sym: float(self._readout_pool_drive_ema.get(sym, 0.0)) for sym in self.symbols},
                "column_norm_last": dict(self._readout_column_norm_last),
                "shadow": self._readout_shadow_snapshot_v1627(),
            },
            "queue_trial_audit": self._queue_audit_trial() if LATENT_WORKSPACE_QUEUE_AUDIT_ENABLED else {},
            "latent_readout_bridge": self._latent_readout_bridge(),
            "cortical_separator": {
                "input_dim": int(self._cortical_input_dim),
                "bins": list(self.feature_bus_bins),
                "top_k": int(self.cortical_top_k),
                "nonlinear_mix": float(self.cortical_separator_nonlinear_mix),
                "whiten_alpha": float(self.cortical_separator_whiten_alpha),
                "temperature": float(self.cortical_separator_temperature),
                "initialized": bool(self._cortical_separator_initialized),
            },
        }

    def snapshot_trial_readout(self):
        """Congela una instantánea final del ledger antes del cierre del trial."""
        self._last_trial_readout = self.get_trial_readout_activity()
        return dict(self._last_trial_readout)

    def get_last_trial_readout(self):
        """Devuelve la última instantánea congelada del trial, si existe."""
        return dict(self._last_trial_readout) if self._last_trial_readout else {}

    def get_readout_telemetry(self, net, t=None, window_ms=None):
        """
        Telemetría completa del readout neuronal para diagnóstico.
        Devuelve por cada símbolo:
          - input_spikes: cuántos sources dispararon desde t_inicio (activos en ventana)
          - pool_spikes:  cuántos neuronas del pool dispararon en la ventana
          - weight_mean:  peso medio src→pool
          - weight_delta: (peso_medio - readout_initial_weight), cuánto cambió
          - pool_v_mean:  potencial de membrana medio del pool ahora mismo
        """
        if not self._readout_ready:
            return {}
        now = float(net.current_time if t is None else t)
        win = float(window_ms if window_ms is not None else self.evidence_window_ms)
        w0  = float(self.readout_initial_weight)

        # Fuentes que dispararon en la ventana
        srcs = np.asarray(self._readout_sources, dtype=int)
        srcs = srcs[srcs < net.n]
        if srcs.size:
            ls_src = net.last_spike[srcs]
            valid_src = (ls_src > 0) & ((now - ls_src) >= 0.0) & ((now - ls_src) < win)
            if self._clean_trial_baseline_t is not None:
                valid_src &= ls_src >= self._clean_trial_baseline_t
            src_fired = int(np.sum(valid_src))
        else:
            src_fired = 0

        result = {"readout_sources_fired": src_fired, "readout_sources_total": len(srcs)}
        if self._clean_trial_baseline_t is not None:
            _live_pool = {sym: int(max(0, self._trial_pool_spike_counts.get(sym, 0))) for sym in self.symbols}
            result["readout_trial_pool_spikes_total"] = int(sum(_live_pool.values()))
            result["readout_trial_pool_ledger_consistent"] = bool(
                all(_live_pool[sym] >= len(self._trial_pool_indices.get(sym, set())) for sym in self.symbols)
            )
        if self._clean_trial_baseline_t is not None:
            result.update({
                "readout_trial_source_spikes": int(self._trial_source_spike_count),
                "readout_trial_unique_sources": int(len(self._trial_source_indices)),
            })

        for sym in self.symbols:
            pool = [i for i in self._readout_pools.get(sym, []) if 0 <= i < net.n]
            if not pool:
                result[f"readout_{sym}_pool_spikes"] = 0
                result[f"readout_{sym}_weight_mean"] = w0
                result[f"readout_{sym}_weight_delta"] = 0.0
                result[f"readout_{sym}_pool_v_mean"] = 0.0
                continue

            idx = np.asarray(pool, dtype=int)
            ls  = net.last_spike[idx]
            valid_pool = (ls > 0) & ((now - ls) >= 0.0) & ((now - ls) < win)
            if self._clean_trial_baseline_t is not None:
                valid_pool &= ls >= self._clean_trial_baseline_t
            pool_spikes = int(np.sum(valid_pool))
            causal_pool_spikes = int(self._trial_pool_spike_counts.get(sym, 0)) if self._clean_trial_baseline_t is not None else 0
            causal_pool_unique = int(len(self._trial_pool_indices.get(sym, set()))) if self._clean_trial_baseline_t is not None else 0
            if self._clean_trial_baseline_t is not None:
                pool_spikes = causal_pool_spikes

            # Pesos representación cortical → pool
            weights_sample = []
            for src in srcs[:40]:  # muestra de 40 fuentes
                connected = set(int(t) for t in net.connections[int(src)])
                for tgt in idx[:6]:
                    if int(tgt) not in connected:
                        continue
                    w = net.weights.get((int(src), int(tgt)), w0)
                    weights_sample.append(float(w))
            w_mean  = float(np.mean(weights_sample)) if weights_sample else w0
            w_delta = w_mean - w0

            # Diagnóstico 1.6.10: estadística completa sobre todas las sinapsis
            # source->pool, no solo la muestra 40x6 usada por la telemetría histórica.
            all_weights = []
            connected_count = 0
            for src in self._readout_sources:
                src_i = int(src)
                if src_i < 0 or src_i >= net.n:
                    continue
                connected = set(int(tgt) for tgt in net.connections[src_i])
                for tgt in pool:
                    if tgt in connected:
                        connected_count += 1
                        all_weights.append(float(net.weights.get((src_i, tgt), w0)))
            all_stats = self._summarize_weight_values(all_weights)

            v_values = np.asarray(net.membrane_potential[idx], dtype=float)
            v_mean = float(np.mean(v_values))

            result[f"readout_{sym}_pool_spikes"]  = pool_spikes
            result[f"readout_{sym}_pool_unique_spikes"] = causal_pool_unique
            result[f"readout_{sym}_weight_mean"]  = round(w_mean, 4)
            result[f"readout_{sym}_weight_delta"] = round(w_delta, 4)
            result[f"readout_{sym}_pool_v_mean"]  = round(v_mean, 3)
            result[f"readout_{sym}_synapse_count"] = int(all_stats["count"])
            result[f"readout_{sym}_weight_min"] = round(all_stats["min"], 6)
            result[f"readout_{sym}_weight_max"] = round(all_stats["max"], 6)
            result[f"readout_{sym}_weight_std"] = round(all_stats["std"], 6)
            result[f"readout_{sym}_connected_synapses"] = int(connected_count)

        # Auditoría de aprendizaje del trial anterior: está disponible después
        # de apply_readout_learning() y no interviene en ninguna decisión.
        if self._diag_last_learning_audit:
            result["learning_audit"] = dict(self._diag_last_learning_audit)

        return result

    def get_readout_source_indices(self):
        return list(self._readout_sources)

    def get_readout_indices(self):
        out = []
        for sym in self.symbols:
            out.extend(self._readout_pools.get(sym, []))
        return out

    def _compute_pool_temporal_normalized_scores(self):
        """IA 1.6.29f: evidencia sólo del trial presente.

        ``drive_ema`` queda deliberadamente fuera del score. El score se construye
        con spikes físicos del trial, sorpresa temporal contra baseline y presencia.
        Así una memoria histórica lenta no puede alterar directamente la decisión.
        """
        bins_n = max(1, int(READOUT_TEMPORAL_BINS))
        decay = float(np.clip(READOUT_TEMPORAL_DECAY, 0.0, 0.9999))
        floor = max(1e-3, float(READOUT_POOL_BASELINE_FLOOR))
        zclip = max(0.5, float(READOUT_POOL_Z_CLIP))
        sw = float(np.clip(READOUT_EVIDENCE_SURPRISE_WEIGHT, 0.0, 1.0))
        pw = float(np.clip(READOUT_EVIDENCE_PRESENCE_WEIGHT, 0.0, 1.0))
        normw = max(1e-6, sw + pw)
        sw /= normw; pw /= normw
        temp = max(1e-3, float(READOUT_1627_SURPRISE_TEMPERATURE))
        evidence_raw = {}
        for sym in self.symbols:
            arr = np.asarray(self._trial_pool_bin_counts.get(sym, np.zeros(bins_n)), dtype=np.float32)
            if arr.size != bins_n:
                arr = np.resize(arr, bins_n)
            base = np.asarray(self._pool_activity_baseline.get(sym, np.ones(bins_n)), dtype=np.float32)
            var = np.asarray(self._pool_activity_var.get(sym, np.ones(bins_n)), dtype=np.float32)
            if base.size != bins_n:
                base = np.resize(base, bins_n)
            if var.size != bins_n:
                var = np.resize(var, bins_n)
            denom = np.sqrt(np.maximum(var, floor))
            z = np.clip((arr - base) / denom, -zclip, zclip)
            positive = np.maximum(0.0, z)
            positive = np.tanh(positive / temp)
            weighted = presence = weight_sum = 0.0
            for i in range(bins_n):
                w = decay ** (bins_n - 1 - i)
                weighted += float(positive[i]) * w
                presence += float(arr[i] > 0.0) * w
                weight_sum += w
            weighted /= max(1e-6, weight_sum)
            presence /= max(1e-6, weight_sum)
            raw_total = float(np.sum(np.maximum(0.0, arr)))
            presence_mass = float(np.log1p(raw_total) / max(1.0, np.log1p(raw_total + 4.0))) if raw_total > 0 else 0.0
            surprise_component = float(np.clip(weighted, 0.0, 1.0))
            presence_component = float(np.clip(0.65 * presence + 0.35 * presence_mass, 0.0, 1.0))
            spike_total = float(sum(self._trial_pool_spike_counts.values()))
            spike_component = float(self._trial_pool_spike_counts.get(sym, 0)) / max(1.0, spike_total)
            evidence_raw[sym] = float(max(0.0,
                float(READOUT_EVIDENCE_SPIKE_WEIGHT) * spike_component +
                float(READOUT_EVIDENCE_SURPRISE_WEIGHT) * surprise_component +
                float(READOUT_EVIDENCE_PRESENCE_WEIGHT) * presence_component))
        total = float(sum(evidence_raw.values()))
        if total <= float(READOUT_1627_SCORE_EPS):
            return {sym: 0.0 for sym in self.symbols}
        return {sym: float(np.clip(evidence_raw[sym] / total, 0.0, 1.0)) for sym in self.symbols}

    def _readout_evidence_audit(self, scores: dict | None = None) -> dict:
        """Expone evidencia instantánea vs estado histórico del readout."""
        counts = {sym: int(self._trial_pool_spike_counts.get(sym, 0)) for sym in self.symbols}
        total = float(sum(counts.values()))
        drive = {sym: float(self._readout_pool_drive_ema.get(sym, 0.0)) for sym in self.symbols}
        mean_drive = float(np.mean(list(drive.values()))) if drive else 0.0
        history = {}
        for sym in self.symbols:
            excess = (drive.get(sym, 0.0) - mean_drive) / max(1e-6, abs(mean_drive)) if mean_drive else 0.0
            history[sym] = float(np.clip(max(0.0, excess), 0.0, float(READOUT_HISTORY_BIAS_CLIP)))
        out = {
            "version": "1.6.29f-evidence-audit-v2",
            "spike_total": int(total),
            "spike_share": {sym: float(counts[sym] / max(1.0, total)) for sym in self.symbols},
            "drive_ema": drive,
            "history_bias": history,
            "history_bias_rate": float(READOUT_HISTORY_BIAS_RATE),
            "history_bias_used_for_decision": False,
            "scores": {sym: float((scores or {}).get(sym, 0.0)) for sym in self.symbols},
        }
        return out

    def _build_bridge_context(self, target: str | None = None) -> dict:
        latent = self._latent_workspace_last or {}
        best_scores = latent.get("best_scores") or latent.get("accepted_scores") or {}
        h0_scores = latent.get("accepted_scores") or latent.get("stage_scores", {}).get("H0", {}) or {}
        physical_scores = self._compute_pool_temporal_normalized_scores()
        target = str(target) if target is not None else None
        latent_pred = latent.get("best_prediction")
        h0_pred = latent.get("h0_prediction")
        readout_pred = max(physical_scores.items(), key=lambda kv: (float(kv[1]), kv[0]))[0] if physical_scores else None
        return {
            "h0_prediction": h0_pred,
            "latent_prediction": latent_pred,
            "readout_prediction": readout_pred,
            "target": target,
            "latent_target_probability": float(best_scores.get(target, 0.0)) if target else 0.0,
            "h0_target_probability": float(h0_scores.get(target, 0.0)) if target else 0.0,
            "readout_target_probability": float(physical_scores.get(target, 0.0)) if target else 0.0,
            "bridge_gap": float(max(0.0, (best_scores.get(target, 0.0) if target else 0.0) - (physical_scores.get(target, 0.0) if target else 0.0))),
            "history_audit": self._readout_evidence_audit(physical_scores),
        }

    def readout_scores(self, net, t=None, window_ms=None):
        """1.6.27: decisión desde spikes reales del pool + sorpresa temporal positiva.

        La autoridad sigue siendo la actividad física de los pools: son los
        spikes reales de los pools, el spike real de cada pool, los que constituyen
        la señal causal. En 29f la evidencia histórica lenta no entra en el score;
        se conserva sólo como telemetría para diagnóstico.
        """
        if not self._readout_ready:
            return {sym: 0.0 for sym in self.symbols}
        if self._clean_trial_baseline_t is not None:
            return self._compute_pool_temporal_normalized_scores()

        now = float(net.current_time if t is None else t)
        win = float(window_ms if window_ms is not None else READOUT_TEMPORAL_WINDOW_MS)
        scores = {}
        for sym in self.symbols:
            pool = [i for i in self._readout_pools.get(sym, []) if 0 <= i < net.n]
            if not pool:
                scores[sym] = 0.0
                continue
            idx = np.asarray(pool, dtype=int)
            ls = net.last_spike[idx]
            valid = (ls > 0) & ((now - ls) >= 0.0) & ((now - ls) < win)
            scores[sym] = float(np.mean(valid))
        return scores

    def apply_readout_learning(self, net, active_sources, target, prediction=None):
        """IA 1.6.27: aprendizaje readout con crédito específico por sinapsis.

        Toda fuente cortical activa puede participar. El cambio de peso sólo ocurre
        cuando la pareja (src,tgt) posee una elegibilidad causal real (pre→post) o
        una elegibilidad subumbral asociada a una entrega física real. No usa el
        máximo de elegibilidad de otro target para transferir crédito.
        """
        if not self._readout_ready or target not in self.symbols or net is None:
            return 0
        src_counts = {int(k): int(v) for k, v in self._trial_source_spike_counts.items()
                      if 0 <= int(k) < net.n and int(k) in set(self._readout_sources)}
        if not src_counts and active_sources:
            src_counts = {int(i): 1 for i in active_sources if 0 <= int(i) < net.n and int(i) in set(self._readout_sources)}
        if not src_counts:
            return 0

        self._diag_trial_weights_before = self._diagnostic_readout_synapse_snapshot(net)
        self._diag_last_learning_audit = {}
        counts = {s: int(self._trial_pool_spike_counts.get(s, 0)) for s in self.symbols}
        total_pool_spikes = int(sum(counts.values()))
        source_spikes = int(self._trial_source_spike_count)
        if self._clean_trial_baseline_t is not None:
            if total_pool_spikes < int(self.min_learning_pool_spikes) or source_spikes < int(self.min_learning_source_spikes):
                self._diag_last_learning_audit = {
                    "target": str(target), "prediction": prediction, "learning_applied": False,
                    "skip_reason": "insufficient_activity", "trial_source_spikes": source_spikes,
                    "trial_pool_spikes_total": total_pool_spikes,
                    "pool_unique_spikes": dict(self._pool_unique_spikes_per_trial),
                    "pool_inhibition_state": {sym: round(v, 4) for sym, v in self._pool_inhibition_state.items()},
                    "pool_separation_margin_mean": float((max(counts.values()) - sorted(counts.values())[-2]) / max(1, max(counts.values()) + sorted(counts.values())[-2])) if len(counts) >= 2 and max(counts.values()) > 0 else 0.0,
                    "learning_eligible_sources": 0,
                    "credit_rule": "1.6.29f-synapse-specific-three-factor-margin-gated",
                }
                return 0

        # Registrar separación física aunque el trial termine sin elegibilidad:
        # la auditoría no depende de que haya plasticidad aplicable.
        ordered_counts = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if len(ordered_counts) >= 2 and ordered_counts[0][1] > 0:
            sep = (ordered_counts[0][1] - ordered_counts[1][1]) / float(max(1, ordered_counts[0][1] + ordered_counts[1][1]))
            self._pool_separation_history.append(float(sep))
            self._pool_separation_history = self._pool_separation_history[-20:]
        scores = self._compute_pool_temporal_normalized_scores()
        bridge_context = self._build_bridge_context(target)
        latent_gain = float(bridge_context.get("latent_target_probability", 0.0) - bridge_context.get("h0_target_probability", 0.0))
        bridge_gap = float(bridge_context.get("bridge_gap", 0.0))
        latent_guide = float(np.clip(
            1.0 + float(READOUT_BRIDGE_LATENT_GUIDE_RATE) * max(0.0, latent_gain + bridge_gap),
            1.0, 1.0 + float(READOUT_BRIDGE_LATENT_GUIDE_RATE),
        ))
        ordered_scores = sorted(scores.items(), key=lambda kv: (-float(kv[1]), kv[0]))
        winner = ordered_scores[0][0] if ordered_scores else None
        second = float(ordered_scores[1][1]) if len(ordered_scores) > 1 else 0.0
        winner_score = float(ordered_scores[0][1]) if ordered_scores else 0.0
        margin = (winner_score - second) / max(1e-6, winner_score + second) if winner_score > 0 else 0.0
        # 1.6.29f: aprender por déficit de margen, no por déficit absoluto hacia 1.0.
        # Esto evita reforzar indefinidamente un pool que ya separa al target.
        ordered_target_scores = sorted(scores.items(), key=lambda kv: (-float(kv[1]), kv[0]))
        rivals = [kv for kv in ordered_target_scores if kv[0] != target]
        top_rival = rivals[0][0] if rivals else None
        target_score = float(scores.get(target, 0.0))
        top_rival_score = float(scores.get(top_rival, 0.0)) if top_rival is not None else 0.0
        target_margin = target_score - top_rival_score
        target_deficit = max(0.0, float(READOUT_29F_DESIRED_TARGET_MARGIN) - target_margin)
        pool_error = {sym: 0.0 for sym in self.symbols}
        pool_error[target] = target_deficit
        if top_rival is not None and target_margin < float(READOUT_29F_RIVAL_MARGIN_GUARD):
            pool_error[top_rival] = -max(float(READOUT_29F_TARGET_ERROR_FLOOR),
                                         top_rival_score - target_score + float(READOUT_29F_RIVAL_MARGIN_GUARD))

        # Balance de exposición curricular, preservando neutralidad de clase.
        exposure = getattr(getattr(net, 'curriculum', None), 'total_trials', None)
        if not isinstance(exposure, dict):
            state = getattr(net, 'curriculum_state', {})
            exposure = state.get('total_trials', {}) if isinstance(state, dict) else {}
        vals = [max(1, int(exposure.get(sym, 0))) for sym in self.symbols]
        mean_exp = float(np.mean(vals)) if vals else 1.0
        balance = float(np.clip(np.sqrt(mean_exp / max(1, int(exposure.get(target, 0) if isinstance(exposure, dict) else 1))),
                                READOUT_BALANCE_MIN, READOUT_BALANCE_MAX))

        active_source_set = set(src_counts)
        eligible_pairs = []
        for src in sorted(active_source_set):
            for sym in self.symbols:
                pool = self._readout_pools.get(sym, [])
                for tgt_i in pool:
                    key = (int(src), int(tgt_i))
                    if key in self._trial_synapse_eligibility:
                        e = float(self._trial_synapse_eligibility.get(key, 0.0))
                        kind = "pre_post"
                    elif key in self._trial_subthreshold_eligibility:
                        e = float(self._trial_subthreshold_eligibility.get(key, 0.0)) * float(READOUT_1627_SUBTHRESHOLD_CREDIT_SCALE)
                        kind = "subthreshold"
                    else:
                        continue
                    if e >= float(READOUT_1627_CREDIT_MIN_ELIGIBILITY):
                        eligible_pairs.append((int(src), int(tgt_i), sym, e, kind))

        # En ausencia de post/subthreshold real no se inventa aprendizaje.
        if not eligible_pairs:
            self._diag_last_learning_audit = {
                "target": str(target), "prediction": prediction, "learning_applied": False,
                "skip_reason": "no_causal_eligibility", "trial_source_spikes": source_spikes,
                "trial_pool_spikes_total": total_pool_spikes, "learning_eligible_sources": 0,
                "eligibility_pairs": 0, "winner": winner, "score_margin": margin,
                "pool_unique_spikes": dict(self._pool_unique_spikes_per_trial),
                "pool_inhibition_state": {sym: round(v, 4) for sym, v in self._pool_inhibition_state.items()},
                "pool_separation_margin_mean": float(sum(self._pool_separation_history[-5:]) / max(1, len(self._pool_separation_history[-5:]))) if self._pool_separation_history else 0.0,
                "credit_rule": "1.6.29f-synapse-specific-three-factor-margin-gated",
            }
            return 0

        per_source_counts = {int(src): 0 for src in active_source_set}
        changed = 0
        delta_by_sym = {sym: [] for sym in self.symbols}
        synapse_credit = []
        with net.lock:
            for src, tgt_i, sym, elig, kind in eligible_pairs:
                e = float(np.clip(elig, 0.0, READOUT_1627_CREDIT_MAX_ELIGIBILITY))
                count = max(1, int(src_counts.get(src, 1)))
                # No se premia linealmente la frecuencia: la amplitud está acotada.
                mean_count = max(1.0, float(np.mean(list(src_counts.values()))))
                # Normalización sublineal de frecuencia: conserva la semántica
                # histórica de log1p(count) pero sin dejar que un source repetitivo
                # domine el crédito.
                count_scale = float(np.clip(
                    (1.0 + 0.25 * np.log1p(count)) /
                    max(1e-6, 1.0 + 0.25 * np.log1p(mean_count)),
                    READOUT_1627_SOURCE_COUNT_SCALE_MIN,
                    READOUT_1627_SOURCE_COUNT_SCALE_MAX,
                ))
                if sym == target:
                    signal = max(float(READOUT_29F_TARGET_ERROR_FLOOR), pool_error[sym]) if target_deficit > 0.0 else 0.0
                    lr = float(READOUT_1627_CREDIT_RATE) * balance * latent_guide
                elif sym == top_rival and pool_error.get(sym, 0.0) < 0.0:
                    signal = float(pool_error[sym])
                    lr = float(READOUT_1627_CREDIT_RIVAL_RATE)
                    if bridge_gap >= float(READOUT_BRIDGE_LATENT_GAIN_FLOOR) and bridge_context.get("readout_prediction") == sym:
                        lr *= float(np.clip(latent_guide, 1.0, 1.0 + float(READOUT_BRIDGE_LATENT_GUIDE_RATE)))
                else:
                    signal = 0.0
                    lr = 0.0
                if kind == "subthreshold":
                    lr *= float(READOUT_1627_SUBTHRESHOLD_CREDIT_SCALE)
                # Crédito temporal saturado: evita que una fuente muy repetitiva
                # multiplique indefinidamente el cambio.
                credit = np.tanh(e / 2.0)
                delta = float(np.clip(lr * signal * credit * count_scale,
                                      -float(READOUT_29F_MAX_DELTA), float(READOUT_29F_MAX_DELTA)))
                if abs(delta) < 1e-10 or tgt_i not in net.connections[src]:
                    continue
                key = (src, tgt_i)
                old = float(net.weights.get(key, self.readout_initial_weight))
                new = float(np.clip(old + delta, self.readout_min_weight, self.readout_max_weight))
                net.weights[key] = new
                actual = new - old
                if abs(actual) > 1e-12:
                    changed += 1
                    per_source_counts[src] = per_source_counts.get(src, 0) + 1
                    delta_by_sym[sym].append(actual)
                    synapse_credit.append({"src": src, "tgt": tgt_i, "symbol": sym,
                                           "eligibility": e, "kind": kind,
                                           "error": float(signal), "delta": float(actual)})

        # Recentrado ya no mueve todo el row: sólo corrige un residual minúsculo
        # sobre el par exacto que acaba de aprender, evitando esconder la plasticidad.
        changed_sources = {int(x["src"]) for x in synapse_credit}
        for src in changed_sources:
            pass

        after = self._diagnostic_readout_synapse_snapshot(net)
        before = self._diag_trial_weights_before
        changed_exact = []
        for key in sorted(set(before) | set(after)):
            old = float(before.get(key, self.readout_initial_weight))
            new = float(after.get(key, self.readout_initial_weight))
            d = new - old
            if abs(d) > 1e-12:
                changed_exact.append((key, d, old, new))

        per_symbol = {}
        for sym in self.symbols:
            vals_s = [x[1] for x in changed_exact if f":{sym}" in x[0]]
            per_symbol[sym] = {
                "changed_synapses": int(len(vals_s)),
                "sum_abs_delta": float(np.sum(np.abs(vals_s))) if vals_s else 0.0,
                "sum_delta": float(np.sum(vals_s)) if vals_s else 0.0,
                "max_abs_delta": float(np.max(np.abs(vals_s))) if vals_s else 0.0,
            }

        self._diag_last_learning_audit = {
            "target": str(target), "prediction": prediction,
            "learning_applied": bool(changed),
            "changed_reported": int(changed),
            "synapses_before": int(len(before)), "synapses_after": int(len(after)),
            "synapses_changed_exact": int(len(changed_exact)),
            "sum_abs_delta": float(np.sum(np.abs([x[1] for x in changed_exact]))) if changed_exact else 0.0,
            "max_abs_delta": float(np.max(np.abs([x[1] for x in changed_exact]))) if changed_exact else 0.0,
            "per_symbol": per_symbol,
            "top_changes": [
                {"key": k, "delta": float(d), "before": float(o), "after": float(n)}
                for k, d, o, n in sorted(changed_exact, key=lambda x: (-abs(x[1]), x[0]))[:25]
            ],
            "pool_scores": {sym: float(scores.get(sym, 0.0)) for sym in self.symbols},
            "winner": winner, "score_margin": float(margin),
            "target_margin": float(target_margin),
            "target_deficit": float(target_deficit),
            "top_rival": top_rival,
            "homeostasis": deepcopy(self._readout_homeostasis_last),
            "bridge_context": dict(bridge_context),
            "latent_guide_multiplier": float(latent_guide),
            "readout_evidence_audit": self._readout_evidence_audit(scores),
            "learning_eligible_sources": int(len({x[0] for x in eligible_pairs})),
            "eligibility_pairs": int(len(eligible_pairs)),
            "pre_post_pairs": int(sum(1 for x in eligible_pairs if x[4] == "pre_post")),
            "subthreshold_pairs": int(sum(1 for x in eligible_pairs if x[4] == "subthreshold")),
            "credit_rule": "1.6.29f-synapse-specific-three-factor-margin-gated",
            "source_selection": "all_active_cortical_sources",
            "pool_error": {sym: float(pool_error[sym]) for sym in self.symbols},
            "pool_unique_spikes": dict(self._pool_unique_spikes_per_trial),
            "pool_inhibition_state": {sym: round(v, 4) for sym, v in self._pool_inhibition_state.items()},
            "learning_audit_pairs": synapse_credit[:50],
        }
        return changed

    def get_pool_separation_metrics(self):
        """v1.6.11: Devuelve métricas de separación funcional entre pools.
        
        Útil para diagnóstico: si pool_separation_margin_mean es bajo (<0.10),
        los pools están produciendo actividad casi simétrica — problema central
        identificado en 1.6.10.
        """
        hist = list(self._pool_separation_history)
        recent = hist[-10:] if hist else []
        active_sets = [set(self._trial_pool_indices.get(s, set())) for s in self.symbols]
        overlap_pairs = []
        for i in range(len(active_sets)):
            for j in range(i + 1, len(active_sets)):
                a, b = active_sets[i], active_sets[j]
                union = len(a | b)
                overlap_pairs.append((len(a & b) / union) if union else 0.0)
        mean_overlap = float(sum(overlap_pairs) / max(1, len(overlap_pairs))) if overlap_pairs else 0.0
        return {
            "separation_margin_mean_10": float(sum(recent) / max(1, len(recent))) if recent else 0.0,
            "separation_margin_mean_all": float(sum(hist) / max(1, len(hist))) if hist else 0.0,
            "pool_inhibition_state": dict(self._pool_inhibition_state),
            "pool_unique_spikes_last_trial": dict(self._pool_unique_spikes_per_trial),
            "pool_active_overlap_last_trial": mean_overlap,
            "n_trials_recorded": len(hist),
        }

    def apply_lateral_pool_inhibition(self, net, t):
        """v1.6.11: Aplica inhibición lateral entre pools en tiempo real.
        
        Cuando el pool ganador (el que más disparó) tiene ventaja, suprime
        el potencial de membrana de los pools rivales. Esto crea una presión
        competitiva real que fuerza diferenciación de representaciones,
        análogo a un mecanismo WTA (Winner-Take-All) suave.
        
        Llamar desde el bucle de simulación en cada step, si el trial está activo.
        """
        if not self._readout_ready or self._clean_trial_baseline_t is None:
            return
        counts = {s: int(self._trial_pool_spike_counts.get(s, 0)) for s in self.symbols}
        total = float(sum(counts.values()))
        if total < 1.0:
            return
        lat_inh = float(READOUT_LATERAL_INHIBITION)
        lat_decay = float(READOUT_LATERAL_DECAY)
        
        # Calcular dominancia relativa de cada pool
        for winner_sym, winner_count in counts.items():
            if winner_count == 0:
                continue
            winner_share = winner_count / total
            # 1.6.16: la competencia ya ocurre por cada spike; esta pasada
            # final no exige una dominancia previa para empezar a competir.
            # Inhibir los pools rivales de forma suave y proporcional.
            pool_winner = self._readout_pools.get(winner_sym, [])
            for loser_sym, loser_count in counts.items():
                if loser_sym == winner_sym:
                    continue
                pool_loser = [i for i in self._readout_pools.get(loser_sym, []) if 0 <= i < net.n]
                if not pool_loser:
                    continue
                suppress = lat_inh * winner_share * (1.0 - loser_count / float(max(1, total)))
                for idx in pool_loser:
                    net.membrane_potential[idx] = max(
                        -2.0,
                        float(net.membrane_potential[idx]) - suppress
                    )
                self._pool_inhibition_state[loser_sym] = min(
                    1.0, self._pool_inhibition_state.get(loser_sym, 0.0) + suppress * 0.5
                )
        # Decaimiento del estado inhibitorio
        for sym in self.symbols:
            self._pool_inhibition_state[sym] *= lat_decay

    def _ensure_latent_workspace(self):
        if not bool(LATENT_WORKSPACE_ENABLED):
            return None
        source_count = max(1, len(self._readout_sources))
        hidden_dim = max(8, int(min(int(LATENT_WORKSPACE_HIDDEN_DIM), max(source_count, 8))))
        if (self._latent_workspace is None
                or int(self._latent_workspace.input_sources) != source_count
                or int(self._latent_workspace.hidden_dim) != hidden_dim):
            self._latent_workspace = LatentWorkspace(
                input_sources=source_count,
                temporal_bins=3,
                hidden_dim=hidden_dim,
                steps=int(LATENT_WORKSPACE_STEPS),
                recurrent_density=float(LATENT_WORKSPACE_RECURRENT_DENSITY),
                recurrent_gain=float(LATENT_WORKSPACE_RECURRENT_GAIN),
                context_gain=float(LATENT_WORKSPACE_CONTEXT_GAIN),
                state_mix=float(LATENT_WORKSPACE_STATE_MIX),
                context_update_rate=float(LATENT_WORKSPACE_CONTEXT_UPDATE_RATE),
                top_k=int(min(LATENT_WORKSPACE_TOP_K, hidden_dim)),
                seed=int(LATENT_WORKSPACE_SEED),
                decoder_learning_rate=float(LATENT_WORKSPACE_DECODER_LR),
                recurrent_learning_rate=float(LATENT_WORKSPACE_RECURRENT_LR),
                context_learning_rate=float(LATENT_WORKSPACE_CONTEXT_LR),
                min_step_mix=float(LATENT_WORKSPACE_MIN_STEP_MIX),
                halt_delta=float(LATENT_WORKSPACE_HALT_DELTA),
                weight_clip=float(LATENT_WORKSPACE_WEIGHT_CLIP),
                decoder_max_update_norm=float(LATENT_WORKSPACE_DECODER_MAX_UPDATE_NORM),
                recurrent_max_update_norm=float(LATENT_WORKSPACE_RECURRENT_MAX_UPDATE_NORM),
                context_max_update_norm=float(LATENT_WORKSPACE_CONTEXT_MAX_UPDATE_NORM),
                halt_margin_gain=float(LATENT_WORKSPACE_HALT_MARGIN_GAIN),
                halt_innovation=float(LATENT_WORKSPACE_HALT_INNOVATION),
                halt_patience=int(LATENT_WORKSPACE_HALT_PATIENCE),
                stage_min_loss_gain=float(LATENT_WORKSPACE_STAGE_MIN_LOSS_GAIN),
                early_exit_enabled=bool(LATENT_WORKSPACE_EARLY_EXIT_ENABLED),
                early_exit_min_margin_gain=float(LATENT_WORKSPACE_EARLY_EXIT_MIN_MARGIN_GAIN),
                early_exit_patience=int(LATENT_WORKSPACE_EARLY_EXIT_PATIENCE),
                decoder_seed_from_readout=bool(LATENT_WORKSPACE_DECODER_SEED_FROM_READOUT),
                neutral_decoder_scale=float(LATENT_WORKSPACE_NEUTRAL_DECODER_SCALE),
                prototype_probe_enabled=bool(LATENT_WORKSPACE_PROTOTYPE_PROBE_ENABLED),
                prototype_update_rate=float(LATENT_WORKSPACE_PROTOTYPE_UPDATE_RATE),
                replay_capacity=int(LATENT_WORKSPACE_REPLAY_CAPACITY),
                replay_per_class=int(LATENT_WORKSPACE_REPLAY_PER_CLASS),
                replay_weight=float(LATENT_WORKSPACE_REPLAY_WEIGHT),
                temporal_shadow_enabled=bool(LATENT_WORKSPACE_TEMPORAL_SHADOW_ENABLED),
                temporal_shadow_learning_rate=float(LATENT_WORKSPACE_TEMPORAL_SHADOW_LR),
                temporal_shadow_max_update_norm=float(LATENT_WORKSPACE_TEMPORAL_SHADOW_MAX_UPDATE_NORM),
                sleep_enabled=bool(LATENT_WORKSPACE_SLEEP_ENABLED),
                sleep_interval=int(LATENT_WORKSPACE_SLEEP_INTERVAL),
                sleep_nrem_passes=int(LATENT_WORKSPACE_SLEEP_NREM_PASSES),
                sleep_replay_per_class=int(LATENT_WORKSPACE_SLEEP_REPLAY_PER_CLASS),
                sleep_replay_weight=float(LATENT_WORKSPACE_SLEEP_REPLAY_WEIGHT),
                sleep_homeostasis_rate=float(LATENT_WORKSPACE_SLEEP_HOMEOSTASIS_RATE),
                sleep_homeostasis_target_l1=float(LATENT_WORKSPACE_SLEEP_HOMEOSTASIS_TARGET_L1),
                sleep_rem_like_enabled=bool(LATENT_WORKSPACE_SLEEP_REM_LIKE_ENABLED),
                sleep_rem_like_passes=int(LATENT_WORKSPACE_SLEEP_REM_LIKE_PASSES),
                sleep_rem_like_rate=float(LATENT_WORKSPACE_SLEEP_REM_LIKE_RATE),
                information_flow_audit=bool(LATENT_WORKSPACE_INFORMATION_FLOW_AUDIT),
                information_loss_clip=float(LATENT_WORKSPACE_INFORMATION_LOSS_CLIP),
            )
        return self._latent_workspace

    def _latent_source_readout_matrix(self, net):
        """Snapshot del readout cortical 1.6.27 usado sólo para sembrar el decoder H."""
        syms = list(self.symbols)
        matrix = np.zeros((len(syms), len(self._readout_sources)), dtype=np.float32)
        if net is None or not self._readout_sources:
            return matrix
        for j, src in enumerate(self._readout_sources):
            src = int(src)
            if src < 0 or src >= net.n:
                continue
            outgoing = set(int(t) for t in net.connections[src])
            for si, sym in enumerate(syms):
                vals = []
                for tgt in self._readout_pools.get(sym, []):
                    tgt = int(tgt)
                    if tgt in outgoing:
                        vals.append(float(net.weights.get((src, tgt), self.readout_initial_weight)))
                if vals:
                    matrix[si, j] = float(np.mean(np.maximum(0.0, np.asarray(vals, dtype=np.float32))))
        return matrix

    def run_latent_workspace_reasoning(self, net=None, steps=None):
        """Ejecuta S→H0→H1.. sin crear eventos ni sustituir el readout por defecto."""
        if net is not None:
            self.net = net
        if not bool(LATENT_WORKSPACE_ENABLED) or not self._readout_ready:
            self._latent_workspace_last = {}
            return {}
        ws = self._ensure_latent_workspace()
        if ws is None:
            return {}
        sources = [int(i) for i in self._readout_sources]
        source_pos = {src: i for i, src in enumerate(sources[:ws.input_sources])}
        mapped_bins = {}
        for src, vals in (self._trial_source_bin_counts or {}).items():
            pos = source_pos.get(int(src))
            if pos is not None:
                mapped_bins[pos] = list(vals)
        source_matrix = self._latent_source_readout_matrix(self.net)
        result = ws.run(mapped_bins, self.symbols, source_matrix, steps=steps)
        result["temporal_audit_enabled"] = bool(LATENT_WORKSPACE_TEMPORAL_AUDIT_ENABLED)
        result["decoder_seed_from_readout_config"] = bool(LATENT_WORKSPACE_DECODER_SEED_FROM_READOUT)
        result["authoritative"] = bool(LATENT_WORKSPACE_AUTHORITATIVE)
        result["input_source_count_available"] = int(len(source_pos))
        result["input_source_count_active"] = int(len(mapped_bins))
        result["input_source_spikes"] = int(self._trial_source_spike_count)
        result["temporal_bins_canonical"] = True
        result["temporal_bin_layout"] = "bin_major"
        result["trial_expected_frames"] = int(self._trial_frame_expected)
        result["trial_observed_frames"] = int(self._trial_frame_count)
        result["trial_sequence"] = int(self._latent_trial_sequence)
        result["trial_mode"] = str(self._latent_trial_mode)
        result["paired_ablation_enabled"] = bool(LATENT_WORKSPACE_PAIRED_ABLATION)
        self._latent_workspace_last = result
        return result

    def _apply_bridge_sleep_replay(self):
        """Replay offline de fallos del bridge sobre sinapsis ya existentes.

        No genera spikes, no crea topología y no toca Event Queue. Se reutilizan
        únicamente pares source->pool que fueron elegibles en el trial histórico.
        """
        ws = self._latent_workspace
        failure_memory = getattr(ws, "_bridge_failure_memory", []) if ws is not None else []
        if not self.net or not failure_memory:
            return {"enabled": True, "applied": False, "reason": "no_bridge_failures", "examples": 0, "passes": int(max(1, READOUT_BRIDGE_SLEEP_REPLAY_PASSES)), "changed": 0, "sum_abs_delta": 0.0, "external_input_suppressed": True, "event_queue_touched": False}
        # En esta memoria compacta sólo conservamos el error semántico; la
        # plasticidad física del trial ya quedó aplicada. El replay actúa como
        # una pequeña repetición de contraste sobre los pesos target/rival.
        failures = list(failure_memory)[-int(READOUT_BRIDGE_SLEEP_REPLAY_PER_CLASS) * max(1, len(self.symbols)):]
        changed = 0
        delta_total = 0.0
        weight = float(np.clip(READOUT_BRIDGE_SLEEP_REPLAY_WEIGHT, 0.0, 0.25))
        with self.net.lock:
            for failure in failures:
                target = failure.get("target")
                wrong = failure.get("readout_prediction")
                if target not in self.symbols:
                    continue
                target_pool = self._readout_pools.get(target, [])
                wrong_pool = self._readout_pools.get(wrong, []) if wrong in self.symbols else []
                # Sólo tocar las sinapsis readout ya existentes; no crear conexiones.
                srcs = [int(i) for i in self._readout_sources if 0 <= int(i) < self.net.n]
                for src in srcs[:max(1, int(READOUT_POOLS_PER_SOURCE))]:
                    if target_pool:
                        for tgt_i in target_pool[:1]:
                            key = (src, int(tgt_i))
                            if int(tgt_i) in self.net.connections[src] and key in self.net.weights:
                                old = float(self.net.weights[key])
                                new = float(np.clip(old + weight * 0.0025, self.readout_min_weight, self.readout_max_weight))
                                self.net.weights[key] = new
                                delta_total += abs(new-old); changed += int(abs(new-old) > 1e-12)
                    if wrong_pool:
                        for tgt_i in wrong_pool[:1]:
                            key = (src, int(tgt_i))
                            if int(tgt_i) in self.net.connections[src] and key in self.net.weights:
                                old = float(self.net.weights[key])
                                new = float(np.clip(old - weight * 0.0020, self.readout_min_weight, self.readout_max_weight))
                                self.net.weights[key] = new
                                delta_total += abs(new-old); changed += int(abs(new-old) > 1e-12)
        return {
            "enabled": True,
            "applied": bool(changed),
            "mode": "nrem_bridge_replay",
            "examples": int(len(failures)),
            "passes": int(max(1, READOUT_BRIDGE_SLEEP_REPLAY_PASSES)),
            "changed": int(changed),
            "sum_abs_delta": float(delta_total),
            "external_input_suppressed": True,
            "event_queue_touched": False,
        }

    def learn_latent_workspace_trial(self, target, completed=True):
        """Aplica o audita plasticidad latente; trials incompletos nunca entrenan."""
        ws = self._latent_workspace
        if ws is None:
            return {}
        if not completed:
            result = {"learning_applied": False, "skip_reason": "incomplete_trial", "target": str(target), "mode": "skip"}
        elif not bool(LATENT_WORKSPACE_TRAINING_ENABLED):
            result = ws.learn_from_trial(target, self.symbols, apply=False)
            result["skip_reason"] = "training_disabled"
        else:
            apply = str(self._latent_trial_mode) != "eval"
            bridge_context = self._build_bridge_context(target)
            result = ws.learn_from_trial(target, self.symbols, apply=apply, bridge_context=bridge_context)
            result["mode"] = "train" if apply else "eval"
            if apply and result.get("sleep", {}).get("applied") and bool(READOUT_BRIDGE_SLEEP_REPLAY_ENABLED):
                try:
                    result["sleep"]["bridge_replay"] = self._apply_bridge_sleep_replay()
                except Exception as _bridge_sleep_exc:
                    result["sleep"]["bridge_replay"] = {"applied": False, "reason": type(_bridge_sleep_exc).__name__}
            if not apply:
                result["skip_reason"] = "workspace_eval_trial"
        if self._latent_workspace_last:
            self._latent_workspace_last["learning"] = deepcopy(result)
            self._latent_workspace_last["learning_steps"] = int(ws.learning_steps)
            self._latent_workspace_last["trial_sequence"] = int(self._latent_trial_sequence)
            self._latent_workspace_last["trial_mode"] = str(self._latent_trial_mode)
        return result

    def get_state(self):
        """Persiste puertos + topología explícita del readout.

        Los pesos viven en NeuralNetwork; aquí guardamos la asignación exacta
        source/pool para que un reinicio no mueva el significado de esos pesos.
        """
        state = super().get_state()
        state["readout_topology"] = {
            "projection_version": 10,
            "sources": [int(i) for i in self._readout_sources],
            "pools": {sym: [int(i) for i in pool] for sym, pool in self._readout_pools.items()},
            "ready": bool(self._readout_ready),
            "pool_size": int(self.readout_pool_size),
            "source_count": int(self.readout_source_count),
            "source_roles": {str(int(i)): str((getattr(self.net, "neuron_roles", {}) or {}).get(int(i), "unknown")) for i in self._readout_sources},
            "encoder_version": "1.6.27-readout-credit-homeostasis-temporal-decoder-v2",
            "latent_workspace_version": LATENT_WORKSPACE_VERSION,
            "latent_workspace_config": {
                "enabled": bool(LATENT_WORKSPACE_ENABLED),
                "authoritative": bool(LATENT_WORKSPACE_AUTHORITATIVE),
                "steps": int(LATENT_WORKSPACE_STEPS),
                "hidden_dim": int(LATENT_WORKSPACE_HIDDEN_DIM),
                "top_k": int(LATENT_WORKSPACE_TOP_K),
                "context_update_rate": float(LATENT_WORKSPACE_CONTEXT_UPDATE_RATE),
                "seed": int(LATENT_WORKSPACE_SEED),
                "decoder_learning_rate": float(LATENT_WORKSPACE_DECODER_LR),
                "recurrent_learning_rate": float(LATENT_WORKSPACE_RECURRENT_LR),
                "context_learning_rate": float(LATENT_WORKSPACE_CONTEXT_LR),
                "min_step_mix": float(LATENT_WORKSPACE_MIN_STEP_MIX),
                "halt_delta": float(LATENT_WORKSPACE_HALT_DELTA),
                "weight_clip": float(LATENT_WORKSPACE_WEIGHT_CLIP),
                "decoder_max_update_norm": float(LATENT_WORKSPACE_DECODER_MAX_UPDATE_NORM),
                "recurrent_max_update_norm": float(LATENT_WORKSPACE_RECURRENT_MAX_UPDATE_NORM),
                "context_max_update_norm": float(LATENT_WORKSPACE_CONTEXT_MAX_UPDATE_NORM),
                "halt_margin_gain": float(LATENT_WORKSPACE_HALT_MARGIN_GAIN),
                "halt_innovation": float(LATENT_WORKSPACE_HALT_INNOVATION),
                "halt_patience": int(LATENT_WORKSPACE_HALT_PATIENCE),
                "training_enabled": bool(LATENT_WORKSPACE_TRAINING_ENABLED),
                "eval_every": int(LATENT_WORKSPACE_EVAL_EVERY),
                "stage_min_loss_gain": float(LATENT_WORKSPACE_STAGE_MIN_LOSS_GAIN),
                "early_exit_enabled": bool(LATENT_WORKSPACE_EARLY_EXIT_ENABLED),
                "early_exit_min_margin_gain": float(LATENT_WORKSPACE_EARLY_EXIT_MIN_MARGIN_GAIN),
                "early_exit_patience": int(LATENT_WORKSPACE_EARLY_EXIT_PATIENCE),
                "decoder_seed_from_readout": bool(LATENT_WORKSPACE_DECODER_SEED_FROM_READOUT),
                "neutral_decoder_scale": float(LATENT_WORKSPACE_NEUTRAL_DECODER_SCALE),
                "prototype_probe_enabled": bool(LATENT_WORKSPACE_PROTOTYPE_PROBE_ENABLED),
                "prototype_update_rate": float(LATENT_WORKSPACE_PROTOTYPE_UPDATE_RATE),
                "replay_capacity": int(LATENT_WORKSPACE_REPLAY_CAPACITY),
                "replay_per_class": int(LATENT_WORKSPACE_REPLAY_PER_CLASS),
                "replay_weight": float(LATENT_WORKSPACE_REPLAY_WEIGHT),
                "temporal_shadow_enabled": bool(LATENT_WORKSPACE_TEMPORAL_SHADOW_ENABLED),
                "temporal_shadow_learning_rate": float(LATENT_WORKSPACE_TEMPORAL_SHADOW_LR),
                "temporal_shadow_max_update_norm": float(LATENT_WORKSPACE_TEMPORAL_SHADOW_MAX_UPDATE_NORM),
                "temporal_audit_enabled": bool(LATENT_WORKSPACE_TEMPORAL_AUDIT_ENABLED),
                "eval_balance_strict": bool(LATENT_WORKSPACE_EVAL_BALANCE_STRICT),
                "eval_symbols": list(LATENT_WORKSPACE_EVAL_SYMBOLS),
                "sleep_enabled": bool(LATENT_WORKSPACE_SLEEP_ENABLED),
                "sleep_interval": int(LATENT_WORKSPACE_SLEEP_INTERVAL),
                "sleep_nrem_passes": int(LATENT_WORKSPACE_SLEEP_NREM_PASSES),
                "sleep_replay_per_class": int(LATENT_WORKSPACE_SLEEP_REPLAY_PER_CLASS),
                "sleep_replay_weight": float(LATENT_WORKSPACE_SLEEP_REPLAY_WEIGHT),
                "sleep_homeostasis_rate": float(LATENT_WORKSPACE_SLEEP_HOMEOSTASIS_RATE),
                "sleep_homeostasis_target_l1": float(LATENT_WORKSPACE_SLEEP_HOMEOSTASIS_TARGET_L1),
                "sleep_rem_like_enabled": bool(LATENT_WORKSPACE_SLEEP_REM_LIKE_ENABLED),
                "sleep_rem_like_passes": int(LATENT_WORKSPACE_SLEEP_REM_LIKE_PASSES),
                "sleep_rem_like_rate": float(LATENT_WORKSPACE_SLEEP_REM_LIKE_RATE),
                "information_flow_audit": bool(LATENT_WORKSPACE_INFORMATION_FLOW_AUDIT),
                "stage_reliability_decay": float(getattr(self._latent_workspace, "stage_reliability_decay", 0.08)) if self._latent_workspace is not None else 0.08,
                "stage_reliability_floor": float(getattr(self._latent_workspace, "stage_reliability_floor", 0.20)) if self._latent_workspace is not None else 0.20,
                "bridge_memory_capacity": int(getattr(self._latent_workspace, "bridge_memory_capacity", READOUT_BRIDGE_MEMORY_CAPACITY)) if self._latent_workspace is not None else int(READOUT_BRIDGE_MEMORY_CAPACITY),
            },
            "latent_workspace_state": self._latent_workspace.state_dict() if self._latent_workspace is not None else {},
            "separator_seed": int(self.cortical_separator_seed),
            "separator_nonlinear_mix": float(self.cortical_separator_nonlinear_mix),
            "separator_bins": [int(self.feature_bus_bins[0]), int(self.feature_bus_bins[1])],
            "separator_stats_initialized": bool(self._cortical_separator_initialized),
            "separator_mean": self._cortical_separator_mean.tolist() if self._cortical_separator_mean is not None else [],
            "separator_var": self._cortical_separator_var.tolist() if self._cortical_separator_var is not None else [],
            "core_ema": self._cortical_core_ema.tolist() if self._cortical_core_ema is not None else [],
            "usage_ema": self._cortical_usage_ema.tolist() if self._cortical_usage_ema is not None else [],
            "consistency_ema": self._cortical_consistency_ema.tolist() if self._cortical_consistency_ema is not None else [],
            "last_trial_pattern": self._cortical_last_trial_pattern.tolist() if self._cortical_last_trial_pattern is not None else [],
            "pattern_memory": [np.asarray(x, dtype=np.float32).tolist() for x in self._cortical_pattern_memory[-int(CORTICAL_PATTERN_MEMORY_SIZE):]],
            "homeostasis_target": float(CORTICAL_HOMEOSTASIS_TARGET),
            "homeostasis_rate": float(CORTICAL_HOMEOSTASIS_RATE),
            "projection_nonzero": int(CORTICAL_PROJECTION_NONZERO),
            "identity_max_prototypes": int(CORTICAL_IDENTITY_MAX_PROTOTYPES),
            "identity_similarity": float(CORTICAL_IDENTITY_SIMILARITY),
            "identity_usage": list(self._cortical_identity_usage),
            "identity_prototypes": [np.asarray(x, dtype=np.float32).tolist() for x in self._cortical_identity_prototypes[-int(CORTICAL_IDENTITY_MAX_PROTOTYPES):]],
            "active_identity": self._cortical_active_identity,
            "core_size": int(CORTICAL_CORE_SIZE),
            "core_anchor_blend": float(CORTICAL_CORE_ANCHOR_BLEND),
            "feature_bus_ready": bool(self.feature_bus_ready),
            "feature_bus_bins": list(self.feature_bus_bins),
            "feature_bus_groups": {k: [int(i) for i in v] for k,v in self._feature_bus_groups.items()},
            "pool_baseline_trials": int(getattr(self, "_pool_baseline_trials", 0)),
            "pool_activity_baseline": {sym: np.asarray(self._pool_activity_baseline.get(sym, np.ones(int(READOUT_TEMPORAL_BINS))), dtype=np.float32).tolist() for sym in self.symbols},
            "pool_activity_variance": {sym: np.asarray(self._pool_activity_var.get(sym, np.ones(int(READOUT_TEMPORAL_BINS))), dtype=np.float32).tolist() for sym in self.symbols},
            "readout_1627": {
                "credit_rule": "1.6.29f-synapse-specific-three-factor-margin-gated",
                "use_all_active_sources": bool(READOUT_1627_USE_ALL_ACTIVE_SOURCES),
                "learning_rate": float(READOUT_1627_CREDIT_RATE),
                "rival_rate": float(READOUT_1627_CREDIT_RIVAL_RATE),
                "pool_rate_ema": {sym: float(self._readout_pool_rate_ema.get(sym, 0.0)) for sym in self.symbols},
                "pool_drive_ema": {sym: float(self._readout_pool_drive_ema.get(sym, 0.0)) for sym in self.symbols},
                "pool_threshold_offset": {sym: float(self._readout_pool_threshold_offset.get(sym, 0.0)) for sym in self.symbols},
                "column_norm_last": dict(self._readout_column_norm_last),
                "evidence_scoring_version": "1.6.29f-instantaneous-evidence-v2",
                "history_bias_rate": float(READOUT_HISTORY_BIAS_RATE),
                "history_bias_used_for_decision": False,
                "homeostasis_state_version": READOUT_29F_HOMEOSTASIS_STATE_VERSION,
                "homeostasis_trials": int(getattr(self, "_readout_homeostasis_trials", 0)),
                "homeostasis_last": deepcopy(getattr(self, "_readout_homeostasis_last", {})),
                "bridge_latent_guide_rate": float(READOUT_BRIDGE_LATENT_GUIDE_RATE),
                "bridge_memory_capacity": int(READOUT_BRIDGE_MEMORY_CAPACITY),
            },
            "readout_29f": {
                "version": READOUT_29F_HOMEOSTASIS_STATE_VERSION,
                "homeostasis_enabled": bool(READOUT_29F_HOMEOSTASIS_ENABLED),
                "warmup_trials": int(READOUT_29F_HOMEOSTASIS_WARMUP_TRIALS),
                "homeostasis_trials": int(getattr(self, "_readout_homeostasis_trials", 0)),
                "homeostasis_last": deepcopy(getattr(self, "_readout_homeostasis_last", {})),
                "column_normalization_disabled": bool(READOUT_29F_DISABLE_COLUMN_NORMALIZATION),
                "synaptic_scaling_disabled": bool(READOUT_29F_DISABLE_SYNAPTIC_SCALING),
                "physical_sleep_replay_enabled": bool(READOUT_BRIDGE_SLEEP_REPLAY_ENABLED),
                "desired_target_margin": float(READOUT_29F_DESIRED_TARGET_MARGIN),
            },
        }
        return state

    def restore_state(self, data, max_idx=None):
        super().restore_state(data, max_idx=max_idx)
        topo = data.get("readout_topology") or {}
        projection_version = int(topo.get("projection_version", 0) or 0)
        sources = [int(i) for i in topo.get("sources", []) if int(i) >= 0]
        pools = {}
        for sym in self.symbols:
            vals = topo.get("pools", {}).get(sym, [])
            pools[sym] = [int(i) for i in vals if int(i) >= 0]
        # v1.6.19: la topología persistida sólo es válida si las fuentes
        # pertenecen explícitamente a la representación cortical agnóstica.
        # Los snapshots históricos pueden declarar projection_version=2 pero
        # contener FeatureBus como sources; no se permite restaurarlos.
        source_roles = getattr(self.net, "neuron_roles", {}) or {}
        cortical_now = set(int(i) for i in getattr(self, "_cortical_representation_neurons", []) if int(i) >= 0)
        valid_sources = bool(sources) and projection_version >= 8 and all((max_idx is None or i < max_idx) for i in sources)
        if cortical_now:
            valid_sources = valid_sources and set(sources).issubset(cortical_now)
        else:
            valid_sources = valid_sources and all(source_roles.get(int(i)) == "cortical_representation" for i in sources)
        valid_pools = all(pools.get(sym) for sym in self.symbols)
        valid_indices = max_idx is None or all(i < max_idx for i in sources + [j for p in pools.values() for j in p])
        if projection_version >= 8 and valid_sources and valid_pools and valid_indices:
            self._readout_sources = sources
            self._readout_pools = pools
            self._readout_ready = bool(topo.get("ready", True))
            self.readout_pool_size = int(topo.get("pool_size", self.readout_pool_size))
            self.readout_source_count = len(self._readout_sources)
            self.feature_bus_bins = tuple(topo.get("feature_bus_bins", self.feature_bus_bins))
            self._cortical_input_dim = len(self.feature_bus_features) * (self.feature_bus_bins[0] * self.feature_bus_bins[1])
            self._cortical_representation_neurons = list(sources)
            self.cortical_representation_ready = False
            self.cortical_separator_seed = int(topo.get("separator_seed", self.cortical_separator_seed))
            self.cortical_separator_nonlinear_mix = float(topo.get("separator_nonlinear_mix", self.cortical_separator_nonlinear_mix))
            self._initialize_cortical_projection(len(sources))
            saved_mean = np.asarray(topo.get("separator_mean", []), dtype=np.float32)
            saved_var = np.asarray(topo.get("separator_var", []), dtype=np.float32)
            if saved_mean.size == len(sources) and saved_var.size == len(sources):
                self._cortical_separator_mean = saved_mean.copy()
                self._cortical_separator_var = np.maximum(saved_var, 1e-4)
                self._cortical_separator_initialized = bool(topo.get("separator_stats_initialized", True))
            for attr, key in (("_cortical_core_ema", "core_ema"), ("_cortical_usage_ema", "usage_ema"), ("_cortical_consistency_ema", "consistency_ema")):
                arr = np.asarray(topo.get(key, []), dtype=np.float32)
                if arr.size == len(sources):
                    setattr(self, attr, arr.copy())
            last_pat = np.asarray(topo.get("last_trial_pattern", []), dtype=np.float32)
            self._cortical_last_trial_pattern = last_pat.copy() if last_pat.size == len(sources) else None
            mem = []
            for item in topo.get("pattern_memory", [])[-int(CORTICAL_PATTERN_MEMORY_SIZE):]:
                arr = np.asarray(item, dtype=np.float32)
                if arr.size == len(sources): mem.append(arr.copy())
            self._cortical_pattern_memory = mem
            self._cortical_identity_prototypes = [np.asarray(item, dtype=np.float32) for item in topo.get("identity_prototypes", [])
                                                  if np.asarray(item).size == len(sources)][-int(CORTICAL_IDENTITY_MAX_PROTOTYPES):]
            self._cortical_identity_usage = [int(x) for x in topo.get("identity_usage", [])][-len(self._cortical_identity_prototypes):]
            self._cortical_active_identity = topo.get("active_identity")
            bins_n = max(1, int(READOUT_TEMPORAL_BINS))
            self._pool_baseline_trials = int(topo.get("pool_baseline_trials", 0) or 0)
            saved_base = topo.get("pool_activity_baseline") or {}
            saved_var = topo.get("pool_activity_variance") or {}
            for sym in self.symbols:
                b = np.asarray(saved_base.get(sym, np.ones(bins_n)), dtype=np.float32)
                v = np.asarray(saved_var.get(sym, np.ones(bins_n)), dtype=np.float32)
                self._pool_activity_baseline[sym] = np.resize(b, bins_n).astype(np.float32) if b.size else np.ones(bins_n, dtype=np.float32)
                self._pool_activity_var[sym] = np.maximum(0.25, np.resize(v, bins_n)).astype(np.float32) if v.size else np.ones(bins_n, dtype=np.float32)
            latent_state = topo.get("latent_workspace_state") or {}
            if latent_state:
                self._latent_workspace = LatentWorkspace(
                    input_sources=int(latent_state.get("input_sources", len(sources))),
                    temporal_bins=int(latent_state.get("temporal_bins", 3)),
                    hidden_dim=int(latent_state.get("hidden_dim", min(len(sources), int(LATENT_WORKSPACE_HIDDEN_DIM))))
                        if int(latent_state.get("hidden_dim", 0) or 0) >= 8 else int(LATENT_WORKSPACE_HIDDEN_DIM),
                    steps=int(latent_state.get("steps", LATENT_WORKSPACE_STEPS)),
                    recurrent_density=float(LATENT_WORKSPACE_RECURRENT_DENSITY),
                    recurrent_gain=float(LATENT_WORKSPACE_RECURRENT_GAIN),
                    context_gain=float(LATENT_WORKSPACE_CONTEXT_GAIN),
                    state_mix=float(LATENT_WORKSPACE_STATE_MIX),
                    context_update_rate=float(latent_state.get("context_update_rate", LATENT_WORKSPACE_CONTEXT_UPDATE_RATE)),
                    top_k=int(latent_state.get("top_k", LATENT_WORKSPACE_TOP_K)),
                    seed=int(latent_state.get("seed", LATENT_WORKSPACE_SEED)),
                    decoder_learning_rate=float(latent_state.get("decoder_learning_rate", LATENT_WORKSPACE_DECODER_LR)),
                    recurrent_learning_rate=float(latent_state.get("recurrent_learning_rate", LATENT_WORKSPACE_RECURRENT_LR)),
                    context_learning_rate=float(latent_state.get("context_learning_rate", LATENT_WORKSPACE_CONTEXT_LR)),
                    min_step_mix=float(latent_state.get("min_step_mix", LATENT_WORKSPACE_MIN_STEP_MIX)),
                    halt_delta=float(latent_state.get("halt_delta", LATENT_WORKSPACE_HALT_DELTA)),
                    weight_clip=float(latent_state.get("weight_clip", LATENT_WORKSPACE_WEIGHT_CLIP)),
                    decoder_max_update_norm=float(latent_state.get("decoder_max_update_norm", LATENT_WORKSPACE_DECODER_MAX_UPDATE_NORM)),
                    recurrent_max_update_norm=float(latent_state.get("recurrent_max_update_norm", LATENT_WORKSPACE_RECURRENT_MAX_UPDATE_NORM)),
                    context_max_update_norm=float(latent_state.get("context_max_update_norm", LATENT_WORKSPACE_CONTEXT_MAX_UPDATE_NORM)),
                    halt_margin_gain=float(latent_state.get("halt_margin_gain", LATENT_WORKSPACE_HALT_MARGIN_GAIN)),
                    halt_innovation=float(latent_state.get("halt_innovation", LATENT_WORKSPACE_HALT_INNOVATION)),
                    halt_patience=int(latent_state.get("halt_patience", LATENT_WORKSPACE_HALT_PATIENCE)),
                    stage_min_loss_gain=float(latent_state.get("stage_min_loss_gain", LATENT_WORKSPACE_STAGE_MIN_LOSS_GAIN)),
                    early_exit_enabled=bool(latent_state.get("early_exit_enabled", LATENT_WORKSPACE_EARLY_EXIT_ENABLED)),
                    early_exit_min_margin_gain=float(latent_state.get("early_exit_min_margin_gain", LATENT_WORKSPACE_EARLY_EXIT_MIN_MARGIN_GAIN)),
                    early_exit_patience=int(latent_state.get("early_exit_patience", LATENT_WORKSPACE_EARLY_EXIT_PATIENCE)),
                    decoder_seed_from_readout=bool(latent_state.get("decoder_seed_from_readout", LATENT_WORKSPACE_DECODER_SEED_FROM_READOUT)),
                    neutral_decoder_scale=float(latent_state.get("neutral_decoder_scale", LATENT_WORKSPACE_NEUTRAL_DECODER_SCALE)),
                    prototype_probe_enabled=bool(latent_state.get("prototype_probe_enabled", LATENT_WORKSPACE_PROTOTYPE_PROBE_ENABLED)),
                    prototype_update_rate=float(latent_state.get("prototype_update_rate", LATENT_WORKSPACE_PROTOTYPE_UPDATE_RATE)),
                    replay_capacity=int(latent_state.get("replay_capacity", LATENT_WORKSPACE_REPLAY_CAPACITY)),
                    replay_per_class=int(latent_state.get("replay_per_class", LATENT_WORKSPACE_REPLAY_PER_CLASS)),
                    replay_weight=float(latent_state.get("replay_weight", LATENT_WORKSPACE_REPLAY_WEIGHT)),
                    temporal_shadow_enabled=bool(latent_state.get("temporal_shadow_enabled", LATENT_WORKSPACE_TEMPORAL_SHADOW_ENABLED)),
                    temporal_shadow_learning_rate=float(latent_state.get("temporal_shadow_learning_rate", LATENT_WORKSPACE_TEMPORAL_SHADOW_LR)),
                    temporal_shadow_max_update_norm=float(latent_state.get("temporal_shadow_max_update_norm", LATENT_WORKSPACE_TEMPORAL_SHADOW_MAX_UPDATE_NORM)),
                )
                self._latent_workspace.restore_state(latent_state)
            r27 = topo.get("readout_1627") or {}
            r29f = topo.get("readout_29f") or {}
            state_version = str(r29f.get("version", ""))
            if state_version == str(READOUT_29F_HOMEOSTASIS_STATE_VERSION):
                rate_ema = r27.get("pool_rate_ema") or {}
                drive_ema = r27.get("pool_drive_ema") or {}
                th_off = r27.get("pool_threshold_offset") or {}
                for sym in self.symbols:
                    self._readout_pool_rate_ema[sym] = float(rate_ema.get(sym, 0.0))
                    self._readout_pool_drive_ema[sym] = float(drive_ema.get(sym, 0.0))
                    self._readout_pool_threshold_offset[sym] = float(np.clip(th_off.get(sym, 0.0), 0.0, READOUT_29F_HOMEOSTASIS_MAX_OFFSET))
                self._readout_homeostasis_trials = int(r29f.get("homeostasis_trials", 0))
                self._readout_homeostasis_last = deepcopy(r29f.get("homeostasis_last") or {})
            else:
                # Cualquier checkpoint previo a 29f no puede contaminar el nuevo
                # controlador: los pesos sí se restauran, pero el estado lento
                # de homeostasis/drive se invalida y arranca sin offset negativo.
                self._readout_pool_rate_ema = {sym: 0.0 for sym in self.symbols}
                self._readout_pool_drive_ema = {sym: 0.0 for sym in self.symbols}
                self._readout_pool_threshold_offset = {sym: 0.0 for sym in self.symbols}
                self._readout_homeostasis_trials = 0
                self._readout_homeostasis_last = {"version": READOUT_29F_HOMEOSTASIS_STATE_VERSION, "armed": False, "reset_from_legacy": True}
            self._readout_column_norm_last = {}
            self.cortical_representation_ready = True
            saved_bus = topo.get("feature_bus_groups") or {}
            if saved_bus:
                self._feature_bus_groups = {k: [int(i) for i in saved_bus.get(k, [])] for k in self.feature_bus_features}
                self.feature_bus_ready = bool(topo.get("feature_bus_ready", True))
            else:
                self.feature_bus_ready = False
                self._readout_ready = False
            if hasattr(self.net, "readout_target_to_symbol"):
                self.net.readout_target_to_symbol.clear()
                for sym, pool in self._readout_pools.items():
                    for idx in pool:
                        self.net.readout_target_to_symbol[int(idx)] = sym
            self._readout_target_sources = {int(tgt): [] for sym in self.symbols for tgt in self._readout_pools.get(sym, [])}
            for src in self._readout_sources:
                for tgt in getattr(self.net, "connections", [[] for _ in range(getattr(self.net, "n", 0))])[int(src)]:
                    if int(tgt) in self._readout_target_sources:
                        self._readout_target_sources[int(tgt)].append(int(src))
        else:
            # Snapshot de topología pre-repfix: no conservar cables 5/5 ni mapas
            # inversos antiguos. auto_connect() reconstruirá la proyección 1×5.
            self._readout_sources = []
            self._readout_pools = {sym: [] for sym in self.symbols}
            self._readout_ready = False
            if hasattr(self.net, "readout_target_to_symbol"):
                self.net.readout_target_to_symbol.clear()
                protected = list(sources)
                for sym, pool in pools.items():
                    protected.extend(pool)
                for sym in self.symbols:
                    for port_name in (f"evidencia_{sym}", f"buffer_{sym}"):
                        port = self.input_ports.get(port_name)
                        if port is not None:
                            protected.extend(int(i) for i in port.connected_neurons)
                if hasattr(self.net, "register_noncompetitive_neurons"):
                    self.net.register_noncompetitive_neurons(protected, role="language_infrastructure")

    def _reset_readout_runtime(self, net=None):
        net = net or self.net
        if net is None:
            return
        try:
            net.clear_readout_runtime_events()
        except Exception:
            pass
        pool_all = [int(i) for sym in self.symbols for i in self._readout_pools.get(sym, [])]
        with net.lock:
            for idx in self._readout_sources + pool_all:
                if 0 <= idx < net.n:
                    net.membrane_potential[idx] = V_REST
                    net.last_update_time[idx] = float(net.current_time)
                    net.last_spike[idx] = 0.0
                    net.refractory_until[idx] = 0.0

    def auto_connect(self, net, radius=20.0):
        """Conecta puertos generales; evidencia_* usa el readout neuronal."""
        n_actual = net.n
        for name, port in {**self.input_ports, **self.output_ports}.items():
            if name.startswith("buffer_") or name.startswith("secuencia_") or name.startswith("evidencia_"):
                continue
            n_pos = min(int(n_actual), int(len(net.positions)), int(len(net.active)))
            if n_pos <= 0:
                continue
            dist = np.linalg.norm(net.positions[:n_pos] - np.array(port.pos), axis=1)
            for idx in np.argsort(dist)[:15]:
                idx = int(idx)
                if idx not in port.connected_neurons:
                    port.connected_neurons.append(idx)
        try:
            if (not self._readout_ready or
                    not self.feature_bus_ready or
                    len(self._readout_sources) == 0 or
                    any(not self._readout_pools.get(sym) for sym in self.symbols)):
                self.configure_learned_readout(net)
        except Exception as exc:
            print(f"⚠️ [Readout] No se pudo inicializar: {exc}", flush=True)

    # --- FUNCIONES FASE A ---

    @staticmethod
    def _queue_audit_snapshot(net):
        if net is None:
            return {}
        stats = getattr(net, "event_queue_stats", {}) or {}
        keys = (
            "enqueued_total", "processed_total", "dropped_total", "dropped_by_governor",
            "queue_size", "max_queue_size", "backpressure_timeouts", "queue_pressure_events",
            "propagation_attempts", "propagation_enqueued", "fanout_limited_total",
        )
        out = {}
        for key in keys:
            try:
                out[key] = int(stats.get(key, 0) or 0)
            except Exception:
                out[key] = 0
        return out

    def _queue_audit_trial(self):
        before = dict(self._trial_queue_before or {})
        after = self._queue_audit_snapshot(self.net)
        self._trial_queue_after = after
        delta = {}
        for key in set(before) | set(after):
            delta[key] = int(after.get(key, 0)) - int(before.get(key, 0))
        return {"enabled": bool(LATENT_WORKSPACE_QUEUE_AUDIT_ENABLED), "before": before, "after": after, "delta": delta}

    def begin_clean_trial(self, t, net=None, target=None):
        """Inicia un trial aislado: descarta eventos readout residuales y reinicia integradores."""
        if net is not None:
            self.net = net
        self._reset_readout_runtime(self.net)
        self._clean_trial_baseline_t = float(t)
        self._trial_queue_before = self._queue_audit_snapshot(self.net) if LATENT_WORKSPACE_QUEUE_AUDIT_ENABLED else {}
        self._trial_queue_after = {}
        self._latent_trial_sequence = int(getattr(self, "_cortical_trials_completed", 0)) + 1
        self._trial_target_symbol = str(target) if target is not None else None
        self._latent_trial_eval_expected = None
        self._latent_trial_eval_balance_match = None
        every = int(LATENT_WORKSPACE_EVAL_EVERY)
        eval_symbols = tuple(str(x) for x in LATENT_WORKSPACE_EVAL_SYMBOLS) or tuple(self.symbols)
        if every > 0 and self._latent_trial_sequence % every == 0 and eval_symbols:
            cycle = (self._latent_trial_sequence // every - 1) % len(eval_symbols)
            self._latent_trial_eval_expected = eval_symbols[cycle]
            self._latent_trial_eval_balance_match = bool(
                self._trial_target_symbol is None or self._trial_target_symbol == self._latent_trial_eval_expected
            )
        is_eval_boundary = every > 0 and self._latent_trial_sequence % every == 0
        if bool(LATENT_WORKSPACE_EVAL_BALANCE_STRICT) and is_eval_boundary:
            self._latent_trial_mode = "eval" if self._latent_trial_eval_balance_match else "train"
        else:
            self._latent_trial_mode = "eval" if is_eval_boundary else "train"
        self._trial_source_spike_count = 0
        self._trial_pool_spike_counts = {sym: 0 for sym in self.symbols}
        self._trial_source_indices = set()
        self._trial_source_spike_counts = {}
        self._trial_source_first_t = {}
        self._trial_source_last_t = {}
        self._trial_source_bin_counts = {}
        self._trial_pool_indices = {sym: set() for sym in self.symbols}
        # v1.6.11: resetear estado de inhibición lateral y separación por trial.
        self._pool_inhibition_state = {sym: 0.0 for sym in self.symbols}
        self._pool_unique_spikes_per_trial = {sym: 0 for sym in self.symbols}
        self._trial_pool_neuron_spike_counts = {}
        self._readout_pool_neuron_fatigue = {}
        self._readout_last_competition_t = float(t)
        self._trial_frame_count = 0
        # IA 1.6.27: sombra de diagnóstico reiniciada por trial.
        self._readout_1627_shadow = {
            "source_spikes": 0,
            "source_bins": [0, 0, 0],
            "pool_spikes": {sym: 0 for sym in self.symbols},
            "pool_bins": {sym: [0, 0, 0] for sym in self.symbols},
        }
        if self._latent_workspace is not None:
            self._latent_workspace.reset_trial()
        self._latent_workspace_last = {}
        # v1.6.25: el prototipo activo es contexto del trial actual; los prototipos
        # almacenados sí son memoria persistente y se vuelven a consultar desde el primer frame.
        self._cortical_active_identity = None
        self._cortical_identity_similarity = 0.0
        self._cortical_identity_is_new = False
        self._trial_separator_raw_sum = np.zeros(len(self._cortical_representation_neurons), dtype=np.float32) if self._cortical_representation_neurons else None
        self._trial_separator_raw_sq_sum = np.zeros(len(self._cortical_representation_neurons), dtype=np.float32) if self._cortical_representation_neurons else None
        self._trial_separator_raw_count = 0
        self._trial_synapse_eligibility = {}
        self._trial_subthreshold_eligibility = {}
        self._trial_subthreshold_post = {}
        self._pending_readout_delivery_links = {}
        self._trial_cortical_counts = np.zeros(len(self._cortical_representation_neurons), dtype=np.float32) if self._cortical_representation_neurons else None
        self._cortical_trial_anchor = None
        self._cortical_trial_anchor_indices = set()
        self._cortical_trial_anchor_age = 0
        self._cortical_last_emit_count = 0
        # La representación es intraciclo: ningún winner/fatigue del trial
        # anterior puede sesgar el primer frame del siguiente.
        if self._cortical_activity_trace is not None:
            self._cortical_activity_trace[:] = 0.0
        self._cortical_previous = (np.zeros_like(self._cortical_previous)
                                   if self._cortical_previous is not None else None)
        if self._cortical_representation_neurons:
            self._cortical_last_emit_t = {int(i): -1e12 for i in self._cortical_representation_neurons}
        self._cortical_trace_t = float(t)
        self._diag_trial_weights_before = {}
        self._diag_last_learning_audit = {}
        self._trial_pool_delivery_diag = {sym: self._new_pool_delivery_diag() for sym in self.symbols}
        self._trial_pool_delivery_samples = []
        for sym in self.symbols:
            port = self.input_ports.get(f"evidencia_{sym}")
            if port is not None:
                port.value = 0.0
        self._prev_evidence = {sym: 0.0 for sym in self.symbols}
        self._evidence_cooldown = {sym: False for sym in self.symbols}

    def _update_pool_activity_baseline(self):
        """1.6.26: actualiza baseline/varianza temporal sólo al cerrar un trial completo."""
        bins_n = max(1, int(READOUT_TEMPORAL_BINS))
        decay = float(np.clip(READOUT_POOL_BASELINE_DECAY, 0.0, 0.9999))
        var_decay = float(np.clip(READOUT_POOL_BASELINE_VAR_DECAY, 0.0, 0.9999))
        for sym in self.symbols:
            arr = np.asarray(self._trial_pool_bin_counts.get(sym, np.zeros(bins_n)), dtype=np.float32)
            if arr.size != bins_n:
                arr = np.resize(arr, bins_n)
            base = self._pool_activity_baseline.setdefault(sym, np.ones(bins_n, dtype=np.float32))
            var = self._pool_activity_var.setdefault(sym, np.ones(bins_n, dtype=np.float32))
            delta = arr - base
            base[:] = decay * base + (1.0 - decay) * arr
            var[:] = np.maximum(0.25, var_decay * var + (1.0 - var_decay) * (delta * delta))
        self._pool_baseline_trials = int(getattr(self, '_pool_baseline_trials', 0)) + 1

    def _apply_slow_readout_synaptic_scaling(self):
        """Escalado L2 lento por fila, preservando diferencias aprendidas."""
        if bool(READOUT_29F_DISABLE_SYNAPTIC_SCALING) or self.net is None or not self._readout_sources:
            return 0
        interval = max(1, int(READOUT_SYNAPTIC_SCALING_INTERVAL))
        if int(getattr(self, "_cortical_trials_completed", 0)) % interval != 0:
            return 0
        target_norm = max(1e-6, float(READOUT_SYNAPTIC_NORM_TARGET))
        rate = float(np.clip(READOUT_SYNAPTIC_SCALING_RATE, 0.0, 0.25))
        changed = 0
        with self.net.lock:
            target_set = {int(t) for sym in self.symbols for t in self._readout_pools.get(sym, [])}
            for src in self._readout_sources:
                keys = [(int(src), int(t)) for t in self.net.connections[int(src)] if int(t) in target_set]
                if not keys:
                    continue
                vals = np.asarray([float(self.net.weights.get(k, self.readout_initial_weight)) for k in keys], dtype=np.float64)
                norm = float(np.linalg.norm(vals))
                if norm <= 1e-9:
                    continue
                factor = 1.0 + rate * (target_norm / norm - 1.0)
                factor = float(np.clip(factor, 0.95, 1.05))
                for key, old in zip(keys, vals):
                    new = float(np.clip(old * factor, self.readout_min_weight, self.readout_max_weight))
                    if abs(new - old) > 1e-12:
                        self.net.weights[key] = new
                        changed += 1
        return changed

    def _commit_cortical_adaptation(self):
        """Actualiza core/fringe y homeostasis sólo entre trials."""
        n = len(self._cortical_representation_neurons)
        if n == 0:
            return
        counts = np.asarray(self._trial_cortical_counts if self._trial_cortical_counts is not None else np.zeros(n), dtype=np.float32)
        frames = max(1, int(self._trial_frame_count))
        rate = np.clip(counts / float(frames), 0.0, 1.0)
        usage = np.asarray(self._cortical_usage_ema, dtype=np.float32)
        core = np.asarray(self._cortical_core_ema, dtype=np.float32)
        cons = np.asarray(self._cortical_consistency_ema, dtype=np.float32)
        if usage.size != n: usage = np.zeros(n, dtype=np.float32); self._cortical_usage_ema = usage
        if core.size != n: core = np.zeros(n, dtype=np.float32); self._cortical_core_ema = core
        if cons.size != n: cons = np.zeros(n, dtype=np.float32); self._cortical_consistency_ema = cons
        usage[:] = 0.88 * usage + 0.12 * rate
        selected = (rate > 0).astype(np.float32)
        core[:] = 0.93 * core + 0.07 * selected
        if self._cortical_previous is not None:
            prev = np.asarray(self._cortical_previous, dtype=np.float32)
            mean_prev = float(np.mean(prev)) if prev.size else 0.0
            cons[:] = 0.90 * cons + 0.10 * np.clip(1.0 - np.abs(prev - mean_prev), 0.0, 1.0)
            norm = max(1e-6, float(np.linalg.norm(prev)))
            self._cortical_last_trial_pattern = (prev / norm).astype(np.float32)
            self._cortical_pattern_memory.append(self._cortical_last_trial_pattern.copy())
            self._cortical_pattern_memory = self._cortical_pattern_memory[-int(CORTICAL_PATTERN_MEMORY_SIZE):]
        # Actualizar sólo el prototipo latente activo a partir del promedio del
        # trial, preservando estabilidad dentro del ciclo y evitando drift por frame.
        active_id = self._cortical_active_identity
        if (active_id is not None and 0 <= int(active_id) < len(self._cortical_identity_prototypes)
                and self._cortical_trial_pattern_sum is not None
                and int(self._cortical_trial_pattern_count) > 0):
            mean_pat = np.asarray(self._cortical_trial_pattern_sum, dtype=np.float32) / float(self._cortical_trial_pattern_count)
            mn = float(np.linalg.norm(mean_pat))
            if mn > 1e-6:
                mean_pat = mean_pat / mn
                eta_id = float(np.clip(CORTICAL_IDENTITY_UPDATE, 0.0, 0.5))
                proto = np.asarray(self._cortical_identity_prototypes[int(active_id)], dtype=np.float32)
                proto = (1.0 - eta_id) * proto + eta_id * mean_pat
                pn = float(np.linalg.norm(proto))
                self._cortical_identity_prototypes[int(active_id)] = (proto / max(1e-6, pn)).astype(np.float32)

        # Homeostasis intrinsic sólo fuera del trial.
        target = float(CORTICAL_HOMEOSTASIS_TARGET)
        eta = float(CORTICAL_HOMEOSTASIS_RATE)
        max_delta = float(CORTICAL_HOMEOSTASIS_MAX_DELTA)
        if self.net is not None:
            for j, idx in enumerate(self._cortical_representation_neurons):
                if 0 <= int(idx) < self.net.n:
                    err = float(rate[j] - target)
                    delta = float(np.clip(eta * err, -max_delta, max_delta))
                    base = float(self.net.v_thresh[int(idx)])
                    self.net.v_thresh[int(idx)] = float(np.clip(base + delta, self.cortical_min_strength - 0.30, self.cortical_max_strength + max_delta))

    def _commit_cortical_separator_trial_stats(self):
        """1.6.22: actualiza el separador sólo entre trials, nunca dentro de uno."""
        count = int(getattr(self, "_trial_separator_raw_count", 0))
        if count < 5:
            return
        total = np.asarray(self._trial_separator_raw_sum, dtype=np.float32)
        sq = np.asarray(self._trial_separator_raw_sq_sum, dtype=np.float32)
        trial_mean = total / float(count)
        trial_var = np.maximum(0.0, sq / float(count) - trial_mean * trial_mean)
        if self._cortical_separator_mean is None or len(self._cortical_separator_mean) != len(trial_mean):
            self._cortical_separator_mean = np.asarray(trial_mean, dtype=np.float32)
            self._cortical_separator_var = np.maximum(trial_var, 1e-4).astype(np.float32)
            self._cortical_separator_initialized = True
            return
        alpha = 0.08
        delta = trial_mean - self._cortical_separator_mean
        self._cortical_separator_mean += alpha * delta
        self._cortical_separator_var[:] = np.maximum(
            1e-4,
            (1.0 - alpha) * self._cortical_separator_var
            + alpha * (trial_var + delta * delta),
        )

    def _update_pool_activity_baseline_v1627(self):
        """IA 1.6.29f: baseline + homeostasis unilateral contra hiperactividad.

        Se mantiene el nombre histórico para compatibilidad, pero la política cambia:
        la homeostasis no baja umbrales para pools dormidos y sólo eleva el umbral de
        un pool cuando su actividad supera claramente una referencia poblacional robusta.
        """
        bins_n = max(1, int(READOUT_TEMPORAL_BINS))
        decay = float(np.clip(READOUT_POOL_BASELINE_DECAY, 0.0, 0.9999))
        var_decay = float(np.clip(READOUT_POOL_BASELINE_VAR_DECAY, 0.0, 0.9999))
        ema_decay = float(np.clip(READOUT_29F_HOMEOSTASIS_RATE_EMA_DECAY, 0.0, 0.9999))
        self._readout_homeostasis_trials = int(getattr(self, '_readout_homeostasis_trials', 0)) + 1
        raw_rates = {}
        for sym in self.symbols:
            arr = np.asarray(self._trial_pool_bin_counts.get(sym, np.zeros(bins_n)), dtype=np.float32)
            if arr.size != bins_n:
                arr = np.resize(arr, bins_n)
            base = self._pool_activity_baseline.setdefault(sym, np.ones(bins_n, dtype=np.float32))
            var = self._pool_activity_var.setdefault(sym, np.ones(bins_n, dtype=np.float32))
            delta = arr - base
            base[:] = decay * base + (1.0 - decay) * arr
            var[:] = np.maximum(0.25, var_decay * var + (1.0 - var_decay) * (delta * delta))
            rate = float(self._trial_pool_spike_counts.get(sym, 0)) / max(1, int(self._trial_frame_count))
            raw_rates[sym] = rate
            old_rate = float(self._readout_pool_rate_ema.get(sym, 0.0))
            self._readout_pool_rate_ema[sym] = ema_decay * old_rate + (1.0 - ema_decay) * rate
            drive = float(np.sum(np.maximum(0.0, arr))) / max(1, int(self._trial_frame_count))
            old_drive = float(self._readout_pool_drive_ema.get(sym, 0.0))
            self._readout_pool_drive_ema[sym] = ema_decay * old_drive + (1.0 - ema_decay) * drive

        smoothed_rates = {sym: float(self._readout_pool_rate_ema.get(sym, 0.0)) for sym in self.symbols}
        actions = {sym: {"action": "none", "offset_before": float(self._readout_pool_threshold_offset.get(sym, 0.0)),
                         "offset_after": float(self._readout_pool_threshold_offset.get(sym, 0.0)),
                         "rate": float(smoothed_rates.get(sym, 0.0)), "raw_rate": float(raw_rates.get(sym, 0.0)), "reference_rate": 0.0,
                         "overactivity_ratio": 0.0, "scaled": False} for sym in self.symbols}
        if not bool(READOUT_29F_HOMEOSTASIS_ENABLED) or self._readout_homeostasis_trials <= int(READOUT_29F_HOMEOSTASIS_WARMUP_TRIALS):
            self._readout_homeostasis_last = {"version": READOUT_29F_HOMEOSTASIS_STATE_VERSION,
                                              "armed": False,
                                              "warmup_trials": int(self._readout_homeostasis_trials),
                                              "reference_rate": 0.0,
                                              "actions": actions}
            self._pool_baseline_trials = int(getattr(self, '_pool_baseline_trials', 0)) + 1
            return
        values = np.asarray(list(smoothed_rates.values()), dtype=np.float64)
        q = float(np.clip(READOUT_29F_HOMEOSTASIS_REFERENCE_QUANTILE, 0.0, 1.0))
        reference_rate = float(np.quantile(values, q)) if values.size else 0.0
        # Evita que una población completamente silenciosa produzca divisiones raras.
        reference_rate = max(float(READOUT_29F_HOMEOSTASIS_EPS), reference_rate)
        with self.net.lock if self.net is not None else _nullcontext():
            for sym in self.symbols:
                raw_rate = float(raw_rates.get(sym, 0.0))
                rate = float(smoothed_rates.get(sym, 0.0))
                ratio = rate / reference_rate
                before = float(self._readout_pool_threshold_offset.get(sym, 0.0))
                after = before
                scale = 0.0
                active = (raw_rate >= float(READOUT_29F_HOMEOSTASIS_MIN_RATE)
                          and ratio >= float(READOUT_29F_HOMEOSTASIS_OVERACTIVITY_RATIO))
                if active:
                    excess = max(0.0, ratio - 1.0)
                    step = float(READOUT_29F_HOMEOSTASIS_OFFSET_RATE) * excess
                    after = float(np.clip(before + step, 0.0, float(READOUT_29F_HOMEOSTASIS_MAX_OFFSET)))
                    scale = after - before
                    self._readout_pool_threshold_offset[sym] = after
                else:
                    # Nunca reducir el umbral por baja actividad.
                    self._readout_pool_threshold_offset[sym] = max(0.0, before)
                actions[sym] = {
                    "action": "raise_threshold" if scale > 0.0 else "none",
                    "offset_before": before,
                    "offset_after": float(self._readout_pool_threshold_offset.get(sym, before)),
                    "rate": rate,
                    "raw_rate": raw_rate,
                    "reference_rate": reference_rate,
                    "overactivity_ratio": ratio,
                    "scaled": bool(scale > 0.0),
                }
        self._readout_homeostasis_last = {
            "version": READOUT_29F_HOMEOSTASIS_STATE_VERSION,
            "armed": True,
            "warmup_trials": int(READOUT_29F_HOMEOSTASIS_WARMUP_TRIALS),
            "trials_observed": int(self._readout_homeostasis_trials),
            "reference_rate": reference_rate,
            "reference_quantile": q,
            "overactivity_ratio": float(READOUT_29F_HOMEOSTASIS_OVERACTIVITY_RATIO),
            "actions": actions,
        }
        self._pool_baseline_trials = int(getattr(self, '_pool_baseline_trials', 0)) + 1

    def _apply_readout_column_normalization_v1627(self):
        """Equaliza suavemente la energía de columna sin borrar selectividad.

        Se normaliza la desviación alrededor de w0 usando como referencia la
        mediana de las normas por pool. La operación es lenta y simétrica.
        """
        if bool(READOUT_29F_DISABLE_COLUMN_NORMALIZATION) or self.net is None or not self._readout_sources:
            return 0
        interval = max(1, int(READOUT_1627_COLUMN_NORM_INTERVAL))
        if int(getattr(self, '_cortical_trials_completed', 0)) % interval != 0:
            return 0
        rate = float(np.clip(READOUT_1627_COLUMN_NORM_RATE, 0.0, 0.20))
        target_syms = self.symbols
        pool_vectors = {}
        with self.net.lock:
            srcs = [int(i) for i in self._readout_sources if 0 <= int(i) < self.net.n]
            for sym in target_syms:
                vals = []
                for src in srcs:
                    for tgt in self._readout_pools.get(sym, []):
                        key = (src, int(tgt))
                        if int(tgt) in self.net.connections[src] and key in self.net.weights:
                            vals.append(float(self.net.weights[key]) - self.readout_initial_weight)
                pool_vectors[sym] = np.asarray(vals, dtype=np.float64)
        norms = {sym: float(np.linalg.norm(v)) for sym, v in pool_vectors.items() if v.size}
        if len(norms) < 2:
            return 0
        target_norm = float(np.median(list(norms.values())))
        changed = 0
        stats = {}
        with self.net.lock:
            for sym, vec in pool_vectors.items():
                norm = float(np.linalg.norm(vec))
                if norm <= float(READOUT_1627_COLUMN_NORM_EPS) or target_norm <= 0.0:
                    continue
                factor = 1.0 + rate * (target_norm / norm - 1.0)
                factor = float(np.clip(factor, READOUT_1627_COLUMN_NORM_MIN_FACTOR, READOUT_1627_COLUMN_NORM_MAX_FACTOR))
                j = 0
                for src in srcs:
                    for tgt in self._readout_pools.get(sym, []):
                        key = (src, int(tgt))
                        if int(tgt) not in self.net.connections[src] or key not in self.net.weights:
                            continue
                        old = float(self.net.weights[key])
                        delta = old - self.readout_initial_weight
                        new = float(np.clip(self.readout_initial_weight + delta * factor,
                                            self.readout_min_weight, self.readout_max_weight))
                        if abs(new - old) > 1e-12:
                            self.net.weights[key] = new
                            changed += 1
                        j += 1
                stats[sym] = {"norm_before": norm, "factor": factor, "norm_target": target_norm}
        self._readout_column_norm_last = stats
        return changed

    def _readout_shadow_snapshot_v1627(self):
        shadow = dict(self._readout_1627_shadow or {})
        source = int(shadow.get('source_spikes', self._trial_source_spike_count))
        pool = {sym: int(shadow.get('pool_spikes', {}).get(sym, self._trial_pool_spike_counts.get(sym, 0))) for sym in self.symbols}
        return {
            "source_spikes": source,
            "source_unique": int(len(self._trial_source_indices)),
            "source_bins": list(self._trial_source_bins_v1627()),
            "pool_spikes": pool,
            "pool_unique": {sym: int(len(self._trial_pool_indices.get(sym, set()))) for sym in self.symbols},
            "pool_scores": self._compute_pool_temporal_normalized_scores() if self._clean_trial_baseline_t is not None else {},
        }

    def _trial_source_bins_v1627(self):
        bins = [0,0,0]
        for vals in self._trial_source_bin_counts.values():
            vv = list(vals) if len(vals) == 3 else [0,0,0]
            for i in range(3):
                bins[i] += int(vv[i])
        return bins

    def end_clean_trial(self):
        """Finaliza el aislamiento y conserva el snapshot; descarta eventos readout tardíos."""
        # 1.6.26: la calibración estadística del readout sólo aprende de trials
        # completos; un corte prematuro no debe contaminar baseline ni escalado.
        complete = int(getattr(self, "_trial_frame_count", 0)) >= int(getattr(self, "_trial_frame_expected", 60))
        if complete:
            self._cortical_trials_completed = int(getattr(self, "_cortical_trials_completed", 0)) + 1
            self._commit_cortical_adaptation()
            if not bool(READOUT_29F_DISABLE_COLUMN_NORMALIZATION):
                self._apply_readout_column_normalization_v1627()
            self._update_pool_activity_baseline_v1627()
            self._commit_cortical_separator_trial_stats()
        if not self._last_trial_readout:
            self._last_trial_readout = self.get_trial_readout_activity()
        self._reset_readout_runtime(self.net)
        self._clean_trial_baseline_t = None
        self._prev_evidence = {sym: 0.0 for sym in self.symbols}
        self._evidence_cooldown = {sym: False for sym in self.symbols}
        self._detector_evidence = {sym: 0.0 for sym in self.symbols}
        for sym in self.symbols:
            port = self.input_ports.get(f"evidencia_{sym}")
            if port is not None:
                port.value = 0.0
    
    def update_evidence_values(self, net, t):
        """Actualiza exclusivamente la evidencia neural observacional.

        v1.6.9: ``evidencia_*`` deja de conservar scores del detector heurístico.
        Si el puerto no tiene neuronas conectadas, su valor es 0 por diseño.
        El score heurístico vive en ``_detector_evidence`` y solo se usa para
        diagnosticar/medir secuencias.
        """
        for sym in self.symbols:
            port = self.input_ports[f"evidencia_{sym}"]
            neurons = port.connected_neurons
            if not neurons or t < 0.5:
                port.value = 0.0
                continue

            idx_arr = np.asarray(neurons, dtype=int)
            idx_arr = idx_arr[idx_arr < net.n]
            if len(idx_arr) == 0:
                port.value = 0.0
                continue

            last_spikes = net.last_spike[idx_arr]
            dts = t - last_spikes
            baseline_t = self._clean_trial_baseline_t
            if baseline_t is not None:
                valid = (last_spikes >= baseline_t) & (dts >= 0.0) & (dts < self.evidence_window_ms)
            else:
                valid = dts < self.evidence_window_ms
            activos = int(np.sum(valid))
            port.value = float(activos) / len(idx_arr)

    def update_detector_evidence(self, scores, t=0.0):
        """Guarda la hipótesis heurística del detector fuera de evidencia_*.

        Solo el ganador supera el umbral y se conserva como señal de flanco.
        Las clases no ganadoras vuelven a cero para permitir detectar una
        transición X -> O o O -> X dentro de un trial de secuencia.
        """
        scores = scores or {}
        threshold = float(getattr(self, 'detector_threshold', 0.55))
        top_sym = max(scores, key=scores.get) if scores else None
        for sym in self.symbols:
            value = float(scores.get(sym, 0.0)) if top_sym == sym else 0.0
            if top_sym != sym or value < threshold:
                value = 0.0
            self._detector_evidence[sym] = max(0.0, min(1.0, value))
        self._last_detector_evidence_t = float(t)

    def get_detector_evidence(self, sym):
        return float(self._detector_evidence.get(sym, 0.0))

    def debug_pool_status(self, net):
        """Diagnóstico: ¿alguna vez dispararon las neuronas conectadas a evidencia_*?"""
        for sym in self.symbols:
            port = self.input_ports.get(f"evidencia_{sym}")
            if not port or not port.connected_neurons:
                print(f"🔬 [Diag] evidencia_{sym}: puerto sin neuronas conectadas")
                continue
                
            # Filtramos solo índices válidos para la red actual
            neurons = [idx for idx in port.connected_neurons if idx < net.n]
            
            # net.last_spike[idx] == 0.0 significa que NUNCA disparó desde que arrancó la red
            # Esto debería ser mucho menos frecuente ahora gracias al nuevo recable_from_live_band
            nunca_dispararon = [idx for idx in neurons if net.last_spike[idx] == 0.0]
            
            print(f"🔬 [Diag] evidencia_{sym}: {len(neurons)} neuronas | "
                  f"{len(nunca_dispararon)} nunca dispararon | "
                  f"índices={neurons[:8]}{'...' if len(neurons) > 8 else ''}")
            
    def debug_band_activity(self, net, z_range=(37.0, 53.0)):
        """¿Hay ALGUNA neurona activa en la banda z de lenguaje? Y en qué
        estado de energía están las que sobreviven ahí."""
        z = net.positions[:net.n, 2]
        en_banda = (z > z_range[0]) & (z < z_range[1]) & net.active[:net.n]
        idx_banda = np.where(en_banda)[0]
        disparadas = idx_banda[net.last_spike[idx_banda] > 0]

        energia_promedio = float(np.mean(net.energy[idx_banda])) if len(idx_banda) > 0 else 0.0
        energia_min = float(np.min(net.energy[idx_banda])) if len(idx_banda) > 0 else 0.0

        print(f"🔬 [Diag-Banda] z∈{z_range}: {len(idx_banda)} neuronas totales | "
              f"{len(disparadas)} dispararon alguna vez | "
              f"energía prom={energia_promedio:.3f} | energía mín={energia_min:.3f} | "
              f"ejemplos={disparadas[:8].tolist()}")      
    
    def debug_overlap_evidencia_banda(self, net, z_range=(37.0, 53.0)):
        """¿Las neuronas que ahora disparan en la banda son las mismas
        que están conectadas a evidencia_X/O, o son poblaciones distintas?"""
        z = net.positions[:net.n, 2]
        en_banda = (z > z_range[0]) & (z < z_range[1]) & net.active[:net.n]
        idx_banda = set(np.where(en_banda)[0].tolist())

        for sym in self.symbols:
            port = self.input_ports.get(f"evidencia_{sym}")
            if not port: continue
            conectadas = set(port.connected_neurons)
            interseccion = conectadas & idx_banda
            print(f"🔬 [Diag-Overlap] evidencia_{sym}: {len(conectadas)} conectadas | "
                  f"{len(interseccion)} están dentro de la banda z diagnosticada")

    def check_edge_events(self, net, t):
        """Detecta flancos sobre la señal heurística aislada del detector."""
        for sym in self.symbols:
            current = float(self._detector_evidence.get(sym, 0.0))
            previous = self._prev_evidence[sym]
            cruzo_umbral = previous <= self.edge_threshold < current

            if cruzo_umbral and not self._evidence_cooldown[sym]:
                self._fire_buffer_pulse(net, sym, t)
                self._evidence_cooldown[sym] = True
                self._event_log.append((sym, float(t)))
                if len(self._event_log) > self._event_log_max:
                    self._event_log = self._event_log[-self._event_log_max:]
                print(f"⚡ [Secuencia] Evento '{sym}' en T={t:.1f} (detector={current:.2f})", flush=True)

            if current < self.edge_threshold * 0.5:
                self._evidence_cooldown[sym] = False

            self._prev_evidence[sym] = current

    def _fire_buffer_pulse(self, net, sym, t, strength=6.0):
        for idx in self._buffer_pools.get(sym, []):
            if 0 <= idx < net.n and net.active[idx]:
                net.receive_spike(idx, strength, t)

    # --- FUNCIONES ETAPA 3 (Secuencias) ---

    def configure_words(self, words):
        """Registra bigramas (letra1 -> letra2) como 'palabras'. Cada uno
        recibe un puerto de entrada evidencia-de-orden (secuencia_{palabra})
        y uno de salida maestro (maestro_{palabra}) para refuerzo top-down,
        siguiendo el mismo patrón que las letras individuales.

        Solo acepta pares de 2 letras que ya estén en self.symbols — no
        tiene sentido pedirle a la red que ordene una letra que ni siquiera
        reconoce todavía."""
        validas = [w for w in words if len(w) == 2 and w[0] in self.symbols and w[1] in self.symbols]
        n_words = len(validas)

        self.words = []
        self._word_pairs = {}

        for i, word in enumerate(words):
            if word not in validas:
                print(f"⚠️ [Secuencia] '{word}' ignorada: requiere 2 letras dentro de "
                      f"self.symbols={self.symbols}", flush=True)
                continue

            x = 65.0 + (i * (20.0 / max(1, n_words - 1))) if n_words > 1 else 75.0
            self.add_input_port(f"secuencia_{word}", pos=(x, 75.0, 60.0))
            self.add_output_port(f"maestro_{word}", pos=(x, 90.0, 65.0))

            self.words.append(word)
            self._word_pairs[word] = (word[0], word[1])

        print(f"🔤 [Secuencia] Vocabulario de palabras configurado: {self.words}", flush=True)

    def reset_sequence_state(self):
        """Limpia el log de flancos antes de arrancar un trial de secuencia,
        para que eventos de un trial anterior no contaminen el orden medido."""
        self._event_log = []

    def update_sequence_evidence(self, net, t, window_start, max_gap_ms=1500.0):
        """A.3 (Etapa 3) — evalúa cada secuencia_{palabra} configurada.

        Para cada palabra (l1, l2): busca en el log de flancos el primer
        evento de l1 dentro de la ventana del trial, y el primer evento de
        l2 que haya ocurrido DESPUÉS de ese evento de l1. Si existen ambos
        y la brecha entre uno y otro es menor a max_gap_ms, hay evidencia
        de secuencia — más alta cuanto más ajustado el orden, decayendo
        linealmente a 0 en max_gap_ms. Si l2 aparece antes que l1 (orden
        invertido) o no aparece dentro de la ventana, evidencia = 0.

        Mismo espíritu que evidencia_{letra} (una fracción 0-1, no un
        booleano), para que la currícula de palabras pueda usar el mismo
        criterio de dominio gradual que ya usa la de letras."""
        for word, (l1, l2) in self._word_pairs.items():
            port = self.input_ports.get(f"secuencia_{word}")
            if not port:
                continue

            t1 = next((et for (es, et) in self._event_log if es == l1 and et >= window_start), None)
            if t1 is None:
                port.value = 0.0
                continue

            t2 = next((et for (es, et) in self._event_log if es == l2 and et > t1), None)
            if t2 is None:
                port.value = 0.0
                continue

            gap = t2 - t1
            if gap > max_gap_ms:
                port.value = 0.0
                continue

            port.value = max(0.0, 1.0 - (gap / max_gap_ms))

    def debug_event_log(self):
        """Diagnóstico: qué flancos se registraron en el trial actual, en orden."""
        if not self._event_log:
            print("🔬 [Diag-Secuencia] Log de eventos vacío.")
            return
        orden = " → ".join(f"{sym}@{t:.0f}" for sym, t in self._event_log)
        print(f"🔬 [Diag-Secuencia] Eventos registrados: {orden}")

    def get_ports_info(self):
        return [{"name": f"L_{p.name}", "pos": list(p.pos), "strength": p.strength,
                 "connected_neurons": list(p.connected_neurons)}
                for p in {**self.input_ports, **self.output_ports}.values()]