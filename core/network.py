# core/network.py
from core.evolution_config import EVO_MENU, DEFAULT_PROFILE

import random
import math
import numpy as np
import threading
import heapq
import logging

from config import (
    GRID_SIZE, GRID_DEPTH, RADIUS_3D, NUM_REGIONS,
    V_REST, V_THRESHOLD, V_RESET, TAU_MEMBRANE, REFRACTORY_PERIOD,
    SYNAPTIC_DELAY_BASE, IGNITION_BOOST_FACTOR, MAX_EVENTS_QUEUE,
    CAUSAL_EVENT_UTILITY_MIN, CAUSAL_EVENT_PREFILTER_RATIO,
    CAUSAL_EVENT_RECENCY_TAU_MS, CAUSAL_EVENT_AGGREGATION_ENABLED,
    CAUSAL_EVENT_AGGREGATION_BUCKET_MS, CAUSAL_EVENT_AGGREGATION_MAX_STRENGTH
)

# Fase 3: límite de seguridad para la cola de eventos.
# El valor normal viene de config.py; este fallback evita que un entorno
# incompleto rompa la red al importar el módulo.
MAX_EVENTS_QUEUE = int(MAX_EVENTS_QUEUE) if MAX_EVENTS_QUEUE else 20000

from systems.genetics import GeneticSystem
from core.region import Region
from core.reward_system import RewardSystem
from modules.heart import HeartModule
from modules.vision import VisionModule
from core.temporal import TemporalCore


class NeuralNetwork:
    
    def __init__(self, n=8000, skip_wiring=False, max_neurons=10000):
        self.max_neurons = max_neurons
        self.n = n
        self.lock = threading.RLock()

        # === ESTADOS VECTORIZADOS ===
        self.active = np.zeros(self.max_neurons, dtype=bool)
        self.active[:n] = True

        self.membrane_potential = np.full(self.max_neurons, V_REST, dtype=np.float32)
        self.last_update_time = np.zeros(self.max_neurons, dtype=np.float32)
        self.refractory_until = np.zeros(self.max_neurons, dtype=np.float32)
        self.energy = np.ones(self.max_neurons, dtype=np.float32)

        # === FITNESS NEURONAL (Condición física por neurona) ===
        # Rango: 0.5 (atrofiada por desuso) a 2.0 (entrenada)
        # Sube cuando la neurona dispara con recompensa dopaminérgica
        # Baja cuando la neurona está inactiva demasiado tiempo
        # Determina el techo máximo de energía alcanzable
        self.fitness_neuronal = np.ones(self.max_neurons, dtype=np.float32)

        self.fired = np.zeros(self.max_neurons, dtype=np.int32)
        self.fired_acumulado = np.zeros(self.max_neurons, dtype=np.int32)
        self.last_spike = np.zeros(self.max_neurons, dtype=np.float32)

        # Compatibilidad con sistemas heredados que todavía esperan estos
        # vectores. Son aliases/estado ligero y no duplican el potencial: el
        # campo canónico sigue siendo membrane_potential.
        self.potential = self.membrane_potential
        self.excitability = np.ones(self.max_neurons, dtype=np.float32)
        self.gaba_sensitivity = np.ones(self.max_neurons, dtype=np.float32)
        self.calcium = np.zeros(self.max_neurons, dtype=np.float32)
        self.global_calcium = 0.0

        # Parámetros biológicos
        self.tau_m = np.full(self.max_neurons, TAU_MEMBRANE, dtype=np.float32)
        self.v_thresh = np.full(self.max_neurons, V_THRESHOLD, dtype=np.float32)
        self.refr_period = np.full(self.max_neurons, REFRACTORY_PERIOD, dtype=np.float32)

        # === QUÍMICA ===
        
        self.nt_vector = np.zeros((self.max_neurons, 5), dtype=np.float32)
        self.receptors = self.nt_vector.copy()          # ← Corrección clave
        
        # Agregar el nivel de dopamina global de la red.
        # Fase 3: valor basal no nulo para evitar que SlowBio nazca en "coma"
        # durante el primer tick.
        self.dopamine_level = 0.5

        # Máscara de marcapasos. Se reconstruye tras conocer las posiciones
        # físicas y también se vuelve a verificar durante load_state().
        self.heart_mask = np.zeros(self.max_neurons, dtype=bool)

        # Posiciones
        self.positions = np.zeros((self.max_neurons, 3), dtype=np.float32)
        for i in range(n):
            self.positions[i] = [random.uniform(0, 150), random.uniform(0, 150), random.uniform(0, 100)]

        # Marcapasos aproximado por franja física Z=37..53.
        self.heart_mask[:n] = (
            (self.positions[:n, 2] >= 37.0) &
            (self.positions[:n, 2] <= 53.0)
        )

        self.connections = [[] for _ in range(self.max_neurons)]
        self.dna = [None] * self.max_neurons
        for i in range(n):
            self.dna[i] = self._generate_initial_dna()

        # Regiones
        self.regions = {i: Region(i) for i in range(NUM_REGIONS)}
        self.neuron_region = np.full(self.max_neurons, -1, dtype=np.int32)
        self.neuron_subregion = np.full(self.max_neurons, -1, dtype=np.int32)

        initial_regions = [random.randint(0, NUM_REGIONS - 1) for _ in range(n)]
        self.neuron_region[:n] = initial_regions

        # Sub-regiones iniciales
        self.regions[0].create_sub_region(sub_id=0, profile=EVO_MENU["RAPID_FIRE"])
        self.regions[0].create_sub_region(sub_id=1, profile=EVO_MENU["BUFFER"])

        # Módulos
        self.genetics = GeneticSystem()
        self.heart = HeartModule()
        self.vision = VisionModule(net=self)
        self.reward_manager = RewardSystem()
        
        
        self.modules = {
            "heart": self.heart,
            "vision": self.vision,
            "reward_manager": self.reward_manager
        }

        # Eventos
        self.event_queue = []
        self.weights = {}
        self.current_time = 0.0
        self.temporal_core = TemporalCore()
        self._processing_event_origin = None
        self._causal_frame_fanout_attempts = 0
        self._causal_frame_fanout_limited = 0
        self._causal_frame_index = -1
        self._causal_frame_budget = 0
        self.ignition_active = False
        # IA 1.6.25: agregación temporal segura de propagación sin readout.
        # La entrada mantiene un representante en el heap y acumula fuerza
        # equivalente dentro de una ventana sub-milisegundo.
        self._causal_aggregate_pending = {}

        # Telemetría de presión de eventos (v0.8).
        # Neuronas exentas del governor de propagación (readout sources, etc.)
        # Se registran externamente (language_io, etc.) después de inicializar.
        self.governor_exempt_targets: set = set()

        self.event_queue_stats = {
            "enqueued_total": 0,
            "governor_exempt_passes": 0,
            "processed_total": 0,
            "dropped_total": 0,
            "dropped_by_hard_cap": 0,
            "dropped_by_governor": 0,
            "propagation_attempts": 0,
            "propagation_enqueued": 0,
            "propagation_governor_drops": 0,
            "fanout_limited_total": 0,
            "sensory_enqueued": 0,
            "electroshock_enqueued": 0,
            "max_queue_size": 0,
            "last_queue_size": 0,
            "ready_backlog_peak": 0,
            "production_minus_consumption": 0,
            "governor_drop_rate": 0.0,
            "ready_events_coalesced": 0,
            "backlog_relief_passes": 0,
            "shed_synaptic_total": 0,
            "backpressure_timeouts": 0,
            "sensory_cap_drops": 0,
            "fanout_attempted_total": 0,
            "fanout_sent_total": 0,
            "accepted_minus_processed": 0,
            "attempt_minus_processed": 0,
            "readout_governor_drops": 0,
            "feature_bus_enqueued": 0,
            "feature_bus_processed": 0,
            "feature_bus_pending": 0,
            "feature_bus_cap_drops": 0,
            "causal_event_seq": 0,
            "causal_budget_attempts": 0,
            "causal_budget_limited": 0,
            "causal_frame_index": -1,
            "causal_prefilter_drops": 0,
            "causal_utility_selected": 0,
            "causal_utility_considered": 0,
            "causal_governor_bypass": 0,
            "causal_events_aggregated": 0,
            "causal_aggregation_strength_added": 0.0,
        }
        # Rutas sensoriales estructurales: se construyen una sola vez sobre
        # neuronas existentes con sinapsis reales. No se usan radios por frame.
        self.sensory_relay_routes = {}
        self._sensory_routes_ready = False
        # Mapeo opcional target->símbolo usado por el fan-out competitivo del readout.
        # Lo registra modules.language_io; permanece vacío para el resto de la red.
        self.readout_target_to_symbol = {}
        self.readout_allowed_sources = set()

        # v1.4: roles funcionales. La competencia biológica sigue activa para
        # la red general, pero los circuitos de infraestructura cognitiva
        # (lenguaje/readout/buffers) no deben perder su función por extinción,
        # homeostasis o STDP genérico.
        self.neuron_roles = {}
        self.noncompetitive_neurons = set()

        # v1.4: canal de eventos dedicado para readout. Evita que la poda de
        # propagación general haga competir una ruta crítica de clasificación
        # contra el ruido/cascada de la red biológica.
        self.readout_event_queue = []
        self.readout_queue_max = 6000
        try:
            from config import READOUT_SYNAPTIC_GAIN
        except Exception:
            READOUT_SYNAPTIC_GAIN = 1.10
        self.readout_synaptic_gain = float(READOUT_SYNAPTIC_GAIN)
        try:
            from config import READOUT_ROUTING_TEMPERATURE, READOUT_ROUTING_MIN_BUDGET, READOUT_ROUTING_MAX_BUDGET
        except Exception:
            READOUT_ROUTING_TEMPERATURE = 0.35
            READOUT_ROUTING_MIN_BUDGET = 0.25
            READOUT_ROUTING_MAX_BUDGET = 0.80
        self.readout_routing_temperature = float(READOUT_ROUTING_TEMPERATURE)
        self.readout_routing_min_budget = float(READOUT_ROUTING_MIN_BUDGET)
        self.readout_routing_max_budget = float(READOUT_ROUTING_MAX_BUDGET)
        self.event_queue_stats.update({
            "readout_enqueued": 0,
            "readout_processed": 0,
            "readout_queue_max": 0,
            "readout_queue_size": 0,
            "readout_ready": 0,
            "readout_cap_drops": 0,
            "readout_priority_passes": 0,
            "readout_foreign_source_blocked": 0,
            "readout_competition_events": 0,
            "readout_intrapool_fatigue_events": 0,
            "noncompetitive_registered": 0,
            "queue_pressure_events": 0,
            "general_synaptic_shed": 0,
            "sensory_preserved_total": 0,
        })


        # Inicialización
        self._apply_initial_genetics()
        self._assign_initial_subregions()

        # --- FIX: cableado inicial es O(n²) (_init_connections_sparse) y
        # su resultado (net.connections, net.weights, tau_m/v_thresh/refr_period
        # de _init_core_connections) queda 100% pisado por load_state() cuando
        # se está restaurando un guardado — son ~40 minutos tirados a la basura
        # en una red de 50k. skip_wiring=True lo salta para ese caso; una red
        # nueva (sin guardado) sigue cableándose como siempre.
        if not skip_wiring:
            self._init_core_connections()
            self._init_connections_sparse()
        else:
            print("⚡ [Optimización] skip_wiring=True: se omite el cableado inicial "
                  "O(n²) porque va a ser reemplazado por load_state().")

        print(f"🚀 NeuralNetwork 1.2.2 | {self.n} neuronas | Ciclo Visión OK")
        
        
    def _apply_phase3_survival_guards(self):
        """Aplica los límites de supervivencia introducidos en Fase 3.

        Se ejecuta después de restaurar el estado persistido y antes de reanudar
        la dinámica de la simulación.
        """
        if not hasattr(self, 'fitness_neuronal'):
            return

        fitness_minimo_viable = 0.35
        fitness_minimo_corazon = 0.65

        active = getattr(self, 'active', None)
        if active is None:
            active = np.ones(self.max_neurons, dtype=bool)

        activos = np.asarray(active, dtype=bool)[:self.n]
        if not np.any(activos):
            return

        fitness = self.fitness_neuronal[:self.n]

        # Rescate global: evita restaurar una red en un estado metabólico
        # donde todo el fitness ya esté por debajo del umbral operativo.
        if float(np.mean(fitness[activos])) < fitness_minimo_viable:
            fitness[activos] = np.maximum(fitness[activos], fitness_minimo_viable)

        # Protección del marcapasos: garantiza un nivel mínimo de fitness en
        # la franja que usamos como zona de corazón.
        if hasattr(self, 'heart_mask') and self.heart_mask is not None:
            heart = self.heart_mask[:self.n] & activos
            if np.any(heart):
                fitness[heart] = np.maximum(fitness[heart], fitness_minimo_corazon)

        self.fitness_neuronal[:self.n] = fitness

    def get_port(self, port_name):
     """Busca un puerto por nombre dentro de todos los módulos registrados."""
     if hasattr(self, 'modules'):
        for module in self.modules.values():
            # Si tus módulos guardan los puertos en un diccionario interno o lista
            if hasattr(module, 'ports') and port_name in module.ports:
                return module.ports[port_name]
            # O si los tenés como atributos directos (como self.out_x)
            for attr_name, attr_val in module.__dict__.items():
                if attr_name.startswith("out_") and hasattr(attr_val, "name") and attr_val.name == port_name:
                    return attr_val
     return None
        

    def _generate_initial_dna(self):
        return [random.uniform(0, 150), random.uniform(0, 150), random.uniform(0, 100),
                random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0.1, 1.0)]

    def _apply_initial_genetics(self):
        for i in range(self.n):
            self.genetics.mutate_new_neuron(self, i)

    def _assign_initial_subregions(self):
        for i in range(self.n):
            r_id = self.neuron_region[i]
            region = self.regions[r_id]
            if region.sub_regions:
                s_id = random.choice(list(region.sub_regions.keys()))
                self.neuron_subregion[i] = s_id
                region.sub_regions[s_id].add_neuron(i)
                prof = region.sub_regions[s_id].current_profile
                self.tau_m[i] = prof.tau_m
                self.v_thresh[i] = prof.v_thresh
                self.refr_period[i] = prof.refr_period

    import math

    def _init_core_connections(self):
        """Prepara los puertos motores sin usar una zona espacial artificial.

        Las neuronas elegidas pertenecen a la red normal; el cableado
        sináptico se realiza después por ``_init_connections_sparse``.
        """
        rf_profile = EVO_MENU["RAPID_FIRE"]
        candidatos = [
            i for i in range(self.n)
            if self.active[i] and not self.heart_mask[i]
        ]
        if not candidatos:
            print("⚠️ [Visión] No hay neuronas disponibles para puertos motores.")
            return

        # Selección estable y distribuida por índice; no depende de distancia.
        total = min(len(candidatos), 36)
        seleccion = candidatos[:total]
        grupos = np.array_split(np.asarray(seleccion, dtype=np.int32), 3)

        groups = grupos
        puertos = (
            (self.vision.coord_x_port, groups[0]),
            (self.vision.coord_y_port, groups[1]),
            (self.vision.fovea_radius_port, groups[2]),
        )
        for port, grupo in puertos:
            for idx in grupo.tolist():
                port.connect(int(idx))
                self.tau_m[idx] = rf_profile.tau_m
                self.v_thresh[idx] = rf_profile.v_thresh
                self.refr_period[idx] = rf_profile.refr_period

        print(
            "🔌 [Visión] Puertos motores conectados estructuralmente → "
            f"X:{len(grupos[0])} | Y:{len(grupos[1])} | Fóvea:{len(grupos[2])}"
        )

    def backfill_orphan_connections(self, max_new_per_neuron=8, min_connections=1):
        """
        Cablea retroactivamente las neuronas activas con MENOS de
        min_connections conexiones. Con min_connections=1 se comporta
        como antes (solo huérfanas totales); con un valor mayor también
        repara a las que solo tenían el vínculo heredado del padre
        (systems/growth.py, antes del fix de vecindario real).
        """
        pocas = [i for i in range(self.n) if self.active[i] and len(self.connections[i]) < min_connections]
        if not pocas:
            print(f"🔌 [Backfill] Ninguna neurona por debajo de {min_connections} conexiones — nada que hacer.")
            return

        conectadas = 0
        for i in pocas:
            r_id = self.neuron_region[i]
            s_id = self.neuron_subregion[i]
            
            if r_id in self.regions and s_id in self.regions[r_id].sub_regions:
                radius = self.regions[r_id].sub_regions[s_id].current_profile.conn_radius
            else:
                radius = RADIUS_3D
    
            pos_i = self.positions[i]
            dist = np.linalg.norm(self.positions[:self.n] - pos_i, axis=1)
            candidatos = np.argsort(dist)
            
            agregadas = 0
            for j in candidatos:
                j = int(j)
                if j == i or not self.active[j] or j in self.connections[i]:
                    continue
                if dist[j] > radius:
                    break
                self._connect(i, j, random.uniform(0.3, 0.7))
                agregadas += 1
                if agregadas >= max_new_per_neuron:
                    break
                
            if agregadas > 0:
                conectadas += 1
            
        print(f"🔌 [Backfill] {conectadas}/{len(pocas)} neuronas recibieron conexiones nuevas.")

    @staticmethod
    def _causal_aggregation_key(t_event, target_idx, origin):
        bucket = max(1e-6, float(CAUSAL_EVENT_AGGREGATION_BUCKET_MS))
        slot = int(round(float(t_event) / bucket))
        return (int(target_idx), str(origin), slot)

    def consume_causal_aggregated_event(self, event):
        """Materializa la fuerza agregada al sacar un evento del heap."""
        if not event or len(event) < 5:
            return event
        try:
            t_ev, seq, idx, strength, origin = event[:5]
            key = self._causal_aggregation_key(t_ev, idx, origin)
        except Exception:
            return event
        with self.lock:
            extra = self._causal_aggregate_pending.pop(key, 0.0)
        if extra <= 0.0:
            return event
        return (float(t_ev), int(seq), int(idx), float(min(CAUSAL_EVENT_AGGREGATION_MAX_STRENGTH, float(strength) + extra)), origin)

    def _causal_candidate_utility(self, target, strength, t_now):
        """Estimación física conservadora de utilidad de una propagación."""
        try:
            target = int(target)
            strength = max(0.0, float(strength))
            if not (0 <= target < self.n):
                return 0.0
            v = float(self.membrane_potential[target])
            th = max(1e-6, float(self.v_thresh[target]))
            headroom = max(1e-6, th - v)
            ratio = strength / headroom
            ratio_score = min(3.0, ratio) / 3.0
            last = float(self.last_spike[target])
            age = max(0.0, float(t_now) - last) if last > 0.0 else 1e6
            freshness = 1.0 - math.exp(-age / max(1e-6, float(CAUSAL_EVENT_RECENCY_TAU_MS)))
            return float(0.72 * ratio_score + 0.28 * freshness)
        except Exception:
            return 0.0

    def _causal_physical_prefilter(self, target, strength, t_now):
        """Descarta sólo entradas claramente incapaces de mover el target."""
        try:
            target = int(target); strength = max(0.0, float(strength))
            if not (0 <= target < self.n):
                return True
            v = float(self.membrane_potential[target])
            th = max(1e-6, float(self.v_thresh[target]))
            headroom = max(1e-6, th - v)
            last = float(self.last_spike[target])
            age = max(0.0, float(t_now) - last) if last > 0.0 else 1e6
            ratio = strength / headroom
            return bool(ratio < float(CAUSAL_EVENT_PREFILTER_RATIO) and age > float(CAUSAL_EVENT_RECENCY_TAU_MS))
        except Exception:
            return False

    def begin_causal_trial(self, trial_id, start_time_ms):
        with self.lock:
            self.temporal_core.begin_trial(trial_id, float(start_time_ms))
            self._causal_frame_fanout_attempts = 0
            self._causal_frame_fanout_limited = 0
            self._causal_frame_index = -1
            self._causal_trial_active = True

    def begin_causal_frame(self, frame_index, watermark_ms, budget):
        with self.lock:
            self.temporal_core.begin_frame(int(frame_index), float(watermark_ms))
            self._causal_frame_fanout_attempts = 0
            self._causal_frame_fanout_limited = 0
            self._causal_frame_index = int(frame_index)
            self._causal_frame_budget = int(budget)
            self.event_queue_stats["causal_frame_index"] = int(frame_index)

    def end_causal_trial(self):
        with self.lock:
            self.temporal_core.end_trial()
            self._causal_trial_active = False
            self._causal_frame_index = -1
            self._causal_frame_budget = 0

    def causal_temporal_snapshot(self):
        with self.lock:
            snap = self.temporal_core.snapshot()
            snap.update({
                "frame_fanout_attempts": int(self._causal_frame_fanout_attempts),
                "frame_fanout_limited": int(self._causal_frame_fanout_limited),
                "frame_budget": int(self._causal_frame_budget),
            })
            return snap

    def _next_event_seq(self, t_event, origin=None):
        seq = self.temporal_core.next_event(float(t_event), origin=origin)
        self.event_queue_stats["causal_event_seq"] = int(seq)
        return int(seq)

    def register_noncompetitive_neurons(self, indices, role="infrastructure"):
        """Marca neuronas funcionales que no deben participar en competencia destructiva.

        La competencia se conserva para neuronas biológicas libres; esta lista
        protege únicamente circuitos que representan una función explícita
        (lenguaje, buffers, readout, etc.).
        """
        added = 0
        for idx in indices or []:
            try:
                idx = int(idx)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < self.n:
                self.noncompetitive_neurons.add(idx)
                self.neuron_roles[idx] = str(role)
                added += 1
        self.event_queue_stats["noncompetitive_registered"] = len(self.noncompetitive_neurons)
        return added

    def is_noncompetitive(self, idx):
        try:
            return int(idx) in self.noncompetitive_neurons
        except Exception:
            return False

    def is_readout_target(self, idx):
        try:
            return int(idx) in (getattr(self, "readout_target_to_symbol", {}) or {})
        except Exception:
            return False

    def _enqueue_readout_event(self, target_idx, strength, arrival_time, origin):
        """Canal dedicado y acotado para source -> pool del readout."""
        stats = self.event_queue_stats
        if len(self.readout_event_queue) >= int(self.readout_queue_max):
            stats["dropped_total"] += 1
            stats["readout_cap_drops"] += 1
            return False
        seq = self._next_event_seq(arrival_time, origin)
        heapq.heappush(self.readout_event_queue, (float(arrival_time), int(seq), int(target_idx), float(strength), origin))
        stats["enqueued_total"] += 1
        stats["readout_enqueued"] += 1
        stats["readout_queue_max"] = max(stats.get("readout_queue_max", 0), len(self.readout_event_queue))
        return True

    def receive_spike(self, target_idx, strength, arrival_time, origin=None):
        with self.lock:
            if not (0 <= target_idx < self.n) or not self.active[target_idx]:
                return False

            stats = self.event_queue_stats
            qsize = len(self.event_queue)

            hard_cap = max(20000, int(MAX_EVENTS_QUEUE))
            if qsize >= 16000:
                stats["queue_pressure_events"] += 1
            # v1.4: los targets del readout entran en un canal dedicado.
            # No pasan por el governor de propagación general porque ese
            # governor existe para proteger la dinámica emergente, no para
            # destruir una función explícita de clasificación.
            if origin == "synaptic" and self.is_readout_target(target_idx):
                return self._enqueue_readout_event(target_idx, strength, arrival_time, origin)
            if origin == "synaptic":
                # IA 1.6.25: dentro del trial causal la economía de eventos se
                # resuelve antes (utility budget + prefilter). No se aplica el
                # governor por fuerza que alteraba la dinámica al 48.9% de drops.
                if bool(getattr(self, "_causal_trial_active", False)):
                    stats["causal_governor_bypass"] = int(stats.get("causal_governor_bypass", 0)) + 1
                # FIX 1.5.7: neuronas readout-source siempre pasan, independientemente de la queue.
                # Son solo 120 de 10000 — el costo de enqueue es mínimo vs el beneficio.
                target_is_readout = int(target_idx) in (getattr(self, "readout_target_to_symbol", {}) or {})
                if int(target_idx) not in self.governor_exempt_targets and not bool(getattr(self, "_causal_trial_active", False)):
                    # Fuera del trial causal se conserva el governor histórico.
                    thresholds = (
                        ((20000, 0.30), (16000, 0.22), (12000, 0.18), (8000, 0.12))
                        if target_is_readout else
                        ((20000, 1.75), (16000, 1.45), (12000, 1.05), (8000, 0.65))
                    )
                    for q_limit, min_strength in thresholds:
                        if qsize >= q_limit and float(strength) < min_strength:
                            stats["dropped_total"] += 1
                            stats["dropped_by_governor"] += 1
                            stats["propagation_governor_drops"] += 1
                            if target_is_readout:
                                stats["readout_governor_drops"] += 1
                            return False
                else:
                    # Compatibilidad con configuraciones antiguas; v1.3 no registra
                    # targets de readout como exentos, por lo que este camino debe ser 0.
                    stats["governor_exempt_passes"] += 1

            # Presupuesto separado: la entrada sensorial no debe ser eliminada
            # por el mismo corte que usa la propagación interna. Esto es crítico
            # durante un trial, donde perder el estímulo invalida toda la medición.
            sensory_cap = hard_cap + 1500
            if qsize >= hard_cap:
                if origin in ("sensory", "electroshock", "feature_bus"):
                    if qsize >= sensory_cap:
                        stats["dropped_total"] += 1
                        stats["sensory_cap_drops"] += 1
                        return False
                elif float(strength) < 4.0 or qsize >= hard_cap + 500:
                    stats["dropped_total"] += 1
                    stats["dropped_by_hard_cap"] += 1
                    stats["general_synaptic_shed"] += 1
                    return False

            # IA 1.6.25: agrupación segura sólo para sinapsis normales durante el
            # trial causal. Readout, sensorial y electroshock jamás se agregan.
            if (bool(getattr(self, "_causal_trial_active", False)) and
                    origin == "synaptic" and
                    bool(CAUSAL_EVENT_AGGREGATION_ENABLED) and
                    not self.is_readout_target(target_idx)):
                key = self._causal_aggregation_key(arrival_time, target_idx, origin)
                current = self._causal_aggregate_pending.get(key)
                if current is not None:
                    remaining = max(0.0, float(CAUSAL_EVENT_AGGREGATION_MAX_STRENGTH) - float(current))
                    added = min(remaining, float(strength))
                    if added > 0.0:
                        self._causal_aggregate_pending[key] = float(current) + added
                        stats["causal_events_aggregated"] = int(stats.get("causal_events_aggregated", 0)) + 1
                        stats["causal_aggregation_strength_added"] = float(stats.get("causal_aggregation_strength_added", 0.0)) + added
                        return True

            # Guardamos el origen dentro del evento para poder hacer backpressure
            # selectivo sin destruir entradas sensoriales.
            seq = self._next_event_seq(arrival_time, origin)
            ev_tuple = (float(arrival_time), int(seq), int(target_idx), float(strength), origin)
            heapq.heappush(self.event_queue, ev_tuple)
            if (bool(getattr(self, "_causal_trial_active", False)) and
                    origin == "synaptic" and bool(CAUSAL_EVENT_AGGREGATION_ENABLED) and
                    not self.is_readout_target(target_idx)):
                key = self._causal_aggregation_key(arrival_time, target_idx, origin)
                self._causal_aggregate_pending.setdefault(key, 0.0)
            stats["enqueued_total"] += 1
            if origin == "synaptic":
                stats["propagation_enqueued"] += 1
            elif origin == "sensory":
                stats["sensory_enqueued"] += 1
            elif origin == "electroshock":
                stats["electroshock_enqueued"] += 1
            elif origin == "feature_bus":
                stats["feature_bus_enqueued"] += 1
                stats["feature_bus_pending"] = int(stats.get("feature_bus_pending", 0)) + 1
            stats["max_queue_size"] = max(stats["max_queue_size"], len(self.event_queue))
            stats["last_queue_size"] = len(self.event_queue)
            if origin == "sensory":
                stats["sensory_preserved_total"] += 1
            return True

    def clear_readout_runtime_events(self):
        """Borra eventos readout pendientes para mantener cada trial aislado."""
        with self.lock:
            removed = len(getattr(self, "readout_event_queue", []))
            self.readout_event_queue = []
            heapq.heapify(self.readout_event_queue)
            self.event_queue_stats["readout_queue_size"] = 0
        return removed

    def coalesce_ready_event_backlog(self, max_events=12000):
        """Reduce duplicados en eventos ya listos bajo congestión severa.

        Solo se aplica cuando la cola está muy alta y todos los eventos inspeccionados
        ya son procesables. Para cada destino conserva un evento con la mayor fuerza;
        esto evita que una cascada de eventos equivalentes mantenga la cola saturada.
        La operación se cuenta explícitamente en telemetría.
        """
        with self.lock:
            if not self.event_queue:
                return 0
            now = float(self.current_time)
            ready = []
            future = []
            # La cola es un heap; conservar el orden de eventos futuros es importante.
            for ev in self.event_queue:
                try:
                    (future if float(ev[0]) > now else ready).append(ev)
                except Exception:
                    future.append(ev)

            if len(ready) <= int(max_events):
                return 0

            best = {}
            passthrough = []
            for ev in ready:
                try:
                    if len(ev) >= 5:
                        t_ev, _, idx, strength, origin = ev[:5]
                    else:
                        t_ev, idx, strength, origin = ev
                except ValueError:
                    t_ev, idx, strength = ev[:3]
                    origin = ev[3] if len(ev) >= 4 else None
                # Nunca colapsar entradas sensoriales/electroshock; son eventos
                # de entrada y no cascadas internas redundantes.
                if origin not in ("synaptic", None):
                    seq = self._next_event_seq(t_ev, origin)
                    passthrough.append((float(t_ev), int(seq), int(idx), float(strength), origin))
                    continue
                key = int(idx)
                prev = best.get(key)
                if prev is None or float(strength) > float(prev[3] if len(prev) >= 5 else prev[2]):
                    seq = self._next_event_seq(t_ev, origin)
                    best[key] = (float(t_ev), int(seq), key, float(strength), origin)

            new_ready = list(best.values()) + passthrough
            before = len(ready)
            removed = before - len(new_ready)
            if removed <= 0:
                return 0

            self.event_queue = new_ready + future
            heapq.heapify(self.event_queue)
            self.event_queue_stats["ready_events_coalesced"] += int(removed)
            self.event_queue_stats["backlog_relief_passes"] += 1
            self.event_queue_stats["last_queue_size"] = len(self.event_queue)
            return int(removed)

    def shed_weak_ready_synaptic_events(self, max_remove=4000, strength_threshold=0.35):
        """Libera presión sin tocar eventos sensoriales ni neuromoduladores."""
        with self.lock:
            if not self.event_queue:
                return 0
            now = float(self.current_time)
            kept = []
            removed = 0
            limit = int(max_remove)
            for ev in self.event_queue:
                try:
                    if len(ev) >= 5:
                        t_ev, _, idx, strength, origin = ev[:5]
                    else:
                        t_ev, idx, strength, origin = ev
                except ValueError:
                    t_ev, idx, strength = ev[:3]
                    origin = ev[3] if len(ev) >= 4 else None
                if (removed < limit and float(t_ev) <= now and
                        origin == "synaptic" and float(strength) < float(strength_threshold)):
                    removed += 1
                    continue
                if len(ev) >= 5:
                    kept.append(tuple(ev[:5]))
                else:
                    seq = self._next_event_seq(t_ev, origin)
                    kept.append((float(t_ev), int(seq), int(idx), float(strength), origin))
            if removed:
                self.event_queue = kept
                heapq.heapify(self.event_queue)
                self.event_queue_stats["shed_synaptic_total"] += int(removed)
                self.event_queue_stats["last_queue_size"] = len(self.event_queue)
            return int(removed)

    def get_event_queue_stats(self):
        with self.lock:
            now = float(self.current_time)
            ready = 0
            for ev in self.event_queue:
                try:
                    if float(ev[0]) <= now:
                        ready += 1
                except Exception:
                    continue
            stats = dict(self.event_queue_stats)
            queue_size = len(self.event_queue)
            stats.update({
                "queue_size": queue_size,
                "ready_count": ready,
                "future_count": max(0, queue_size - ready),
                "current_time": now,
            })
            stats["ready_backlog_peak"] = max(int(stats.get("ready_backlog_peak", 0)), int(ready))
            accepted = int(stats.get("enqueued_total", 0))
            processed = int(stats.get("processed_total", 0))
            attempts = int(stats.get("propagation_attempts", 0))
            drops = int(stats.get("propagation_governor_drops", 0))
            fanout_attempted = int(stats.get("fanout_attempted_total", 0))
            fanout_sent = int(stats.get("fanout_sent_total", 0))
            stats["accepted_minus_processed"] = accepted - processed
            stats["attempt_minus_processed"] = attempts - processed
            # Compatibilidad: este campo ahora significa accepted - processed.
            stats["production_minus_consumption"] = accepted - processed
            stats["fanout_attempted_total"] = fanout_attempted
            stats["fanout_sent_total"] = fanout_sent
            stats["fanout_send_rate"] = (fanout_sent / fanout_attempted) if fanout_attempted else 0.0
            stats["governor_drop_rate"] = (drops / attempts) if attempts else 0.0
            stats["temporal_core"] = self.temporal_core.snapshot()
            stats["causal_frame_fanout_attempts"] = int(getattr(self, "_causal_frame_fanout_attempts", 0))
            stats["causal_frame_fanout_limited"] = int(getattr(self, "_causal_frame_fanout_limited", 0))
            stats["causal_frame_budget"] = int(getattr(self, "_causal_frame_budget", 0))
            return stats

    def _build_sensory_relay_routes(self, channels=3, bins_x=8, bins_y=8, neurons_per_route=1):
        """Construye una topología sensorial fija sin búsqueda por radio.

        La selección se hace una sola vez usando neuronas activas que ya
        poseen conexiones sinápticas normales. Cada celda visual apunta a
        uno o varios relés fijos; después la actividad entra en esos relés y
        se propaga exclusivamente mediante ``connections``.
        """
        with self.lock:
            candidates = [
                i for i in range(self.n)
                if self.active[i] and not self.heart_mask[i] and len(self.connections[i]) > 0
            ]

        required = channels * bins_x * bins_y * neurons_per_route
        if len(candidates) < required:
            raise RuntimeError(
                f"No hay suficientes neuronas con sinapsis para rutas visuales: "
                f"{len(candidates)} < {required}"
            )

        # Selección espacial: cada bin busca las neuronas más cercanas a su centro
        # XY y, opcionalmente, restringe el canal por banda Z. Esto preserva
        # correspondencia espacial en lugar de asignar IDs arbitrarios.
        pos = np.asarray(self.positions[candidates], dtype=float)
        candidates_arr = np.asarray(candidates, dtype=int)
        routes = {}
        x_max = float(np.max(pos[:, 0])) if len(pos) else 150.0
        y_max = float(np.max(pos[:, 1])) if len(pos) else 150.0
        for ch in range(channels):
            z_lo, z_hi = ((0.0, 35.0), (35.0, 60.0), (60.0, 100.0))[min(ch, 2)]
            mask = (pos[:, 2] >= z_lo) & (pos[:, 2] < z_hi if ch < 2 else pos[:, 2] <= z_hi)
            pool_idx = candidates_arr[mask]
            pool_pos = pos[mask]
            if len(pool_idx) < bins_x * bins_y * neurons_per_route:
                pool_idx, pool_pos = candidates_arr, pos
            used = set()
            for by in range(bins_y):
                for bx in range(bins_x):
                    cx = ((bx + 0.5) / bins_x) * x_max
                    cy = ((by + 0.5) / bins_y) * y_max
                    d2 = (pool_pos[:,0]-cx)**2 + (pool_pos[:,1]-cy)**2
                    order = np.argsort(d2)
                    chosen = []
                    for oi in order:
                        idx = int(pool_idx[oi])
                        if idx in used and len(order) > len(used):
                            continue
                        chosen.append(idx)
                        used.add(idx)
                        if len(chosen) >= neurons_per_route:
                            break
                    routes[(bx, by, ch)] = chosen

        with self.lock:
            self.sensory_relay_routes = routes
            self._sensory_routes_ready = True

        logging.info(
            f"🔌 [Vision] Rutas sensoriales estructurales activadas: "
            f"{len(routes)} celdas, {required} neuronas relé, sin radio por frame."
        )

    def ensure_sensory_routes_ready(self) -> bool:
        """Garantiza que la grilla sensorial está construida.

        Llámalo desde cualquier módulo que necesite acceder a
        sensory_relay_routes antes del primer inject_sensory_activity.
        Devuelve True si la grilla quedó lista, False si falló.
        """
        if self._sensory_routes_ready:
            return True
        try:
            self._build_sensory_relay_routes()
            return True
        except Exception as exc:
            logging.warning(f"[Network] ensure_sensory_routes_ready falló: {exc}")
            return False

    @staticmethod
    def _sensory_channel(z):
        """Agrupa canales físicos y outputs de detectores sin usar distancia."""
        if z < 35.0:
            return 0       # movimiento / magno
        if z < 60.0:
            return 1       # estático / parvo
        return 2           # características visuales y capas derivadas

    def inject_sensory_activity(self, points):
        """Inyecta estímulos a receptores visuales estructurales fijos.

        Ya no busca neuronas por radio ni por distancia en cada frame.
        Cada punto se cuantiza a una celda visual y activa el relé fijo de
        esa celda; a partir de ahí la propagación ocurre por las sinapsis
        normales de la red.
        """
        if not points:
            return

        if not self._sensory_routes_ready:
            self._build_sensory_relay_routes()

        with self.lock:
            t = float(self.current_time)
            routes = dict(self.sensory_relay_routes)

        aggregated = {}
        for p in points:
            try:
                x = min(0.999999, max(0.0, float(p[0])))
                y = min(0.999999, max(0.0, float(p[1])))
                z = float(p[2]) if len(p) > 2 else 95.0
                intensity = max(0.0, float(p[3]) if len(p) > 3 else 7.0)
            except (TypeError, ValueError, IndexError):
                continue

            bx = min(7, int(x * 8.0))
            by = min(7, int(y * 8.0))
            ch = min(2, self._sensory_channel(z))
            for idx in routes.get((bx, by, ch), ()): 
                prev = aggregated.get(int(idx))
                if prev is None or intensity > prev:
                    aggregated[int(idx)] = intensity

        if not aggregated:
            return

        with self.lock:
            for idx, strength in aggregated.items():
                self.receive_spike(idx, strength, t, origin="sensory")

    def process_event(self, t_event, idx, strength, origin=None, event_seq=None):
        # Acepta tanto la firma histórica de 4 elementos
        # (t, idx, strength, origin) como la tupla causal de 5 elementos
        # (t, event_seq, idx, strength, origin) pasada con process_event(*ev).
        if (
            isinstance(event_seq, str)
            and isinstance(origin, (int, float, np.integer, np.floating))
            and isinstance(strength, (int, np.integer))
        ):
            legacy_event_seq = idx
            legacy_idx = strength
            legacy_strength = origin
            legacy_origin = event_seq
            t_event = float(t_event)
            idx = int(legacy_idx)
            strength = float(legacy_strength)
            origin = legacy_origin
            event_seq = int(legacy_event_seq)
        with self.lock:
            if event_seq is None:
                event_seq = self.temporal_core.next_event(float(t_event), origin=origin)
            else:
                self.temporal_core.observe_event(int(event_seq), float(t_event), origin=origin)
            self._processing_event_origin = origin
            self.event_queue_stats["processed_total"] += 1
            if origin == "feature_bus":
                self.event_queue_stats["feature_bus_processed"] += 1
                self.event_queue_stats["feature_bus_pending"] = max(0, int(self.event_queue_stats.get("feature_bus_pending", 0)) - 1)
            self.event_queue_stats["last_queue_size"] = len(self.event_queue)
        if not (0 <= idx < self.n) or t_event < self.refractory_until[idx]:
            return

        # v1.6.18: conservar el potencial previo al procesamiento para diagnóstico causal.
        pre_v = float(self.membrane_potential[idx])
        refractory_blocked = bool(t_event < self.refractory_until[idx])
        dt = t_event - self.last_update_time[idx]

        # --- FIX: blindaje contra dt negativo (condición de carrera entre
        # hilos que empujan al event_queue con timestamps de momentos
        # distintos) y contra exponentes extremos que overflowean al
        # castear a float32 en membrane_potential.
        if dt < 0:
            dt = 0.0

        exponente = -dt / self.tau_m[idx]
        exponente = min(700.0, max(-700.0, exponente))  # exp(700) ya roza el límite de float64

        v_decayed = V_REST + (self.membrane_potential[idx] - V_REST) * math.exp(exponente)
        v_now = v_decayed + strength
        self.membrane_potential[idx] = np.clip(v_now, -1e6, 1e6)  # blindaje extra si strength también se desboca
        self.last_update_time[idx] = t_event

        threshold = float(self.v_thresh[idx])
        try:
            lang = self.modules.get("lenguaje")
            target_map = getattr(self, "readout_target_to_symbol", {})
            if int(idx) in target_map and hasattr(lang, "get_readout_effective_threshold"):
                threshold = float(lang.get_readout_effective_threshold(int(idx), float(t_event), threshold))
        except Exception:
            pass

        fired = bool(v_now >= threshold)
        try:
            lang = self.modules.get("lenguaje")
            if lang is not None and hasattr(lang, "note_readout_delivery") and self.is_readout_target(int(idx)):
                lang.note_readout_delivery(
                    int(idx), float(t_event), float(strength), pre_v, float(v_decayed),
                    float(v_now), float(threshold), refractory_blocked=bool(refractory_blocked),
                    fired=fired, base_threshold=float(self.v_thresh[idx]),
                )
        except Exception:
            pass
        if fired:
            self._execute_spike(idx, t_event)

    def _execute_spike(self, idx, t):
        # --- FASE 1: PROTEGIDA ---
        with self.lock:
            self.fired[idx] = 1
            self.last_spike[idx] = float(t)
            self.membrane_potential[idx] = V_RESET
            self.refractory_until[idx] = t + self.refr_period[idx]
    
            # FASE 1: Eliminadas las variables is_coord_x, is_coord_y, is_fovea
            # worker_spiking es el único responsable de los votos motores
            my_nt          = self.nt_vector[idx].copy()
            my_pos         = self.positions[idx].copy()
            my_connections = list(self.connections[idx])

        # v1.2: contador causal de spikes del readout. Esto registra el spike real
        # de la neurona en el instante de disparo, no una inferencia basada en last_spike.
        try:
            lang = self.modules.get("lenguaje")
            if lang is not None and hasattr(lang, "note_readout_spike"):
                lang.note_readout_spike(idx, float(t))
        except Exception:
            pass

        # --- FASE 2: PROPAGACIÓN PURA ---
        # El motor ocular fue eliminado de acá.
        # worker_spiking acumula votos → update_fovea_physics los consume.
        # (El bloque intermedio de recolección de puertos genéricos fue eliminado
        # para evitar redundancia y mejorar drásticamente el rendimiento).
        # v0.8: bajo congestión se limita el fan-out y se priorizan las
        # transmisiones más fuertes. En régimen normal la topología no cambia.
        try:
            with self.lock:
                qsize_now = len(self.event_queue)
        except Exception:
            qsize_now = 0

        fanout_cap = None
        if qsize_now >= 20000:
            fanout_cap = 2
        elif qsize_now >= 16000:
            fanout_cap = 3
        elif qsize_now >= 12000:
            fanout_cap = 5
        elif qsize_now >= 8000:
            fanout_cap = 8

        candidates = []
        raw_readout = []
        for target in my_connections:
            if not self.active[target]:
                continue
            base_weight = self.weights.get((idx, target), 0.5)
            affinity = np.dot(my_nt, self.receptors[target])
            chemical_factor = 0.5 + (affinity * 0.5)
            final_strength = base_weight * chemical_factor
            if getattr(self, 'ignition_active', False):
                final_strength *= IGNITION_BOOST_FACTOR
            item = (float(final_strength), int(target))
            candidates.append(item)
            if int(target) in (getattr(self, 'readout_target_to_symbol', {}) or {}):
                raw_readout.append(item)

        original_candidate_count = len(candidates)
        target_map = getattr(self, "readout_target_to_symbol", {}) or {}
        # v1.6.15-cal: acondicionador competitivo por fuente calibrado a la excitabilidad real.
        # La 1.6.14 dejaba el presupuesto total en <=0.80, mientras el pool
        # necesitaba ~1.55 mV para disparar y cada fuente cortical sólo aporta
        # una fracción a cada clase. Resultado: actividad cortical real pero
        # cero spikes en todos los pools.
        # Aquí elevamos el presupuesto total por fuente a una ventana funcional
        # calibrada (2.80..4.00), manteniendo el reparto softmax por pesos. Esto
        # NO añade afinidad de clase: sólo corrige la interfaz energética.
        routing_temp = max(1e-4, float(getattr(self, 'readout_routing_temperature', 0.35)))
        min_budget = float(getattr(self, 'readout_routing_min_budget', 2.80))
        max_budget = float(getattr(self, 'readout_routing_max_budget', 4.00))
        routing_gain = float(getattr(self, 'readout_synaptic_gain', 1.10))
        if raw_readout:
            raw_strengths = np.asarray([max(0.0, float(v)) for v, _ in raw_readout], dtype=float)
            logits = raw_strengths / routing_temp
            logits -= float(np.max(logits))
            probs = np.exp(np.clip(logits, -40.0, 40.0))
            probs /= max(1e-12, float(np.sum(probs)))
            mean_raw = float(np.mean(raw_strengths))
            total_budget = float(np.clip(routing_gain * mean_raw, min_budget, max_budget))
            # Telemetría explícita del presupuesto que realmente llega al readout.
            try:
                self.event_queue_stats["readout_calibrated_budget_sum"] = (
                    self.event_queue_stats.get("readout_calibrated_budget_sum", 0.0) + total_budget
                )
                self.event_queue_stats["readout_calibration_events"] = (
                    self.event_queue_stats.get("readout_calibration_events", 0) + 1
                )
            except Exception:
                pass
            readout_candidates = [
                (float(total_budget * float(prob)), int(target))
                for prob, (_, target) in zip(probs, raw_readout)
            ]
        else:
            readout_candidates = []
        normal_candidates = [c for c in candidates if int(c[1]) not in target_map]
        if getattr(self, "_causal_trial_active", False):
            # v1.6.25: selección por utilidad física + presupuesto temporal; no por
            # recorte ciego del fan-out. Esto preserva transiciones potencialmente útiles.
            budget = int(getattr(self, "_causal_frame_budget", 0))
            remaining = max(0, budget - int(getattr(self, "_causal_frame_fanout_attempts", 0))) if budget else len(normal_candidates)
            scored = []
            for strength, target in normal_candidates:
                self.event_queue_stats["causal_utility_considered"] += 1
                if self._causal_physical_prefilter(target, strength, float(t)):
                    self.event_queue_stats["causal_prefilter_drops"] += 1
                    continue
                utility = self._causal_candidate_utility(target, strength, float(t))
                if utility < float(CAUSAL_EVENT_UTILITY_MIN):
                    self.event_queue_stats["causal_prefilter_drops"] += 1
                    continue
                scored.append((float(utility), float(strength), int(target)))
            scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
            if remaining < len(scored):
                limited = len(scored) - remaining
                self._causal_frame_fanout_limited += max(0, limited)
                self.event_queue_stats["causal_budget_limited"] += max(0, limited)
                scored = scored[:remaining]
            normal_candidates = [(strength, target) for _, strength, target in scored]
            self.event_queue_stats["causal_utility_selected"] += len(normal_candidates)
        elif fanout_cap is not None and len(normal_candidates) > fanout_cap:
            before_cap = len(normal_candidates)
            normal_candidates.sort(key=lambda x: (-x[0], x[1]))
            normal_candidates = normal_candidates[:fanout_cap]

        # Las 5 conexiones de readout por fuente forman un canal dedicado de bajo
        # fan-out; no compiten con la poda de la red general.
        candidates = readout_candidates + normal_candidates
        if len(candidates) < original_candidate_count:
            with self.lock:
                self.event_queue_stats["fanout_limited_total"] += original_candidate_count - len(candidates)

        with self.lock:
            self.event_queue_stats["fanout_attempted_total"] += len(candidates)

        for final_strength, target in candidates:
            with self.lock:
                self.event_queue_stats["propagation_attempts"] += 1
                qsize_now = len(self.event_queue)

            target_pos = self.positions[target]
            dist = math.sqrt(sum((my_pos[k] - target_pos[k])**2 for k in range(3)))
            arrival_time = t + SYNAPTIC_DELAY_BASE + (dist * 0.01)

            # v1.2: el readout ya no queda globalmente exento del governor.
            # Se mantiene la misma política de presión, registrando por separado
            # las pérdidas que afectan específicamente a sus pools.
            target_is_readout = target in getattr(self, "readout_target_to_symbol", {})
            if target_is_readout and int(idx) not in getattr(self, "readout_allowed_sources", set()):
                # 1.6.16: ningún origen ajeno a la representación cortical
                # puede introducir actividad en los pools del clasificador.
                with self.lock:
                    self.event_queue_stats["readout_foreign_source_blocked"] = (
                        self.event_queue_stats.get("readout_foreign_source_blocked", 0) + 1
                    )
                continue
            # Un readout target pertenece al canal funcional protegido: llega
            # directamente a la cola dedicada y NO atraviesa el governor de la
            # dinámica emergente. El límite de seguridad es readout_queue_max.
            if target_is_readout:
                # v1.14: final_strength ya viene del acondicionador competitivo
                # por fuente; volver a multiplicarlo destruiría la limitación de
                # presupuesto y volvería a saturar los cinco pools.
                with self.lock:
                    self.event_queue_stats["readout_priority_passes"] = (
                        self.event_queue_stats.get("readout_priority_passes", 0) + 1
                    )
            elif target not in self.governor_exempt_targets:
                min_prop_strength = 0.0
                if qsize_now >= 20000:
                    min_prop_strength = 1.75
                elif qsize_now >= 16000:
                    min_prop_strength = 1.45
                elif qsize_now >= 12000:
                    min_prop_strength = 1.05
                elif qsize_now >= 8000:
                    min_prop_strength = 0.65
                elif qsize_now >= 5000:
                    min_prop_strength = 0.35

                if final_strength < min_prop_strength:
                    with self.lock:
                        self.event_queue_stats["dropped_by_governor"] += 1
                        self.event_queue_stats["propagation_governor_drops"] += 1
                        self.event_queue_stats["dropped_total"] += 1
                    continue
            if target_is_readout:
                try:
                    lang = self.modules.get("lenguaje")
                    if lang is not None and hasattr(lang, "note_readout_eligibility"):
                        lang.note_readout_eligibility(int(idx), int(target), float(arrival_time), float(final_strength))
                except Exception:
                    pass
            accepted = self.receive_spike(target, final_strength, arrival_time, origin="synaptic")
            if accepted:
                with self.lock:
                    self.event_queue_stats["fanout_sent_total"] += 1
            elif target_is_readout:
                with self.lock:
                    self.event_queue_stats["readout_governor_drops"] += 1

    def add_neuron(self, pos=None, region_id=0, parent_idx=None):
         with self.lock:
             if self.n >= self.max_neurons:
                 return None
             idx = self.n

             # Autorreparación
             for attr in ['neuron_region', 'neuron_subregion']:
                 a = getattr(self, attr)
                 if len(a) < self.max_neurons:
                     new_a = np.full(self.max_neurons, -1, dtype=np.int32)
                     new_a[:len(a)] = a
                     setattr(self, attr, new_a)

             self.active[idx] = True
             self.positions[idx] = pos if pos is not None else [75., 75., 50.]
             self.neuron_region[idx] = int(region_id)
             self.energy[idx] = 0.6
             self.membrane_potential[idx] = V_REST
             self.connections[idx] = []
             self.dna[idx] = list(self.dna[parent_idx]) if parent_idx is not None else self._generate_initial_dna()

             self.n += 1
             return idx

    def reset_noncompetitive_dynamic_state(self):
        """Reestablece mínimos funcionales tras una sesión sin borrar aprendizaje explícito."""
        protected = [i for i in self.noncompetitive_neurons if 0 <= i < self.n and self.active[i]]
        if not protected:
            return
        for idx in protected:
            self.energy[idx] = max(float(self.energy[idx]), 0.75)
        self.event_queue_stats["noncompetitive_registered"] = len(self.noncompetitive_neurons)

    def _init_connections_sparse(self):
        print("🔌 Tejiendo conexiones sparse...")
        for i in range(self.n):
            r_id = self.neuron_region[i]
            s_id = self.neuron_subregion[i]
            radius = self.regions[r_id].sub_regions[s_id].current_profile.conn_radius if s_id != -1 else RADIUS_3D

            for j in range(i+1, self.n):
                if np.linalg.norm(self.positions[i] - self.positions[j]) <= radius:
                    self._connect(i, j, random.uniform(0.3, 0.7))

    def _connect(self, i, j, weight):
        with self.lock:
            if j not in self.connections[i]: self.connections[i].append(j)
            if i not in self.connections[j]: self.connections[j].append(i)
            self.weights[(i, j)] = weight
            self.weights[(j, i)] = weight

    def apply_dopamine_reward(self, visual_success, habituacion=1.0):
        for region in self.regions.values():
            r_val = self.reward_manager.compute_reward(region, self, visual_success, habituacion)
            self.reward_manager.apply_dopamine(region, r_val, self)

        # Sincronizar dopamine_level global con el promedio real de regiones
        if self.regions:
            self.dopamine_level = sum(r.dopamine for r in self.regions.values()) / len(self.regions)

        # === FITNESS NEURONAL: ENTRENAMIENTO ===
        # Si hay éxito visual y dopamina alta, las neuronas activas
        # suben su fitness (se "entrenan"). Si no hay éxito, decae levemente.
        if not hasattr(self, 'fitness_neuronal'):
            return

        dopamina_actual = self.dopamine_level

        if visual_success and dopamina_actual > 0.15:
            # Recompensa de entrenamiento: neuronas que dispararon suben su fitness
            # La ganancia es proporcional a la dopamina y la habituación
            ganancia = 0.002 * dopamina_actual * habituacion
            mascara_activas = self.fired[:self.n].astype(bool) & self.active[:self.n]
            self.fitness_neuronal[:self.n][mascara_activas] += ganancia
        else:
            # Sin éxito: pequeño decaimiento del fitness (desentrenamiento lento)
            decaimiento = 0.0001
            self.fitness_neuronal[:self.n][self.active[:self.n]] -= decaimiento

        # Mantener dentro del rango biológico [0.5, 2.0]
        self.fitness_neuronal[:self.n] = np.clip(
            self.fitness_neuronal[:self.n], 0.5, 2.0
        )
        
        
    
    def inject_inhibitory_signal(self, dolor_multiplier: float):
        """
        Aplica un 'calambre' inhibitorio masivo a la capa motora (Z=60)
        para paralizar los motores que empujan contra la pared.
        """
        # Multiplicador fuerte negativo (ej: -50mV si dolor es máximo)
        fuerza_inhibicion = -50.0 * dolor_multiplier 
        
        with self.lock:
            # OPICIÓN A: Si usas matrices Numpy (Recomendado para SNN rápidas)
            if hasattr(self, 'voltages') and hasattr(self, 'positions'):
                # Busca las neuronas motoras (Capa Z cercana a 60)
                mascara_motora = (self.positions[:, 2] >= 55.0) & (self.positions[:, 2] <= 65.0)
                # Inyecta el voltaje negativo
                self.voltages[mascara_motora] += fuerza_inhibicion
                
            # OPCIÓN B: Si usas una lista de objetos 'Neuron' en Python puro
            elif hasattr(self, 'neurons'):
                for n in self.neurons:
                    if hasattr(n, 'z') and 55.0 <= n.z <= 65.0:
                        n.voltage += fuerza_inhibicion    