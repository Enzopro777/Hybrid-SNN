#core/region.py


import random
import time
import numpy as np
from typing import Optional, Any
from dataclasses import dataclass
from collections import deque

@dataclass
class Profile:
    """ADN de comportamiento eléctrico para una SubRegión."""
    name: str
    tau_m: float           # Constante de tiempo de membrana (Leaky)
    v_thresh: float        # Umbral de disparo
    refr_period: float     # Periodo refractario (ms)
    energy_drain: float    # Costo metabólico extra por paso (Fase 2)
    conn_radius: float     # Radio de búsqueda para sinapsis

class SubRegion:
    def __init__(self, sub_id: int, parent_id: int, profile: Profile):
        self.sub_id = sub_id
        self.parent_id = parent_id
        self.neurons = []  # Lista de índices de neuronas en NeuralNetwork
        
        # --- ESTADO EVOLUTIVO ---
        self.current_profile = profile
        self.best_profile = profile
        self.backup_profile = None  # Snapshot para Rollback
        
        # --- MONITOREO DE RENDIMIENTO ---
        self.stress_level = 0.0
        self.saturation = 0.0      # % de actividad neuronal reciente
        self.success_history = deque(maxlen=50) 
        
        # --- CONTROL DE MUTACIÓN (Fase 3) ---
        self.last_mutation_time = 0.0
        self.eval_steps_left = 0    # Contador de pasos para Paso 3
        self.is_evaluating = False

    def add_neuron(self, neuron_idx: int):
        if neuron_idx not in self.neurons:
            self.neurons.append(neuron_idx)

    def calculate_stress(self, regional_dopamine: float):
        """Motor de Estrés v1.5: Conversión forzada de tipos para evitar errores de NumPy."""
        import numpy as np
        from collections import deque

        # 1. Cálculo base
        ws, wd = 0.4, 0.6
        dopamine_term = 1.0 - max(0.0, min(1.0, regional_dopamine))
        self.stress_level = (ws * self.saturation) + (wd * dopamine_term)

        # --- REPARACIÓN DINÁMICA DE TIPO ---
        # Si success_history es un array de numpy o no tiene el método append...
        if isinstance(self.success_history, np.ndarray) or not hasattr(self.success_history, "append"):
            # Lo convertimos a una lista clásica de Python y luego a deque
            try:
                # Si es un array de numpy, .tolist() es lo más seguro
                data_list = self.success_history.tolist() if hasattr(self.success_history, "tolist") else list(self.success_history)
            except:
                data_list = []
            
            # Re-inicializamos como deque con límite de 50 para no saturar la RAM
            self.success_history = deque(data_list, maxlen=50)
        # -----------------------------------

        # 2. Ahora el .append() funcionará SIEMPRE
        self.success_history.append(1.0 - self.stress_level)
        
        return self.stress_level

    # --- LÓGICA DE FASE 3: CICLO ESPECULATIVO ---

    def initiate_speculative_jump(self, available_profiles: list, eval_window: int = 100):
        """
        Paso 1: Snapshot Quirúrgico.
        Paso 2: Salto Evolutivo.
        """
        # Guardamos solo el perfil (Snapshot) para posible Rollback
        self.backup_profile = self.current_profile
        
        # Selección reactiva al tipo de estrés
        if self.saturation > 0.8:
            # Si hay saturación, priorizamos perfiles de alta velocidad (RAPID_FIRE)
            candidates = [p for p in available_profiles if p.energy_drain > self.current_profile.energy_drain]
        else:
            # Si el estrés es por dopamina, probamos variedad
            candidates = [p for p in available_profiles if p.name != self.current_profile.name]

        if candidates:
            self.current_profile = random.choice(candidates)
            
        self.eval_steps_left = eval_window
        self.is_evaluating = True
        self.last_mutation_time = time.time()
        
        print(f"🧬 [JUMP] SubRegión {self.sub_id}: Probando {self.current_profile.name} por {eval_window} pasos.")



    def evaluate_jump(self, net: Optional[Any] = None):
        """
        Paso 4: Decisión (Rollback o Commit).
        Actualización física de sinapsis con blindaje contra AttributeError y KeyError de NumPy.
        """
        if not self.is_evaluating:
            return

        # 1. Análisis de éxito
        # Calculamos el promedio histórico para comparar con el rendimiento actual
        avg_historical_success = sum(self.success_history) / len(self.success_history) if self.success_history else 0.5
        current_success = 1.0 - getattr(self, 'stress_level', 0.5)

        # --- Extracción segura de parámetros de perfil ---
        # Blindaje contra perfiles incompletos usando getattr
        curr_rad = getattr(self.current_profile, 'conn_radius', 25.0)
        back_rad = getattr(self.backup_profile, 'conn_radius', 25.0)

        # 2. Lógica de Decisión
        if current_success > avg_historical_success:
            # --- COMMIT: La mutación es beneficiosa ---
            self.best_profile = self.current_profile
            print(f"✅ [COMMIT] SubRegión {self.sub_id}: Perfil {self.current_profile.name} estabilizado.")
            
            # --- ACTUALIZACIÓN FÍSICA (Re-Wiring) ---
            if net and hasattr(net, 'refresh_neuron_connections'):
                if curr_rad != back_rad:
                    print(f"🔌 [Re-Wiring] Radio cambió ({back_rad} -> {curr_rad}). Re-escaneando...")
                    for idx in self.neurons:
                        # FIX: Forzamos int() para evitar KeyError: np.int32(0) en el diccionario de red
                        safe_idx = int(idx)
                        if net.active[safe_idx]:
                            net.refresh_neuron_connections(safe_idx)
        else:
            # --- ROLLBACK: La mutación falló, regresamos al snapshot anterior ---
            print(f"⏪ [ROLLBACK] SubRegión {self.sub_id}: Regresando a {self.backup_profile.name}.")
            self.current_profile = self.backup_profile
            
            # Restauramos el estado de salud regional si es posible
            if hasattr(self, 'energy_buffer'):
                self.energy_buffer = 1.0 

        # 3. Limpieza y Finalización del Ciclo
        self.is_evaluating = False
        self.eval_steps_left = 0 # Asegúrate de que coincida con el nombre de tu contador
        
        # Limpiamos el historial para empezar la medición del nuevo perfil desde cero
        if hasattr(self, 'success_history') and hasattr(self.success_history, 'clear'):
            self.success_history.clear()

class Region:
    def __init__(self, region_id: int):
        self.id = region_id
        self.sub_regions = {} # {sub_id: SubRegion}

        # === ADN REGIONAL (Firma Química) ===
        self.signature = [random.uniform(0.1, 0.5) for _ in range(5)]
        self._normalize_signature()

        # === NEUROMODULADORES ===
        self.dopamine = 0.1
        self.serotonin = 0.5
        self.acetylcholine = 0.4

        # === ESTADO METABÓLICO ===
        self.energy = 1.0
        self.prev_energy = 1.0
        self.is_active = True 

        # === MÉTRICAS DE RENDIMIENTO ===
        self.activity = 0.0
        self.prev_activity = 0.0
        self.output = 0.0
        self.fitness = 1.0 
        self.stress = 0.0
        self.novelty = 0.0
        self.env_interaction = 0.0

        # === ESTRATEGIA Y CONTROL ===
        self.strategy = "balanced"
        self.low_dopamine_counter = 0
        self._energy_lock = None
        try:
            import threading
            self._energy_lock = threading.RLock()
        except Exception:
            self._energy_lock = None

    # --- GESTIÓN DE JERARQUÍA ---

    def create_sub_region(self, sub_id: int, profile: Profile) -> SubRegion:
        """Crea un micro-circuito con un perfil genético específico."""
        new_sub = SubRegion(sub_id, self.id, profile)
        self.sub_regions[sub_id] = new_sub
        return new_sub

    def get_avg_subregion_stress(self) -> float:
        if not self.sub_regions:
            return self.stress
        return sum(s.stress_level for s in self.sub_regions.values()) / len(self.sub_regions)

    # --- DINÁMICA QUÍMICA ---

    def _normalize_signature(self):
        total = sum(self.signature) or 1.0
        self.signature = [s / total for s in self.signature]

    def update_signature(self, avg_nt_vector, lr=0.05):
        if avg_nt_vector is None or len(avg_nt_vector) == 0: 
            return
        for i in range(len(self.signature)):
            self.signature[i] += (avg_nt_vector[i] - self.signature[i]) * lr
        self._normalize_signature()

    # --- DINÁMICA METABÓLICA ---

    def update_energy(self, delta_energy: float):
        """Actualiza energía regional en una sola publicación ya acotada."""
        lock = self._energy_lock
        if lock is None:
            next_energy = (float(self.energy) + float(delta_energy)) * 0.998
            self.energy = max(0.1, min(2.5, next_energy))
            return
        with lock:
            next_energy = (float(self.energy) + float(delta_energy)) * 0.998
            self.energy = max(0.1, min(2.5, next_energy))

    def adjust_energy(self, delta_energy: float):
        """Ajuste metabólico acotado sin aplicar el decaimiento basal."""
        lock = self._energy_lock
        if lock is None:
            self.energy = max(0.1, min(2.5, float(self.energy) + float(delta_energy)))
            return
        with lock:
            self.energy = max(0.1, min(2.5, float(self.energy) + float(delta_energy)))

    def stabilize_dopamine(self):
        if abs(self.dopamine) > 0.01:
            self.dopamine *= 0.95
        self.dopamine = max(-0.6, min(0.8, self.dopamine))

    def update_strategy(self):
        """Cambia el modo de operación según los recursos y estrés regional."""
        if self.energy < 0.6:
            self.strategy = "conservative"
        elif self.stress > 0.8:
            self.strategy = "defensive"
        elif self.dopamine > 0.4:
            self.strategy = "explorer"
        else:
            self.strategy = "balanced"

    def update_state(
        self,
        activity: float,
        output: float,
        energy_input: float,
        env_signal: float,
        net: Optional[Any] = None
    ) -> None:
        """Ciclo principal de actualización (Metabolismo + Estrés Sub-Regional)."""
        self.prev_activity = self.activity
        self.activity = activity
        self.output = output
        self.prev_energy = self.energy

        # 1. PROCESAMIENTO DE SUB-REGIONES (Fase 2)
        extra_profile_cost = 0.0
        if self.sub_regions and net is not None:
            for sub in self.sub_regions.values():
                # Calcular saturación (actividad real de sus neuronas)
                if sub.neurons:
                    fired_now = sum(1 for idx in sub.neurons if net.fired[idx])
                    sub.saturation = fired_now / len(sub.neurons)
                
                # Actualizar estrés individual basado en la dopamina actual de la región
                sub.calculate_stress(self.dopamine)
                
                # Sumar costo energético del perfil (Rapid_Fire gasta más)
                extra_profile_cost += sub.current_profile.energy_drain

        # 2. COSTO METABÓLICO
        base_cost = abs(self.activity) * 0.05
        mults = {"conservative": 0.5, "explorer": 1.4, "defensive": 0.8, "balanced": 1.0}
        
        # El costo total suma la actividad general y el mantenimiento de perfiles
        activity_cost = (base_cost * mults.get(self.strategy, 1.0)) + (extra_profile_cost * 0.1)
        self.update_energy(energy_input - activity_cost)

        # 3. ACTUALIZACIÓN DE MÉTRICAS REGIONALES
        self.fitness = (self.energy * 0.4) + (self.dopamine * 0.6)
        self.stress = max(0.0, 1.0 - self.energy)
        self.novelty = abs(self.activity - self.prev_activity)
        self.env_interaction = activity * env_signal

        self.update_strategy()

        # 4. DINÁMICA DE NEUROMODULADORES
        delta_e = self.energy - self.prev_energy
        target_dopamine = (delta_e * 1.5) + (self.novelty * 0.4) + (env_signal * 0.2)
        
        self.dopamine += (target_dopamine - self.dopamine) * 0.1
        self.serotonin += ((1.0 - self.stress) - self.serotonin) * 0.05
        self.acetylcholine += ((self.novelty + self.stress) - self.acetylcholine) * 0.1

        self.stabilize_dopamine()

        # Recuperación de depresión dopaminérgica
        if self.dopamine < -0.3:
            self.low_dopamine_counter += 1
            if self.low_dopamine_counter > 30:
                self.dopamine = 0.1
                self.low_dopamine_counter = 0
        else:
            self.low_dopamine_counter = 0
            
    # --- COMPETENCIA INTER-REGIONAL ---
    
    def apply_lateral_inhibition(self, other_regions):
        REGION_INHIBITION_STRENGTH = 0.5
        REGION_DOMINANCE_THRESHOLD = 0.8

        if self.activity > REGION_DOMINANCE_THRESHOLD:
            bias_force = (self.activity - REGION_DOMINANCE_THRESHOLD) * REGION_INHIBITION_STRENGTH
            
            for other in other_regions:
                if other.id == self.id or not other.is_active:
                    continue
                
                inhibition_debt = bias_force * 0.15 
                
                # FIX: aplicar el mismo clamp que usa update_energy, en vez
                # de sumar/restar sin límite
                if hasattr(other, "adjust_energy"):
                    other.adjust_energy(-inhibition_debt)
                else:
                    other.energy = max(0.1, min(2.5, float(other.energy) - inhibition_debt))
                if hasattr(self, "adjust_energy"):
                    self.adjust_energy(inhibition_debt * 0.5)
                else:
                    self.energy = max(0.1, min(2.5, float(self.energy) + inhibition_debt * 0.5))
                other.stress = min(1.0, other.stress + (bias_force * 0.2))

    def calculate_regional_gain(self) -> float:
        """Determina la sensibilidad de disparo de la región."""
        gain = (self.energy * 0.7) + (self.dopamine * 0.3)
        return max(0.5, min(2.0, gain))