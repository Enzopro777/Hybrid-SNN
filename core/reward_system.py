# core/reward_system.py
import random
import logging

class RewardSystem:
    def __init__(self):
        # === PESOS DE RECOMPENSA (Fase 3) ===
        # Conservamos la arquitectura rica del proyecto, pero reducimos los
        # dos componentes visuales principales para evitar saturación.
        self.w_vision      = 0.6
        self.w_energy      = 0.6
        self.w_env         = 0.4
        self.w_novelty     = 0.4
        self.w_efficiency  = 0.7
        self.w_stress      = 1.5   # Penalización

    def compute_reward(self, region, net, visual_success=0.0, habituacion=1.0):
        """
        Calcula el valor de fitness/recompensa para una región específica.
        Incluye atenuación por habituación/inmovilidad.
        """
        try:
            # 1. DRIVE DE ENERGÍA
            delta_energy = region.energy - region.prev_energy
            drive_energy = (delta_energy * 2.0) + (region.energy * 0.2)

            # 2. NOVEDAD: Exploración de conexiones
            drive_novelty = region.novelty * 2.0

            # 3. ESTRÉS
            drive_stress = -region.stress * 1.5

            # 4. EFICIENCIA
            energy_used = max(0.01, abs(delta_energy))
            efficiency = min(2.0, region.output / energy_used)
            drive_efficiency = efficiency * 1.2

            # --- APLICAR HABITUACIÓN AL ÉXITO VISUAL ---
            # Escalamos el éxito visual antes de sumarlo al total
            visual_component = self.w_vision * visual_success * habituacion

            # --- CÁLCULO DE RECOMPENSA TOTAL ---
            reward = (
                (self.w_energy * drive_energy) +
                (self.w_env * region.env_interaction) +
                (self.w_novelty * drive_novelty) +
                (self.w_efficiency * drive_efficiency) +
                (self.w_stress * drive_stress) +
                visual_component
            )

            # --- SI ESTÁ TOTALMENTE QUIETO (Habituación crítica), PENALIZAR APATÍA ---
            # Si el parche detectó inmovilidad, bajamos la recompensa basal para que no sea rentable ser catatónico
            if habituacion <= 0.05:
                reward -= 0.4  # Penalización por aburrimiento (fuerza el escape metabólico)

            # --- MECANISMOS DE REGULACIÓN ---
            # A. Anti-Coma
            if region.activity < 0.02:
                reward += 0.3
            
            # B. Anti-Caos
            if region.activity > 0.7:
                reward -= 1.2

            # C. Control de Población
            vivas = sum(1 for a in net.active if a)
            if vivas > 500:
                reg_count = net.region_counts[region.id] if hasattr(net, 'region_counts') else 0
                if (reg_count / vivas) > 0.4: 
                    reward -= 0.6

            if region.id == 0 and random.random() < 0.05:
                print(f"DEBUG R0: Reward={reward:.2f} | Energy={region.energy:.2f} | Habituac={habituacion:.2f}")

            return reward

        except Exception as e:
            reg_id = region.id if hasattr(region, 'id') else region
            logging.error(f"Error en compute_reward Región {reg_id}: {e}")
            return 0.0

    def calculate_reward(self, visual_input, novelty_score):
        """API compacta de Fase 3 para consumidores simples.

        Usa la misma ponderación visual/novedad del sistema completo y no
        sustituye compute_reward(), que sigue siendo el motor principal.
        """
        visual = max(0.0, min(1.0, float(visual_input)))
        novelty = max(0.0, min(1.0, float(novelty_score)))
        return (visual * self.w_vision) + (novelty * self.w_novelty)

    def apply_dopamine(self, region, reward, net):
        """
        Transforma reward en Dopamina con recaptación biológica real.
        """
        # Inicializar la propiedad si no existe en la estructura de la región
        if not hasattr(region, 'dopamine'):
            region.dopamine = 0.0
        
        # 1. RECAPTACIÓN BIOLÓGICA (Clearance): La dopamina vieja decae un 20% antes de sumar la nueva
        # Esto evita que se quede estancada arriba si la recompensa baja
        region.dopamine *= 0.80
        
        # 2. Inyección de la nueva recompensa normalizada
        reward_clipped = max(-1.0, min(1.0, reward))
        
        # Suavizado (Inercia del 10%)
        region.dopamine += (reward_clipped) * 0.1
        region.dopamine = max(0.0, min(2.0, region.dopamine)) # Límites saludables

        # --- LOG DE ÉXITO CORREGIDO ---
        # Ahora printeamos el valor REAL del tanque de dopamina de la región
        if region.dopamine > 0.1:
            print(f"✨ [Dopamina Real] Región {region.id}: {region.dopamine:.2f}")

        # --- CASTIGO METABÓLICO ---
        if reward < -0.6:
            for i in range(net.n):
                if net.neuron_region[i] == region.id and net.active[i]:
                    net.energy[i] -= 0.05
                    
        # --- PREMIO GENÉTICO ---
        region.reproduction_chance = max(0.0, reward) if reward > 0.2 else 0.0