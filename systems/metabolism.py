

# systems/metabolism.py
import numpy as np

class MetabolismSystem:
    def __init__(self):
        # Umbrales críticos
        self.death_threshold = 0.15      # Si baja de aquí, la neurona muere
        self.atrophy_limit = 8000.0      # ms de silencio antes de considerar atrofia severa
        print("🧬 Sistema Metabólico v3.2 (Selección Natural) inicializado.")

    
    
    
    
    def get_status(self, net):
        """Devuelve un resumen de salud de la red."""
        active_count = np.sum(net.active)
        avg_energy = np.mean(net.energy[net.active]) if active_count > 0 else 0
        return f"Salud: {avg_energy:.2f} | Vivas: {active_count}"
    
    
    
            
    def update(self, net, t, dt, state="awake"):
        if net.n == 0: return

        r_attr = 'neuron_region' if hasattr(net, 'neuron_region') else 'region'
        region_indices = getattr(net, r_attr)

        heart_pulse = 0.0
        if hasattr(net, 'heart') and net.heart.active:
            heart_pulse = getattr(net.heart.pulse_port, 'value', 0.0)
        global_nutrient = heart_pulse * 0.02

        # Parámetros por estado
        if state == "awake":
            base_cost = 0.0012
            recovery  = 0.0025
        elif state == "drowsy":
            base_cost = 0.0008
            recovery  = 0.004
        else:  # sleep
            base_cost = 0.0003
            recovery  = 0.009

        # Verificar si el sistema de fitness está disponible
        tiene_fitness = hasattr(net, 'fitness_neuronal')

        active_idx = np.flatnonzero(net.active[:net.n])
        if active_idx.size == 0:
            return

        # Trabajo vectorizado para el consumo/recuperación base. La parte
        # dependiente de región/muerte queda en un recorrido reducido.
        last_sp = net.last_spike[active_idx]
        time_since = np.maximum(0.0, float(t) - last_sp)
        atrophy = np.ones(active_idx.size, dtype=np.float32)
        mask_atrophy = time_since > self.atrophy_limit
        atrophy[mask_atrophy] = np.minimum(3.0, 1.0 + (time_since[mask_atrophy] - self.atrophy_limit) / 5000.0)
        base_costs = base_cost * atrophy
        net.energy[active_idx] -= base_costs + 0.035 * net.fired[active_idx]

        protected_mask = np.array([hasattr(net, 'is_noncompetitive') and net.is_noncompetitive(int(i)) for i in active_idx], dtype=bool)
        region_idx = region_indices[active_idx]
        zone_eff = np.ones(active_idx.size, dtype=np.float32)
        for rid, region in net.regions.items():
            m = region_idx == rid
            if np.any(m):
                zone_eff[m] = 1.0 + (region.acetylcholine * 1.5)
        net.energy[active_idx] += (recovery + global_nutrient) * zone_eff

        for pos, i in enumerate(active_idx.tolist()):
            i = int(i)
            time_since_spike = float(time_since[pos])
            atrophy_factor = float(atrophy[pos])

            r_idx = region_indices[i]
            # 3. MUERTE POR INANICIÓN
            protected = hasattr(net, 'is_noncompetitive') and net.is_noncompetitive(i)
            if net.energy[i] < self.death_threshold and not protected:
                net.active[i] = False
                net.energy[i] = 0.0
                if hasattr(net, 'region_counts') and r_idx < len(net.region_counts):
                    net.region_counts[r_idx] -= 1
                continue

            if protected:
                net.energy[i] = max(float(net.energy[i]), 0.50)
                continue

            # 4. TECHO DINÁMICO POR FITNESS NEURONAL
            # Una neurona entrenada (fitness=2.0) puede tener hasta 2x más energía
            # Una neurona atrofiada (fitness=0.5) tiene techo muy bajo
            if tiene_fitness:
                techo = float(net.fitness_neuronal[i])

                # Durante el sueño, el techo sube un 20% extra para consolidación
                if state == "sleep":
                    techo = min(2.0, techo * 1.2)

                net.energy[i] = min(techo, net.energy[i])

                # === DECAIMIENTO DE FITNESS POR INACTIVIDAD ===
                # Si la neurona lleva mucho tiempo sin disparar, pierde condición física
                if time_since_spike > self.atrophy_limit * 2:
                    decaimiento_fitness = 0.00005 * atrophy_factor
                    net.fitness_neuronal[i] = max(0.5, net.fitness_neuronal[i] - decaimiento_fitness)
            else:
                # Fallback si fitness no existe: techo fijo original
                net.energy[i] = min(1.1, net.energy[i])

        if hasattr(net, 'heart'):
            net.heart.pulse_port.value *= 0.6

    def get_sleep_time(self, stress_factor):
        return 1.0 + (stress_factor * 2.0)    
               