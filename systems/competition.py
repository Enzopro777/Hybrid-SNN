# systems/competition.py

import random
import numpy as np

class CompetitionSystem:
    def __init__(self):
        self.base_death_rate = 0.0002
        self.energy_death_threshold = 0.25
        self.max_deaths_per_step = 1

    def compete(self, net):
        if net.n == 0:
            return 0
        deaths = 0
        region_attr = 'neuron_region' if hasattr(net, 'neuron_region') else 'region'
        regions_map = getattr(net, 'regions', {})
        region_ids = getattr(net, region_attr)
        safe_n = min(net.n, len(net.active), len(net.energy), len(region_ids))
        indices = list(range(safe_n))
        random.shuffle(indices)

        # Conteos competitivos vectorizados y reutilizados por todo el ciclo.
        ids = np.asarray(region_ids[:safe_n], dtype=int)
        active = np.asarray(net.active[:safe_n], dtype=bool)
        protected = np.zeros(safe_n, dtype=bool)
        if hasattr(net, 'noncompetitive_neurons') and net.noncompetitive_neurons:
            valid = [i for i in net.noncompetitive_neurons if 0 <= int(i) < safe_n]
            if valid:
                protected[np.asarray(valid, dtype=int)] = True
        self._competitive_counts = {int(r): int(np.sum(active & ~protected & (ids == int(r))))
                                    for r in regions_map}

        for i in indices:
            if not net.active[i] or (hasattr(net, 'is_noncompetitive') and net.is_noncompetitive(i)):
                continue
            r_idx = int(region_ids[i])
            if r_idx not in regions_map:
                continue
            region = regions_map[r_idx]
            neuron_energy = float(net.energy[i])
            death_prob = self.base_death_rate
            if neuron_energy < self.energy_death_threshold:
                death_prob += 0.05
            if getattr(region, 'dopamine', 0.0) < 0:
                death_prob += abs(float(region.dopamine)) * 0.01
            else:
                death_prob *= 0.1

            # Conteo regional O(n) calculado una vez por ciclo; el bucle
            # anterior hacía otro barrido completo de la red por cada neurona
            # candidata (O(n²)).
            in_region = self._competitive_counts.get(r_idx, 0)
            if in_region < 50 and neuron_energy > 0.1:
                continue

            if random.random() < death_prob:
                net.active[i] = False
                net.energy[i] = 0.0
                deaths += 1
                if deaths >= self.max_deaths_per_step:
                    break

        return deaths
