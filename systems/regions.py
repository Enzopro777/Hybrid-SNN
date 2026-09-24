# systems/regions.py

import numpy as np
from config import NUM_REGIONS

class RegionSystem:
    def __init__(self):
        self.iteration = 0
        self.extinction_threshold = 0.2
        self.learning_rate_sig = 0.01

    def _competitive_mask(self, net):
        mask = np.asarray(net.active[:net.n], dtype=bool).copy()
        protected = getattr(net, 'noncompetitive_neurons', set())
        if protected:
            ids = np.fromiter((i for i in protected if 0 <= i < net.n), dtype=int)
            if len(ids):
                mask[ids] = False
        return mask

    def update(self, net):
        self.iteration += 1
        region_data = {r.id: {"potentials": [], "nt_vector": []} for r in net.regions.values()}
        competitive = self._competitive_mask(net)
        for i in range(net.n):
            if not competitive[i]:
                continue
            rid = int(net.neuron_region[i])
            if rid in region_data:
                region_data[rid]["potentials"].append(net.membrane_potential[i])
                region_data[rid]["nt_vector"].append(net.nt_vector[i])

        active_list = [r for r in net.regions.values() if r.is_active]
        for r in active_list:
            data = region_data[r.id]
            avg_act = np.mean(data["potentials"]) if data["potentials"] else 0.0
            avg_nt = np.mean(data["nt_vector"], axis=0) if data["nt_vector"] else None
            if avg_nt is not None:
                r.update_signature(avg_nt, lr=self.learning_rate_sig)
            r.update_state(
                activity=float(avg_act),
                output=float(avg_act * 0.5),
                energy_input=net.heart.energy_port.value * 0.1,
                env_signal=net.heart.pulse_port.value,
                net=net,
            )

        # v1.4: la inhibición lateral es para población biológica competitiva.
        # No se retira del mapa regional; simplemente no usa infraestructura
        # cognitiva como combustible de dominancia/extinción.
        for r in active_list:
            if r.activity > 0.1:
                r.apply_lateral_inhibition(active_list)

        if self.iteration % 100 == 0:
            self._process_competition(net)

    def _process_competition(self, net):
        active_regions = [r for r in net.regions.values() if r.is_active]
        if len(active_regions) <= 2:
            return
        competitive = self._competitive_mask(net)
        region_ids = np.asarray(net.neuron_region[:net.n], dtype=int)
        self._last_competitive_counts = {}
        for r in active_regions:
            self._last_competitive_counts[r.id] = int(np.sum(competitive & (region_ids == r.id)))

        for r in active_regions:
            neuron_count = self._last_competitive_counts.get(r.id, 0)
            if r.fitness < self.extinction_threshold or neuron_count < 5:
                # Nunca extinguir una región únicamente por falta de neuronas
                # competitivas si contiene infraestructura cognitiva protegida.
                protected_here = bool(np.any((region_ids == r.id) & ~competitive))
                if protected_here and neuron_count < 5:
                    continue
                self._extinguish_region(net, r.id)

    def _extinguish_region(self, net, r_id):
        if r_id not in net.regions:
            return
        region_ids = np.asarray(net.neuron_region)
        for i in range(min(net.n, len(region_ids))):
            if int(region_ids[i]) != int(r_id):
                continue
            if hasattr(net, 'is_noncompetitive') and net.is_noncompetitive(i):
                continue
            new_reg_id = r_id
            if hasattr(net, '_assign_region_by_affinity'):
                try:
                    new_reg_id = net._assign_region_by_affinity(i)
                except Exception:
                    pass
            net.neuron_region[i] = int(new_reg_id)
