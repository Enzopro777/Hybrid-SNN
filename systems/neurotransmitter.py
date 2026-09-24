# systems/neurotransmitter.py

import random
from config import VESICLE_RELEASE_PROB

class NeurotransmitterSystem:
    def modulate(self, net, i, j, signal):
        if net.vesicles[j] <= 0:
            return 0.0

        # Acetilcolina aumenta la probabilidad de liberación (atención)
        ach_factor = 0.6 + 0.8 * net.regions[net.neuron_region[j]].acetylcholine

        release_prob = VESICLE_RELEASE_PROB * ach_factor
        if random.random() > release_prob:
            return 0.0

        net.vesicles[j] -= 1

        affinity = sum(a * b for a, b in zip(net.nt_vector[j], net.receptors[i]))
        calcium_factor = 0.5 + net.calcium[j]

        return signal * (0.5 + affinity) * calcium_factor