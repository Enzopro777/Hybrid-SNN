# systems/inhibition.py
# Sistema de Inhibición GABA - Equilibrio E/I

import random
import numpy as np

class InhibitionSystem:
    def __init__(self):
        self.base_gaba = 0.22                    # fuerza base de inhibición
        self.gaba_decay = 0.915                  # decaimiento de sensibilidad

    def apply(self, net):
        """
        Aplica inhibición GABA de forma realista:
        - Más fuerte cuando hay alta actividad (evita sobre-excitación)
        - Modulada por serotonina (estabilidad)
        - Afecta tanto el potencial como la excitabilidad
        """
        for i in range(net.n):
            if not net.active[i]:
                continue

            if hasattr(net, 'is_noncompetitive') and net.is_noncompetitive(i):
                continue
            region = net.regions[net.neuron_region[i]]

            # Fuerza de GABA depende de:
            # - Serotonina alta = más inhibición (estabilidad)
            # - Actividad reciente = más inhibición (control de picos)
            gaba_strength = self.base_gaba

            gaba_strength += (region.serotonin - 0.5) * 0.25      # serotonina aumenta inhibición
            gaba_strength += net.fired[i] * 0.18                  # más fuerte después de spike

            # Sensibilidad individual de la neurona
            gaba_strength *= net.gaba_sensitivity[i]

            # Aplicar inhibición al potencial canónico.
            net.membrane_potential[i] -= gaba_strength * 0.85

            # Proxy modular de excitabilidad: umbral de disparo.
            factor = (0.97 + region.serotonin * 0.03)
            net.v_thresh[i] = float(np.clip(net.v_thresh[i] / max(factor, 1e-6), 2.0, 35.0))

        # Decaimiento lento de sensibilidad GABA (adaptación)
        for i in range(net.n):
            if net.active[i]:
                net.gaba_sensitivity[i] *= self.gaba_decay
                net.gaba_sensitivity[i] = max(0.35, net.gaba_sensitivity[i])