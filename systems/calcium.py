# systems/calcium.py

from config import CALCIUM_DECAY, GLOBAL_CALCIUM_DECAY

class CalciumSystem:
    def update(self, net):
        total_fired = sum(net.fired)

        for i in range(net.n):
            if hasattr(net, "is_noncompetitive") and net.is_noncompetitive(i):
                continue
            net.calcium[i] *= CALCIUM_DECAY
            if net.fired[i]:
                net.calcium[i] += 0.38

        # Calcio global mucho más dinámico
        activity_ratio = total_fired / max(1, net.n)

        # Recuperación depende de la actividad global
        recovery = 0.018 + activity_ratio * 0.035

        # Serotonina alta ayuda a estabilizar (menos oscilación)
        if hasattr(net.regions[0], 'serotonin'):
            avg_serotonin = sum(r.serotonin for r in net.regions) / len(net.regions)
            recovery *= (1.0 - (avg_serotonin - 0.5) * 0.25)

        net.global_calcium *= GLOBAL_CALCIUM_DECAY
        net.global_calcium = min(1.05, net.global_calcium + recovery)

        # Si hay muy poca actividad, el calcio global se recupera más lento
        if activity_ratio < 0.08:
            net.global_calcium *= 0.92