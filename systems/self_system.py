import numpy as np

class SelfSystem:
    def update(self, net):
        if net.n == 0:
            return
        active = np.asarray(net.active[:net.n], dtype=bool)
        if not np.any(active):
            return
        energy = np.asarray(net.energy[:net.n], dtype=np.float32)
        fired = np.asarray(net.fired[:net.n], dtype=np.float32)
        potential = np.asarray(net.membrane_potential[:net.n], dtype=np.float32)
        net.self_state["energy"] = float(np.mean(energy[active]))
        net.self_state["activity"] = float(np.mean(fired[active]))
        net.self_state["stress"] = float(np.mean(np.abs(potential[active])))
        net.self_state["coherence"] = float(1.0 - min(1.0, np.var(fired[active])))
