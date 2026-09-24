from config import TARGET_ACTIVITY, HOMEOSTASIS_RATE
import numpy as np

class HomeostasisSystem:
    def apply(self, net):
        """Homeostasis compatible con la red modular.

        La infraestructura no competitiva queda fuera; para la red biológica
        se ajusta v_thresh de forma suave como proxy de excitabilidad.
        """
        n = int(getattr(net, 'n', 0))
        if n <= 0:
            return
        active = np.asarray(net.active[:n], dtype=bool)
        if not np.any(active):
            return
        fired = np.asarray(net.fired[:n], dtype=np.float32)
        region_ids = np.asarray(getattr(net, 'neuron_region', np.zeros(n, dtype=int))[:n], dtype=int)
        protected = np.zeros(n, dtype=bool)
        for idx in getattr(net, 'noncompetitive_neurons', set()):
            if 0 <= int(idx) < n:
                protected[int(idx)] = True
        mask = active & ~protected
        if not np.any(mask):
            return
        for rid in np.unique(region_ids[mask]):
            region_mask = mask & (region_ids == rid)
            if not np.any(region_mask):
                continue
            region = getattr(net, 'regions', {}).get(int(rid))
            dopamine = float(getattr(region, 'dopamine', 0.0)) if region is not None else 0.0
            ach = float(getattr(region, 'acetylcholine', 0.0)) if region is not None else 0.0
            target = float(np.clip(TARGET_ACTIVITY + dopamine * 0.03, 0.005, 0.25))
            speed = HOMEOSTASIS_RATE * (0.5 + ach)
            silent = region_mask & (fired <= 0)
            active_fire = region_mask & (fired > 0)
            if np.any(silent):
                net.v_thresh[silent] = np.maximum(2.0, net.v_thresh[silent] - speed * target * 0.8)
            if np.any(active_fire):
                net.v_thresh[active_fire] = np.minimum(35.0, net.v_thresh[active_fire] + speed * (1.0 - target) * 0.12)
