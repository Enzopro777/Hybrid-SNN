
#systems/neuromodulation.py



class NeuromodulationSystem:
    def __init__(self):
        self.global_dopamine = 0.5
        self.global_serotonin = 0.55
        self.global_acetylcholine = 0.65
        self.basal_dopamine = 0.15 # Subimos un poco el humor base

    def update(self, net):
        if net.n == 0: return

        # --- ACTIVIDAD RELATIVA ---
        spikes = sum(net.fired)
        # Sensibilidad ajustada: 3% de la red ya se considera mucha actividad
        sensibilidad = max(1, net.n * 0.03) 
        activity_signal = min(1.0, spikes / sensibilidad)

        # --- DOPAMINA DINÁMICA ---
        # Si hay actividad, sube fuerte. Si no, tiende a basal_dopamine
        if activity_signal > 0:
            # Reacciona más rápido a los spikes
            self.global_dopamine = (self.global_dopamine * 0.85) + (activity_signal * 0.25)
        else:
            # Recuperación pasiva hacia el humor base
            self.global_dopamine = (self.global_dopamine * 0.98) + (self.basal_dopamine * 0.02)
        
        # --- SEROTONINA (Resiliencia) ---
        # Premiamos que la red no esté colapsada
        poblacion_saludable = 1.0 if net.n > 500 else 0.5
        self.global_serotonin = 0.99 * self.global_serotonin + 0.01 * poblacion_saludable

        # --- ACETILCOLINA (Enfoque) ---
        # Sube con la actividad, esencial para la plasticidad
        self.global_acetylcholine = 0.92 * self.global_acetylcholine + 0.08 * activity_signal
        
        if self.global_acetylcholine < 0.45:
            self.global_acetylcholine += 0.01

        # --- PROPAGACIÓN A LAS REGIONES ---
        for region in net.regions:
            # Transición suave local-global
            region.dopamine = (region.dopamine * 0.7) + (self.global_dopamine * 0.3)
            region.serotonin = (region.serotonin * 0.7) + (self.global_serotonin * 0.3)
            region.acetylcholine = (region.acetylcholine * 0.7) + (self.global_acetylcholine * 0.3)

            # --- CLAMPS PROTECTORES ---
            # Evitamos el -0.09 clavado. El mínimo ahora es -0.2 (tristeza leve)
            region.dopamine = max(-0.20, min(1.0, region.dopamine))
            region.serotonin = max(0.40, min(1.0, region.serotonin))
            region.acetylcholine = max(0.40, min(1.0, region.acetylcholine))