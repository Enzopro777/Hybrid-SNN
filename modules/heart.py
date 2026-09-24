# modules/heart.py
# v3.2.1 — Fix: net.get_voltage() reemplazado por net.membrane_potential[idx].
#
# CAMBIO ARQUITECTURAL (IA 1.4.1):
#   La red ahora usa bloques con neuronas inmortales que gestionan su
#   propia energía localmente. El corazón deja de ser proveedor de energía
#   y pasa a ser exclusivamente un metrónomo de ritmo base.
#
#   Eliminado: bradicardia por energía baja (freq *= 0.6 cuando avg_energy < 0.3).
#              Esa lógica creaba un círculo vicioso: poca energía → corazón lento
#              → menos spikes → menos energía.
#   Agregado:  auto_connect() reconecta SOLO neuronas con energy > 0.5 para
#              evitar inyectar en neuronas muertas.
#              Piso de energía en marcapasos: garantiza que las neuronas del
#              pulse_port no caigan por debajo de HEART_ENERGY_FLOOR.

from core.module import Module
import random
import numpy as np

HEART_ENERGY_FLOOR = 0.6   # piso mínimo para neuronas marcapasos

class HeartModule(Module):
    def __init__(self):
        super().__init__("Corazón")

        self.stress_port = self.add_input_port("stress_input",
                                               strength=0.85,
                                               pos=(60.0, 75.0, 60.0))
        self.energy_port = self.add_input_port("energy_input",
                                               strength=0.70,
                                               pos=(90.0, 75.0, 60.0))
        self.pulse_port  = self.add_output_port("pulse_output",
                                                strength=1.2,
                                                pos=(75.0, 75.0, 60.0))

        self.pulse_port.value   = 0.0
        self.base_frequency     = 1.2   # Hz — metrónomo base
        self.last_pulse_time    = 0.0

        print("❤️ Corazón v3.2: metrónomo global puro (sin bradicardia energética).")

    # ------------------------------------------------------------------
    # MANTENER ENERGÍA DE MARCAPASOS
    # ------------------------------------------------------------------
    def _enforce_pacemaker_energy(self, net):
        """Garantiza que las neuronas marcapasos tengan energía mínima."""
        if not self.pulse_port.connected_neurons:
            return
        with net.lock:
            for idx in self.pulse_port.connected_neurons:
                if 0 <= idx < net.n and net.active[idx]:
                    if net.energy[idx] < HEART_ENERGY_FLOOR:
                        net.energy[idx] = HEART_ENERGY_FLOOR

    # ------------------------------------------------------------------
    # UPDATE — llamar desde worker_slow cada ciclo bio
    # ------------------------------------------------------------------
    def update(self, net, t):
        if not self.active:
            return

        # 1. Mantener energía en marcapasos (nuevo en v3.2)
        self._enforce_pacemaker_energy(net)

        # 2. Leer estrés (sigue siendo útil para acelerar en situaciones de alerta)
        stress = 0.0
        if self.stress_port.connected_neurons:
            for idx in self.stress_port.connected_neurons:
                if 0 <= idx < net.n:
                    stress += net.membrane_potential[idx] * self.stress_port.strength
            avg_stress = stress / len(self.stress_port.connected_neurons)
        else:
            avg_stress = 0.0

        # 3. Frecuencia: solo estrés la modifica, nunca la energía
        #    (la energía la gestionan los bloques localmente)
        freq = self.base_frequency * (1.0 + avg_stress * 1.8)
        freq = max(0.8, min(4.5, freq))   # piso subido de 0.4 → 0.8

        # Decaimiento del valor del puerto
        self.pulse_port.value *= 0.7

        # 4. Disparo del latido
        if t - self.last_pulse_time >= (1000.0 / freq):
            self.last_pulse_time = t
            self.pulse_port.value = 1.0

            # Strength fija: ya no hay "modo crítico" porque los bloques
            # gestionan su propia energía
            strength = self.pulse_port.strength * 1.5

            injected = 0
            # FIX 1.5.6: no latir durante ventana causal de un trial limpio
            # (la red ya tiene actividad visual suficiente; el latido solo satura la queue)
            _trial_active = getattr(net, '_causal_trial_active', False)
            _queue_ok = len(net.event_queue) < 18000  # umbral más conservador
            if _queue_ok and not _trial_active:
                for idx in self.pulse_port.connected_neurons:
                    if 0 <= idx < net.n and net.active[idx]:
                        jitter = random.uniform(0.0, 5.0)
                        net.receive_spike(idx, strength, t + jitter)
                        injected += 1

            if random.random() < 0.05:
                print(f"❤️ Latido: {freq:.2f}Hz | Spikes: {injected}", flush=True)

            # Reconectar de emergencia si no hay marcapasos
            if not self.pulse_port.connected_neurons and net is not None:
                print("❤️ [Corazón] Sin marcapasos — reconectando de emergencia", flush=True)
                if hasattr(self, 'auto_connect'):
                    self.auto_connect(net)

    def get_ports_info(self):
        ports_info = []
        mapping = [
            (self.stress_port, '❤️ Estrés (IN)'),
            (self.energy_port, '❤️ Energía (IN)'),
            (self.pulse_port,  '❤️ Pulso (OUT)'),
        ]
        for port_obj, label in mapping:
            if port_obj:
                ports_info.append({
                    'name':     label,
                    'pos':      list(getattr(port_obj, 'pos', [75, 75, 60])),
                    'strength': getattr(port_obj, 'strength', 1.0),
                })
        return ports_info

    def auto_connect(self, net):
        """
        Conexión por proximidad — igual que v3.1 pero filtra neuronas
        con energía muy baja para no desperdiciar marcapasos en neuronas
        prácticamente muertas.
        """
        influence_radius = 25.0

        self.stress_port.connected_neurons = []
        self.energy_port.connected_neurons = []
        self.pulse_port.connected_neurons  = []

        p_stress = np.array(self.stress_port.pos)
        p_energy = np.array(self.energy_port.pos)
        p_pulse  = np.array(self.pulse_port.pos)

        for i in range(net.n):
            if not net.active[i]:
                continue
            pos_i = net.positions[i]

            if np.linalg.norm(pos_i - p_stress) < influence_radius:
                self.stress_port.connected_neurons.append(i)

            if np.linalg.norm(pos_i - p_energy) < influence_radius:
                self.energy_port.connected_neurons.append(i)

            # Pulse: solo conectar neuronas con energía mínima viable
            if (np.linalg.norm(pos_i - p_pulse) < (influence_radius * 0.8)
                    and net.energy[i] >= HEART_ENERGY_FLOOR):
                self.pulse_port.connected_neurons.append(i)

        print(f"❤️ Corazón Auto-Conectado: {len(self.pulse_port.connected_neurons)} marcapasos viables.")
