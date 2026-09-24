# systems/plasticity.py

import math
import random
import numpy as np
from config import LEARNING_RATE, ENERGY_LEARN_COST

class PlasticitySystem:
    def __init__(self):
        self.birth_prob = 0.01  # Subido ligeramente para compensar el filtrado
        self.death_prob = 0.0001
        self.max_weight = 3.0     # Permitimos sinapsis más fuertes para "memorias" claras
        self.min_weight = -1.0
        self.max_connections_per_neuron = 20 # Más "Sparse" para obligar a la especialización

    def apply(self, net, t=None):
        """
        Aplica aprendizaje Hebbiano (STDP) y poda estructural.
        Optimizado para la detección de patrones visuales.
        """
        # 1. --- STDP DINÁMICO (Aprendizaje basado en picos) ---
        fired_indices = [i for i, f in enumerate(net.fired) if f == 1]
    
        for i in fired_indices:
            if not net.active[i]: continue
            if hasattr(net, 'is_noncompetitive') and net.is_noncompetitive(i):
                continue
            
            r_i = net.neuron_region[i]           # FIX: era net.region[i]
            region_i = net.regions[r_i]

            # Filtro metabólico: Si la región no tiene energía, no hay plasticidad
            if region_i.energy < ENERGY_LEARN_COST:
                continue

            # Revisamos conexiones de salida para fortalecer/debilitar
            for j in net.connections[i]:
                if not net.active[j]: continue

                dt = net.last_spike[j] - net.last_spike[i]
                
                if 0 < dt < 40.0: 
                    dw = LEARNING_RATE * math.exp(-dt / 15.0) * 2.0
                elif dt < 0:
                    dw = -LEARNING_RATE * math.exp(dt / 30.0) * 0.8
                else:
                    dw = LEARNING_RATE * 0.1

                mod = (0.5 + region_i.dopamine * 2.0)
                dw *= mod
            
                old_w = net.weights.get((i, j), 0.1)
                net.weights[(i, j)] = np.clip(old_w + dw, self.min_weight, self.max_weight)
            
                if hasattr(region_i, "adjust_energy"):
                    region_i.adjust_energy(-ENERGY_LEARN_COST * 0.02)
                else:
                    region_i.energy = max(0.1, min(2.5, float(region_i.energy) - ENERGY_LEARN_COST * 0.02))

        # 2. --- PLASTICIDAD ESTRUCTURAL (Crecimiento Dirigido) ---
        active_indices = [i for i, a in enumerate(net.active) if a]
    
        for i in random.sample(active_indices, min(len(active_indices), 100)):
            if hasattr(net, 'is_noncompetitive') and net.is_noncompetitive(i):
                continue
            num_conn = len(net.connections[i])
        
            if num_conn < self.max_connections_per_neuron:
                # FIX: era net.regions[net.region[i]].dopamine
                if random.random() < self.birth_prob * (1 + net.regions[net.neuron_region[i]].dopamine):
                    j_new = self._find_target_neuron(i, net, active_indices)
                    
                    if j_new is not None and j_new not in net.connections[i]:
                        net.connections[i].append(j_new)
                        net.weights[(i, j_new)] = 0.15 

            if num_conn > 3 and random.random() < self.death_prob:
                self._prune_connections(i, net)

    
    def _find_target_neuron(self, i, net, active_indices):
        """Busca una neurona candidata con sesgo vertical (hacia arriba)"""
        pos_i = net.positions[i]
        candidates = random.sample(active_indices, min(len(active_indices), 60))
        
        best_j = None
        max_score = -1.0
        
        for j in candidates:
            if i == j: continue
            pos_j = net.positions[j]
            dist = np.linalg.norm(pos_i - pos_j)
            
            if dist < 45.0: # Radio de búsqueda
                # MÉTRICA DE PUNTUACIÓN:
                # Premiamos que la neurona destino esté MÁS ARRIBA que la origen
                vertical_gain = (pos_j[2] - pos_i[2]) * 0.5
                score = (45.0 - dist) + vertical_gain
                
                if score > max_score:
                    max_score = score
                    best_j = j
        return best_j
    
    def _prune_connections(self, i, net):
        """Elimina sinapsis que no transmiten información relevante"""
        for j in list(net.connections[i]):
            w = net.weights.get((i, j), 0)
            # Si el peso es ínfimo, la conexión es un desperdicio de energía
            if abs(w) < 0.02:
                net.connections[i].remove(j)
                net.weights.pop((i, j), None)