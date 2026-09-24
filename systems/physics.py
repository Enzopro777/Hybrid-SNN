# systems/physics.py

import math
import random
from config import GRID_SIZE, GRID_DEPTH

class PhysicsSystem:
    def __init__(self):
        self.attraction_force = 0.02  # Fuerza de cohesión regional
        self.repulsion_force = 0.05   # Fuerza para evitar colisiones
        self.min_dist = 1.5           # Distancia mínima saludable
        self.update_rate = 5          # Se ejecuta cada 5 pasos para ahorrar CPU

    def apply_forces(self, net, iteration):
        if iteration % self.update_rate != 0:
            return

        for i in range(net.n):
            if not net.active[i]: continue
            if hasattr(net, "is_noncompetitive") and net.is_noncompetitive(i):
                continue
            
            # Buscamos una neurona vecina aleatoria para interactuar
            # Esto es más eficiente que comparar todas contra todas
            j = random.randint(0, net.n - 1)
            if i == j or not net.active[j]: continue

            dist = net.distance(i, j)
            if dist == 0: dist = 0.1
            
            # Dirección del vector entre neuronas
            dx = (net.positions[j][0] - net.positions[i][0]) / dist
            dy = (net.positions[j][1] - net.positions[i][1]) / dist
            dz = (net.positions[j][2] - net.positions[i][2]) / dist

            move_x, move_y, move_z = 0, 0, 0

            # --- 1. ATRACCIÓN POR AFINIDAD (Misma Región) ---
            if net.neuron_region[i] == net.neuron_region[j]:
                # Se acercan si están en la misma región
                move_x += dx * self.attraction_force
                move_y += dy * self.attraction_force
                move_z += dz * self.attraction_force
            
            # --- 2. REPULSIÓN POR CONTACTO (Cualquier Región) ---
            if dist < self.min_dist:
                # Se alejan si están "pisándose"
                move_x -= dx * self.repulsion_force
                move_y -= dy * self.repulsion_force
                move_z -= dz * self.repulsion_force

            # Aplicar movimiento al ADN y posición
            new_x = max(0, min(GRID_SIZE-1, net.dna[i][0] + move_x))
            new_y = max(0, min(GRID_SIZE-1, net.dna[i][1] + move_y))
            new_z = max(0, min(GRID_DEPTH-1, net.dna[i][2] + move_z))

            net.dna[i][0], net.dna[i][1], net.dna[i][2] = new_x, new_y, new_z
            
            # 🔥 CORRECCIÓN: Forzar formato de lista mutable para evitar bloqueos/errores de tipo en sub-hilos
            net.positions[i] = [new_x, new_y, new_z]