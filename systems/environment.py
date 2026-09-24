# =============================
# systems/environment.py
# =============================

import random
import math
from config import GRID_SIZE, GRID_DEPTH

class EnvironmentSystem:
    def __init__(self):
        # Definimos 3 módulos en posiciones fijas (puedes cambiarlas o agregar más)
        self.modules = [
            {"pos": (GRID_SIZE//4, GRID_SIZE//4, GRID_DEPTH//2), "strength": 1.2, "radius": 12},
            {"pos": (GRID_SIZE*3//4, GRID_SIZE//4, GRID_DEPTH//2), "strength": 1.0, "radius": 10},
            {"pos": (GRID_SIZE//2, GRID_SIZE*3//4, GRID_DEPTH//3), "strength": 0.9, "radius": 15}
        ]

    def distance_3d(self, x1, y1, z1, x2, y2, z2):
        return math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)

    def update(self, net):
        for i in range(net.n):
            x, y, z = net.positions[i]

            # Suma de señales de todos los módulos (gradiente inverso a distancia)
            total_signal = 0.0
            for mod in self.modules:
                mx, my, mz = mod["pos"]
                dist = self.distance_3d(x, y, z, mx, my, mz)
                if dist < mod["radius"]:
                    # Decaimiento inverso (más fuerte cerca del centro)
                    signal = mod["strength"] / (1 + dist * 0.3)  # 0.3 = decaimiento suave
                    total_signal += signal

            # 🔥 CAMBIO SOLICITADO: Reducir fuertemente el poder del entorno
            net.membrane_potential[i] += total_signal * 0.2          # antes era * 0.8

            # Boost energético mucho más suave (opcional pero recomendado)
            net.energy[i] += total_signal * 0.01            # antes era * 0.02

            # Input externo aleatorio residual (ruido muy bajo)
            if random.random() < 0.005:
                net.potential[i] += 0.5