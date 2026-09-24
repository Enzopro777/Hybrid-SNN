# systems/growth.py
import random
import numpy as np
from config import RADIUS_3D

class GrowthSystem:
    def __init__(self):
        # Umbral alto para que solo las neuronas "exitosas" se reproduzcan
        self.energy_threshold = 0.85 
        self.repro_chance = 0.05      

    def apply(self, net):
        if net.n >= net.max_neurons: return
        
        current_n = net.n 
        
        for i in range(current_n):
            if not net.active[i]: continue
            
            # Solo se reproducen si tienen mucha energía (metabolismo sano)
            if net.energy[i] > self.energy_threshold:
                if random.random() < self.repro_chance:
                    
                    parent_pos = net.positions[i]
                    
                    # --- LÓGICA DE COLONIZACIÓN VERTICAL ---
                    # Si el padre está en la zona visual (abajo), el hijo tiende a subir 
                    # para llevar la información a la capa de atención.
                    if parent_pos[2] < 35.0:
                        var_z = random.uniform(5, 15)  # Impulso hacia arriba
                        var_xy = random.uniform(-5, 5) # Mutación lateral pequeña
                    else:
                        # Si ya está arriba, se expande horizontalmente para crear 
                        # mapas de atención más amplios.
                        var_z = random.uniform(-3, 3)
                        var_xy = random.uniform(-15, 15)

                    new_x = np.clip(parent_pos[0] + var_xy, 0, 150)
                    new_y = np.clip(parent_pos[1] + var_xy, 0, 150)
                    new_z = np.clip(parent_pos[2] + var_z, 0, 100)

                    # 1. Crear la neurona en la red
                    child_idx = net.add_neuron(parent_idx=i)
                    if child_idx is None: continue

                    # 2. Asignar posición física
                    net.positions[child_idx] = [new_x, new_y, new_z]

                    # 3. REPARTO ENERGÉTICO (Taxa de nacimiento)
                    # El padre queda cansado, el hijo nace con energía media.
                    net.energy[child_idx] = 0.5 
                    net.energy[i] *= 0.5 
                    
                    # 4. HERENCIA SINÁPTICA (Vital para que el hilo Spiking lo reconozca)
                    # El hijo "hereda" el historial de actividad para no ser ignorado por Atención
                    if hasattr(net, 'last_spike'):
                        net.last_spike[child_idx] = net.last_spike[i]
                    
                    # --- 5. CONEXIÓN: padre + vecindario real (NUEVO) ---
                    # Antes: una sola sinapsis al padre. Si el padre estaba en
                    # zona muerta, el hijo heredaba el aislamiento completo.
                    # Ahora: mantenemos el vínculo con el padre (linaje) y
                    # además tejemos conexiones normales con vecinos reales,
                    # igual que hace la red al nacer (_init_connections_sparse).
                    net._connect(i, child_idx, weight=0.5)

                    r_id_child = net.neuron_region[child_idx]
                    s_id_child = net.neuron_subregion[child_idx] if hasattr(net, 'neuron_subregion') else -1
                    if r_id_child in net.regions and s_id_child in net.regions[r_id_child].sub_regions:
                        radius = net.regions[r_id_child].sub_regions[s_id_child].current_profile.conn_radius
                    else:
                        radius = RADIUS_3D

                    dist_child = np.linalg.norm(net.positions[:net.n] - net.positions[child_idx], axis=1)
                    vecinos = np.argsort(dist_child)
                    agregadas = 0
                    for j in vecinos:
                        j = int(j)
                        if j == child_idx or j == i or not net.active[j]:
                            continue
                        if dist_child[j] > radius:
                            break
                        net._connect(child_idx, j, random.uniform(0.3, 0.7))
                        agregadas += 1
                        if agregadas >= 6:
                            break

        if random.random() < 0.05: 
             print(f"🌱 Growth: Población estable en {net.n}n. Estratificando capas...")