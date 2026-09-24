# modules/attention.py
import numpy as np
import math
import random
from core.module import Module

class AttentionModule(Module):
    def __init__(self, net=None):
        super().__init__("Atención")
        # Ejes visuales principales
        self.out_x = self.add_output_port("coord_x", strength=1.0, pos=(40.0, 75.0, 35.0))
        self.out_y = self.add_output_port("coord_y", strength=1.0, pos=(110.0, 75.0, 35.0))
        
        # 🔥 SEPARACIÓN ANTAGÓNICA DEL ZOOM (Puertos 9 y 10)
        self.out_zoom_in = self.add_output_port("zoom_in", strength=1.0, pos=(65.0, 75.0, 35.0))
        self.out_zoom_out = self.add_output_port("zoom_out", strength=1.0, pos=(85.0, 75.0, 35.0))
        
        # [X, Y, Radius] -> El radio arranca balanceado en 0.2
        self.detected_coords = [0.5, 0.5, 0.2]
        self.boredom = 0.0  

    def auto_connect(self, net):
        """ 
        Repara la conexión de los 4 puertos de atención si la red SNN no llegó a cablearlos.
        """
        # Ahora el mapeo incluye los dos nuevos actuadores de zoom
        my_ports = [self.out_x, self.out_y, self.out_zoom_in, self.out_zoom_out]
        
        for port in my_ports:
            if len(port.connected_neurons) < 2:
                distances = np.linalg.norm(net.positions - np.array(port.pos), axis=1)
                closest_indices = np.argsort(distances)[:5]
                
                for idx in closest_indices:
                    if idx not in port.connected_neurons:
                        port.connected_neurons.append(idx)
                
                print(f"👁️ [Atención] Puerto {port.name} reconectado a {len(port.connected_neurons)} neuronas.")

    
    def update(self, net, t):
        if not self.active: return
        
        # 1. FILTRADO RÁPIDO Y CONTROL DE ENTRADA
        if len(self.out_x.connected_neurons) == 0:
            return

        all_connected = np.array(self.out_x.connected_neurons)
        last_spikes = net.last_spike[all_connected]
        dts = t - last_spikes
        mask = dts < 250.0
        active_indices = all_connected[mask]
        
        # --- LÓGICA DE ABURRIMIENTO ---
        if len(active_indices) == 0:
            self.boredom += 0.04  
            # Si no hay estímulos, el radio colapsa lentamente por defecto
            self.detected_coords[2] = self.detected_coords[2] * 0.99 + 0.1 * 0.01
        else:
            self.boredom *= 0.92 

        # 2. SELECCIÓN DE LÍDERES (CENTRO DE MASA PARA X / Y)
        if len(active_indices) > 0:
            relevances = net.membrane_potential[active_indices] / (dts[mask] + 1.0)
            top_k = min(5, len(active_indices))
            top_idx_in_relevances = np.argpartition(relevances, -top_k)[-top_k:]
            
            top_neuron_ids = active_indices[top_idx_in_relevances]
            top_relevances = relevances[top_idx_in_relevances]
            top_positions = net.positions[top_neuron_ids][:, :2] / 150.0 

            total_rel = np.sum(top_relevances)
            if total_rel > 0:
                target_xy = np.sum(top_positions * top_relevances[:, np.newaxis], axis=0) / total_rel
                
                # Dinámica de movimiento para coordenadas de mirada (LERP)
                dist_total = np.linalg.norm(target_xy - self.detected_coords[:2])
                lerp_xy = 0.8 if dist_total > 0.1 else 0.3
                
                self.detected_coords[0] = self.detected_coords[0] * (1 - lerp_xy) + target_xy[0] * lerp_xy
                self.detected_coords[1] = self.detected_coords[1] * (1 - lerp_xy) + target_xy[1] * lerp_xy

        # 3. DINÁMICA DEL RADIO (ZOOM IN vs ZOOM OUT)
        # Evaluamos los spikes que caen en las neuronas asignadas a expandir o contraer
        spikes_in = np.sum(net.fired[self.out_zoom_in.connected_neurons]) if len(self.out_zoom_in.connected_neurons) > 0 else 0
        spikes_out = np.sum(net.fired[self.out_zoom_out.connected_neurons]) if len(self.out_zoom_out.connected_neurons) > 0 else 0
        
        # Cada spike neto altera el radio en pasos controlados
        zoom_delta = (spikes_in - spikes_out) * 0.005
        self.detected_coords[2] += zoom_delta

        # --- 4. MICRO-SACADAS Y SALTOS DE EXPLORACIÓN ---
        chance = 0.05 + (self.boredom * 0.2)
        if random.random() < chance:
            if self.boredom > 0.6:
                self.detected_coords[0] = random.uniform(0.1, 0.9)
                self.detected_coords[1] = random.uniform(0.1, 0.9)
                self.boredom *= 0.4
                print(f"👁️ [Atención] Salto explorativo (Boredom={self.boredom:.2f}): {self.detected_coords[:2]}")
            else:
                jitter_range = 0.01 + (self.boredom * 0.05)
                self.detected_coords[0] += random.uniform(-jitter_range, jitter_range)
                self.detected_coords[1] += random.uniform(-jitter_range, jitter_range)

        # Límites mecánicos estrictos de la fóvea en espacio normalizado
        self.detected_coords = np.clip(self.detected_coords, [0.01, 0.01, 0.05], [0.99, 0.99, 0.45])
        
        # 🔥 ¡LO QUE FALTA! Publicar los datos calculados en los puertos de salida
        self.out_x.value = float(self.detected_coords[0])
        self.out_y.value = float(self.detected_coords[1])
        
        # Exponemos el radio actual en el puerto de zoom_in (o donde lo lea tu motor físico)
        self.out_zoom_in.value = float(self.detected_coords[2])
        
        
    
    def get_ports_info(self):
        ports_info = []
        mapping = [
            ("coord_x", self.out_x),
            ("coord_y", self.out_y),
            ("zoom_in", self.out_zoom_in),
            ("zoom_out", self.out_zoom_out)
        ]
    
        for name, port in mapping:
            ports_info.append({
                "name": f"👁️ {name}", 
                "pos": [float(p) for p in port.pos],
                "strength": float(port.strength),
                "connected_neurons": list(port.connected_neurons)
            })
        return ports_info