#visulizer.py

import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import RadioButtons
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import os
import matplotlib.colors as mcolors

plt.style.use('dark_background')

class SNNVisualizer:
    def __init__(self, file_path="simulation_state.pkl"):
        self.file_path = file_path
        self.data = self._load_data()
        if self.data is None: return
        
        self.fig = plt.figure(figsize=(14, 9), facecolor='black')
        self.ax = self.fig.add_subplot(111, projection='3d', facecolor='black')
        
        # Panel de control
        plt.subplots_adjust(left=0.15) 
        rax = plt.axes([0.02, 0.7, 0.1, 0.15], facecolor='#111111')
        self.radio = RadioButtons(rax, ('Macro', 'Micro'), active=0, activecolor='cyan')
        self.radio.on_clicked(self.update_mode)

        self.draw('Macro')

    def _load_data(self):
        if not os.path.exists(self.file_path):
            print(f"❌ ERROR: {self.file_path} no encontrado.")
            return None
        try:
            with open(self.file_path, "rb") as f:
                data = pickle.load(f)
                net = data.get('network') or data.get('net')
                if net:
                    net.positions = np.array(net.positions)
                    net.active = np.array(net.active)
                    # Asegurar que regions sea un array usable
                    if not hasattr(net, 'regions') or np.array(net.regions).ndim == 0:
                        net.regions = (net.positions[:, 2] // 20).astype(int)
                    else:
                        net.regions = np.array(net.regions)
                return data
        except Exception as e:
            print(f"❌ Error al cargar: {e}")
            return None

    def draw(self, mode):
        self.ax.clear()
        net = self.data.get('network') or self.data.get('net')
        ports = self.data.get('ports', [])
        pos, active, reg_map = net.positions, net.active, net.regions

        # Alineación de dimensiones
        if active.shape[0] != reg_map.shape[0]:
            reg_map = np.resize(reg_map, active.shape[0])

        if mode == 'Macro':
            self._draw_macro(pos, active, reg_map, ports)
        else:
            self._draw_micro(pos, active, reg_map, net, ports)

        self.ax.set_title(f"SNN Explorer | {mode} View", color='white', size=14, pad=20)
        self.ax.axis('off')
        
        # Mantener una vista consistente al cambiar de modo
        self.ax.view_init(elev=20, azim=45)
        plt.draw()

    def _draw_macro(self, pos, active, reg_map, ports):
        """Vista de centros de masa y conexiones lógicas."""
        unique_regs = np.unique(reg_map[active])
        centers = {}
        
        # 1. Dibujar esferas de promedio por región
        for r_id in unique_regs:
            mask = (reg_map == r_id) & active
            if not np.any(mask): continue
            center = pos[mask].mean(axis=0)
            centers[r_id] = center
            count = np.sum(mask)
            col = self.get_col(r_id)
            
            self.ax.scatter(*center, s=count*2.5, color=col, alpha=0.6, edgecolors='white', linewidth=0.5)
            self.ax.text(center[0], center[1], center[2] + 8, f"R{r_id}", color='white', fontsize=9, ha='center')

        # 2. Dibujar puertos y sus conexiones a los promedios regionales
        self._draw_ports_and_cables(ports, centers, reg_map, is_micro=False)

    def _draw_micro(self, pos, active, reg_map, net, ports):
        """Vista de neuronas, sinapsis y puertos."""
        # 1. Dibujar todas las neuronas activas (puntos pequeños)
        cols = np.array([self.get_col(reg_map[i]) for i in range(len(pos))])
        self.ax.scatter(pos[active,0], pos[active,1], pos[active,2], c=cols[active], s=8, alpha=0.4, edgecolors='none')
        
        # 2. Dibujar Sinapsis Fuertes (Optimizado)
        if hasattr(net, 'synapses'):
            # Umbral de 0.7 para mantener fluidez en i5/8GB
            lines = [[pos[i], pos[j]] for i, j, w in net.synapses if abs(w) > 0.7 and active[i] and active[j]]
            if lines:
                lc = Line3DCollection(lines, colors='white', linewidths=0.2, alpha=0.15)
                self.ax.add_collection3d(lc)

        # 3. Dibujar puertos (sin cables para no obstruir la vista micro)
        self._draw_ports_and_cables(ports, {}, reg_map, is_micro=True)

    def _draw_ports_and_cables(self, ports, centers, reg_map, is_micro=False):
        """Lógica unificada para dibujar estrellas de puertos."""
        for p in ports:
            p_pos = np.array(p.get('pos'))
            if p_pos is None: continue
            
            p_name = p.get('name', 'Port').lower()
            # Asignación de colores por tipo de puerto
            if any(x in p_name for x in ['retina', '👁️', 'v']): p_col = 'cyan'
            elif any(x in p_name for x in ['atención', '🎯', 'a']): p_col = 'lime'
            elif any(x in p_name for x in ['corazón', '❤️', 'h', 'pulse']): p_col = 'magenta'
            else: p_col = 'white'

            # Dibujar Estrella
            self.ax.scatter(*p_pos, c=p_col, s=350, marker='*', edgecolors='white', zorder=10)
            
            # Etiqueta limpia (sin emojis para evitar errores de fuente)
            clean_name = p.get('name', 'Port').replace('❤️','H').replace('👁️','V').replace('🎯','A')
            self.ax.text(p_pos[0], p_pos[1], p_pos[2]-10, clean_name, color=p_col, fontsize=10, weight='bold', ha='center')

            # En modo Macro, dibujar los "nervios" hacia los promedios regionales
            if not is_micro and centers:
                target_neurons = p.get('neurons', [])
                if len(target_neurons) > 0:
                    connected_regs = np.unique(reg_map[[int(i) for i in target_neurons if int(i) < len(reg_map)]])
                    for r_id in connected_regs:
                        if r_id in centers:
                            r_pos = centers[r_id]
                            self.ax.plot([p_pos[0], r_pos[0]], [p_pos[1], r_pos[1]], [p_pos[2], r_pos[2]], 
                                         color=p_col, alpha=0.4, lw=1.5, linestyle='--')

    def get_col(self, r_id):
        base_cols = list(mcolors.TABLEAU_COLORS.values())
        return base_cols[r_id % len(base_cols)]

    def update_mode(self, label):
        self.draw(label)

if __name__ == "__main__":
    viz = SNNVisualizer()
    if hasattr(viz, 'fig'):
        plt.show()