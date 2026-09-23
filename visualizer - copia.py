# visualizer.py - Versión v1.6.0 (Port-to-Region Connectivity)
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import matplotlib.colors as mcolors

plt.style.use('dark_background')

def get_region_color(region_id):
    base_colors = list(mcolors.TABLEAU_COLORS.values())
    if region_id < len(base_colors):
        return base_colors[region_id]
    h = (region_id * 0.618033988749895) % 1.0
    return mcolors.hsv_to_rgb((h, 0.85, 0.95))

def visualize_macro_with_ports(file_path="simulation_state.pkl"):
    print(f"--- Visualizador de Arquitectura y Puertos v1.6.0 ---")
    
    if not os.path.exists(file_path):
        print(f"❌ ERROR: El archivo '{file_path}' no existe.")
        return

    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            net = data.get('network') or data.get('net')
            ports = data.get('ports', [])
    except Exception as e:
        print(f"❌ ERROR al leer pickle: {e}")
        return

    fig = plt.figure(figsize=(14, 10), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')

    # === PROCESAMIENTO DE DATOS ===
    positions = np.array(net.positions)
    active_mask = np.array(net.active)
    
    # Parche de seguridad para el mapeo de regiones
    raw_regions = np.array(net.regions)
    if raw_regions.size != active_mask.size:
        neuron_region_map = getattr(net, 'neuron_regions', (positions[:, 2] // 20).astype(int))
    else:
        neuron_region_map = raw_regions

    # === DIBUJAR REGIONES ===
    unique_regions = np.unique(neuron_region_map[active_mask])
    region_centers = {}
    
    for r_id in unique_regions:
        mask = (neuron_region_map == r_id) & active_mask
        if not np.any(mask): continue
        
        center = positions[mask].mean(axis=0)
        region_centers[r_id] = center
        count = np.sum(mask)

        # Dibujar esfera de región
        color = get_region_color(r_id)
        ax.scatter(center[0], center[1], center[2], s=count*1.5, color=color, alpha=0.6, edgecolors='white')
        ax.text(center[0], center[1], center[2] + 12, f"R{r_id}", color='white', weight='bold', ha='center')

    # === CONEXIONES A PUERTOS (Sensores/Actores) ===
    if ports:
        print(f"🔌 Mapeando {len(ports)} puertos...")
        for p in ports:
            p_pos = np.array(p.get('pos'))
            p_name = p.get('name', 'Port')
            target_neurons = p.get('neurons', [])

            if p_pos is not None:
                # --- NUEVA LÓGICA DE COLOR ---
                p_name_lower = p_name.lower()
                
                if "atención" in p_name_lower or "zoom" in p_name_lower or "🎯" in p_name_lower:
                    p_col = 'lime'
                elif "corazón" in p_name_lower or "stress" in p_name_lower or "energy" in p_name_lower or "pulse" in p_name_lower or "❤️" in p_name_lower:
                    p_col = 'magenta' # El magenta resalta mejor que el rojo puro en fondo negro
                elif "retina" in p_name_lower or "👁️" in p_name_lower:
                    p_col = 'cyan'
                else:
                    p_col = 'white'
                
                # Dibujar el puerto (estrella)
                ax.scatter(p_pos[0], p_pos[1], p_pos[2], c=p_col, s=300, marker='*', edgecolors='white', zorder=10)
                
                # Limpiar nombre para evitar errores de fuente (emojis)
                display_name = p_name.replace('❤️', 'H').replace('👁️', 'V').replace('🎯', 'A')
                ax.text(p_pos[0], p_pos[1], p_pos[2] - 12, display_name, color=p_col, fontsize=10, ha='center', weight='bold')

                # --- TRAZADO DE CABLES ---
                if len(target_neurons) > 0:
                    # Convertir a lista de enteros por seguridad
                    target_neurons = [int(idx) for idx in target_neurons]
                    
                    # Filtramos solo neuronas válidas
                    valid_neurons = [idx for idx in target_neurons if idx < len(neuron_region_map)]
                    
                    if valid_neurons:
                        # Identificamos las regiones conectadas
                        connected_regions = np.unique(neuron_region_map[valid_neurons])
                        
                        for r_id in connected_regions:
                            if r_id in region_centers:
                                r_pos = region_centers[r_id]
                                # Dibujamos el nervio con un degradado de transparencia (alpha)
                                ax.plot([p_pos[0], r_pos[0]], 
                                        [p_pos[1], r_pos[1]], 
                                        [p_pos[2], r_pos[2]], 
                                        color=p_col, alpha=0.6, linewidth=2.5, 
                                        zorder=5, solid_capstyle='round')
    # === CONECTOMA INTER-REGIONAL (Líneas tenues) ===
    # (Opcional: puedes mantenerlo del código anterior para ver flujo interno)

    # Ajustes de visualización
    ax.set_xlim(0, 150); ax.set_ylim(0, 150); ax.set_zlim(-20, 100)
    ax.view_init(elev=20, azim=45)
    ax.axis('off')
    
    plt.title(f"SNN v1.1.3 | Mapa de Puertos e Interfaces Biológicas", color='white', size=15)
    print("🚀 Visualización de puertos lista.")
    plt.show()

if __name__ == "__main__":
    visualize_macro_with_ports()