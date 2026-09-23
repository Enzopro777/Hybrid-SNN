#main.py


import time
import logging
import os
import pickle  # <-- NUEVO: Para inspeccionar el tamaño real de la red guardada

# --- Importaciones del proyecto ---
from config import *
from core.network import NeuralNetwork
from engine.simulation import SimulationEngine
from engine.thread_manager import ThreadManager

try:
    from screen_interface import ScreenProcessor
    HAS_SCREEN = True
except ImportError:
    logging.warning("screen_interface.py no encontrado. Visión desactivada.")
    HAS_SCREEN = False

def main():
    print("-" * 60)
    print("🚀 INICIANDO SIMULACIÓN  SNN  -  (Multi-Threaded)")
    print("🎯 Reward System: ACTIVO (Dopamina por Visión)")
    print("-" * 60)

    # Nombres base coherentes con simulation.py
    filename_base = "simulation_state"
    file_meta = f"{filename_base}_meta.pkl"
    file_np = f"{filename_base}_matrices.npz"

    # ---------------------------------------------------------------------
    # 1. INICIALIZACIÓN INTELIGENTE DE LA RED
    # ---------------------------------------------------------------------
    if os.path.exists(file_meta) and os.path.exists(file_np):
        print(f"💾 [Persistencia] ¡Guardado previo detectado! Reconstruyendo cerebro...")
        try:
            with open(file_meta, "rb") as f:
                data = pickle.load(f)
            n_guardado = data['network_meta'].get('n', 1000)
            
            print(f"🧠 Dimensionando NeuralNetwork a {n_guardado} neuronas...")
            net = NeuralNetwork(n=n_guardado)
            sim = SimulationEngine(net)
            
            if sim.load_state(filename_base):
                print(f"✅ Cerebro restaurado y enlazado. Listo para reanudar.")
            else:
                raise Exception("Fallo en load_state interno.")
                
        except Exception as e:
            logging.error(f"Error en reconstrucción: {e}. Iniciando red limpia.")
            net = NeuralNetwork(n=1000)
            sim = SimulationEngine(net)
    else:
        print("🌱 [Persistencia] Creando red base con 8000 neuronas.")
        net = NeuralNetwork(n=8000)
        sim = SimulationEngine(net)
    
    # --- NUEVO: backfill de sinapsis (una vez por arranque; idempotente) ---
    if hasattr(net, 'backfill_orphan_connections'):
        net.backfill_orphan_connections(min_connections=4)

    # 2. Procesador de pantalla para la entrada visual (si existe)
    scr_processor = ScreenProcessor() if HAS_SCREEN else None

    # 3. Ejecución
    try:
        # sim.run ya inicializa e inicia internamente el ThreadManager
        sim.run(screen_processor=scr_processor)
    except Exception as e:
        logging.error(f"Error crítico en la ejecución: {e}")
    finally:
        print("\n✅ Simulación finalizada correctamente.")
        print(f"📊 Estado final: {net.n} neuronas | Memoria liberada.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(threadName)s] %(message)s')
    main()




# =============================
# === config.py - v1.3.0 (Fase 4: Persistencia) ===
# =============================

# --- PARÁMETROS BASE ---
NUM_NEURONS = 8000
TIME_STEPS = 200
GRID_SIZE = 50
RADIUS = 3
NUM_REGIONS = 5
NUM_NEUROTRANSMITTERS = 10
LEARNING_RATE = 0.05

# --- ECONOMÍA ENERGÉTICA ---
ENERGY_FIRE_COST = 0.05
ENERGY_LEARN_COST = 0.01
ENERGY_MUTATION_COST = 0.05

# --- DINÁMICA SINÁPTICA ---
CALCIUM_DECAY = 0.95
GLOBAL_CALCIUM_DECAY = 0.98
VESICLE_RELEASE_PROB = 0.3

# --- HOMEOSTASIS ---
TARGET_ACTIVITY = 0.05
HOMEOSTASIS_RATE = 0.009

# --- ARQUITECTURA 3D ---
GRID_DEPTH = 18
RADIUS_3D = 45.0

# --- EVOLUCIÓN DE REGIONES ---
REGION_MIN_NEURONS = 50
REGION_REPRODUCE_DOP = 0.50
REGION_REPRODUCE_NEURONS = 180
REGION_EXTINCT_DOP = -0.55
REGION_EXTINCT_NEURONS = 60
REGION_REPRODUCE_PROB = 0.018
REGION_EXTINCT_PROB = 0.012
REGION_FUSION_PROB = 0.008
EVOLUTION_CHECK_EVERY = 60

# --- FASE 1: PARÁMETROS LIF ---
V_REST = 0.0
V_THRESHOLD = 20.0
V_RESET = 0.0
TAU_MEMBRANE = 12.0
REFRACTORY_PERIOD = 2.0
SYNAPTIC_DELAY_BASE = 1.0
MAX_EVENTS_PER_STEP = 10000
SLOW_SYSTEMS_TICK = 50.0
# --- FASE 2: EVENT-DRIVEN ---
MAX_IDLE_MS = 8.0
VISUAL_PERSISTENCE = 45.0
POISSON_NOISE_WINDOW = 120.0
MIN_EVENT_GAP = 0.05
NOISE_FREQ_HZ = 2.5

# --- FASE 3: MULTI-THREADING ---
VISION_FPS_TARGET = 60
SLOW_SYSTEMS_MS = 80.0
MAX_EVENTS_QUEUE = 25000
THREAD_SLEEP_IDLE = 0.0008
VISION_SLEEP = 0.012
EVENT_BATCH_SIZE = 200

# --- FASE 4: DARWINISMO Y COMPETENCIA ---
ENERGY_DECAY_IDLE = 0.005
NEURON_DEATH_THRESHOLD = 0.20
REGION_INHIBITION_STRENGTH = 0.9
REGION_DOMINANCE_THRESHOLD = 0.65
IGNITION_THRESHOLD = 30
IGNITION_BOOST_FACTOR = 4.0
ENERGY_RECOVERY_SPIKE = 0.02
MAX_NEURON_ENERGY = 2.0





# --- VISION RADIAL (Fase 1) ---
FOVEA_THRESHOLD = 0.15    # Radio de visión 20/20
PARACENTRO_THRESHOLD = 0.40
COLOR_SENSITIVITY = 1.1  # Multiplicador de fuerza para spikes de color
MOTION_SENSITIVITY = 2.8  # Más sensibilidad en la periferia para alertas

# =====================================================================
# === PROXY DE COMPATIBILIDAD EVOLUTIVA (Fase 4 - Bridge Directo) ===
# =====================================================================
try:
    from core import evolution_config
    
    # 1. Enrutar la clase (Tu código viejo busca 'Profile', el nuevo usa 'PROFILE')
    Profile = evolution_config.PROFILE
    
    # 2. Reconstruir PROFILES_DICT a partir del nuevo EVO_MENU
    PROFILES_DICT = evolution_config.EVO_MENU
    
    # 3. Reconstruir la lista PROFILES indexando los valores en memoria
    PROFILES = list(evolution_config.EVO_MENU.values())
    
    # 4. Mantener el alias histórico EVO_MENU
    EVO_MENU = PROFILES_DICT

except ImportError as e:
    import logging
    logging.error(f"🚨 Error crítico en el Proxy de config.py: No se pudo enlazar core.evolution_config. {e}")
    # Red de seguridad extrema para evitar que caiga el hilo metabólico
    PROFILES_DICT = {}
    PROFILES = []
    EVO_MENU = {}


# screen_interface.py

import cv2
import numpy as np
from PIL import ImageGrab
import math
import random
import logging
import time
import json
from datetime import datetime

class ScreenProcessor:
    def __init__(self, width=800, height=600, real_screen_mode=False):
        # === CONFIGURACIÓN DE MODO ===
        self.real_screen_mode = real_screen_mode

        # === CONFIGURACIÓN DE PANTALLA VIRTUAL ===
        self.width  = width
        self.height = height
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # === PROCESAMIENTO RETINIANO ===
        self.scale_factor    = 0.15
        self.last_frame_gray = None

        # === HABITUACIÓN Y ESTADO ===
        self.stagnation_counter   = 0.0
        self.last_pos             = [0.5, 0.5]
        self.stagnation_threshold = 95.0
        self.stagnation_decay     = 22.0
        self.boredom_half_life    = 320.0

        print("🔬 [ScreenProcessor] Experimentos A/B retirados — lienzo neutro listo para entrenamiento de letras")

    # ------------------------------------------------------------------
    # GENERACIÓN DE FRAME VIRTUAL CON DOS OBJETOS EN MOVIMIENTO
    # ------------------------------------------------------------------

    def generate_mission_frame(self):
        """Lienzo neutro. Los objetos de los Experimentos A/B ya no se dibujan."""
        self.canvas.fill(0)
        return self.canvas

    
    # ------------------------------------------------------------------
    # MÉTODOS EXISTENTES
    # ------------------------------------------------------------------
    
    def generate_letter_frame(self, letra, pos=(0.5, 0.5), size=6, color=(255,255,255)):
        import numpy as np
        import cv2
        # Asumiendo resolución estándar de 800x600, ajústalo a la tuya si difiere
        frame = np.zeros((600, 800, 3), dtype=np.uint8) 
        px, py = int(pos[0] * 800), int(pos[1] * 600)
        
        cv2.putText(frame, letra, (px-40, py+40), cv2.FONT_HERSHEY_SIMPLEX, size, color, 8)
        return frame

    

    def apply_retinal_mask(self, h, w, focus_pt, sigma=0.22):
        xf, yf  = int(focus_pt[0] * w), int(focus_pt[1] * h)
        y, x    = np.ogrid[:h, :w]
        dist_sq = ((x - xf)**2 + (y - yf)**2) / (max(w, h)**2)
        return np.exp(-dist_sq / (2 * sigma**2))

    def get_activity_coords(self, focus_pt=(0.5, 0.5), focus_radius=0.15,
                            virtual_frame=None):

        if self.real_screen_mode:
            try:
                screen_rgb = np.array(ImageGrab.grab())
                screen_bgr = cv2.cvtColor(screen_rgb, cv2.COLOR_RGB2BGR)
                frame_bgr  = cv2.resize(screen_bgr, None,
                                        fx=self.scale_factor, fy=self.scale_factor)
            except Exception as e:
                logging.error(f"Error capturando pantalla real: {e}")
                return [], 0.0, list(focus_pt)
        else:
            screen    = virtual_frame if virtual_frame is not None \
                        else self.generate_mission_frame()
            frame_bgr = cv2.resize(screen, None,
                                   fx=self.scale_factor, fy=self.scale_factor)

        h, w            = frame_bgr.shape[:2]
        importance_mask = self.apply_retinal_mask(h, w, focus_pt, focus_radius)

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.last_frame_gray is None or \
           self.last_frame_gray.shape != gray.shape:
            self.last_frame_gray = gray
            return [], 0.0, list(focus_pt)

        diff = cv2.absdiff(self.last_frame_gray, gray)
        self.last_frame_gray = gray
        _, activity_thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

        coords_active        = np.column_stack(np.where(activity_thresh > 0))
        normalized_points    = []
        fovea_interest_score = 0.0

        if len(coords_active) > 0:
            step = 2 if len(coords_active) < 1200 else 6
            for p in coords_active[::step]:
                y_idx, x_idx = p[0], p[1]
                weight       = importance_mask[y_idx, x_idx]

                if random.random() < (weight * 0.92 + 0.08):
                    # --- NUEVO: Extraemos la intensidad del movimiento ---
                    intensity = float(diff[y_idx, x_idx]) / 255.0
                    
                    if weight > 0.55:
                        b, g, r = frame_bgr[y_idx, x_idx].astype(np.float32)
                        if abs(r - g) > 35 or abs(b - (r + g) / 2) > 35:
                            fovea_interest_score += 0.12

                    # --- MODIFICADO: Agregamos la intensidad a la tupla ---
                    normalized_points.append((x_idx / w, y_idx / h, 0, intensity))

        peripheral_interest = min(1.8, len(coords_active) / 1800.0)
        base_score          = min(1.4, (fovea_interest_score / 18.0) +
                                       (peripheral_interest * 0.35))

        

        # --- Homeostasis ---
        dist_moved = math.sqrt((focus_pt[0] - self.last_pos[0])**2 +
                               (focus_pt[1] - self.last_pos[1])**2)

        if dist_moved < 0.012:
            self.stagnation_counter += 0.28
        else:
            self.stagnation_counter = max(0.0,
                                          self.stagnation_counter - self.stagnation_decay)

        if self.stagnation_counter > self.stagnation_threshold:
            focus_pt = [
                max(0.12, min(0.88, focus_pt[0] + random.uniform(-0.35, 0.35))),
                max(0.12, min(0.88, focus_pt[1] + random.uniform(-0.35, 0.35)))
            ]
            self.stagnation_counter = 0.0
            boredom_factor          = 1.0
        else:
            boredom_factor = math.exp(-self.stagnation_counter /
                                      self.boredom_half_life)

        self.last_pos = list(focus_pt)
        return normalized_points, base_score * max(0.35, boredom_factor), focus_pt
    
    def get_static_contrast_coords(self, frame_bgr, focus_pt, focus_radius=0.15):
        """
        Canal 'parvo': contraste espacial del frame actual (Formas estáticas).
        Usa Diferencia de Gaussianas (DoG) simulando el campo receptivo centro-periferia.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        
        # Aprovechamos la misma máscara de atención foveal que ya tienes
        importance_mask = self.apply_retinal_mask(h, w, focus_pt, focus_radius)
        
        # Diferencia de Gaussianas (DoG)
        narrow = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
        wide   = cv2.GaussianBlur(gray, (0, 0), sigmaX=3.0)
        dog    = cv2.absdiff(narrow, wide)
        
        # Umbral: el '10' es el que Claude sugiere afinar si captura mucho o poco ruido
        _, contrast_mask = cv2.threshold(dog, 10, 255, cv2.THRESH_BINARY)
        
        coords = np.column_stack(np.where(contrast_mask > 0))
        points = []
        
        if len(coords) > 0:
            step = 2 if len(coords) < 1200 else 6
            for p in coords[::step]:
                y_idx, x_idx = p[0], p[1]
                weight = importance_mask[y_idx, x_idx]
                
                # Probabilidad de disparo basada en si está en el centro de atención
                if random.random() < (weight * 0.92 + 0.08):
                    intensity = float(dog[y_idx, x_idx]) / 255.0
                    
                    # El '1' indica tipo "forma estática", que luego mandarás a z=45
                    points.append((x_idx / w, y_idx / h, 1, intensity))
                    
        return points
    


# =================================================================
# TEST RÁPIDO DE INTENSIDAD (Verificación Claude)
# =================================================================
#if __name__ == "__main__":
#    print("Iniciando test rápido de ScreenProcessor...")
#    sp = ScreenProcessor()
#    
#    # 1. Primer frame (calibra el fondo, no devuelve puntos)
#    f1 = sp.generate_mission_frame()
#    sp.get_activity_coords(virtual_frame=f1)          
#    
#    # 2. Segundo frame (genera movimiento artificial)
#    f2 = sp.generate_mission_frame()                  
#    points, score, _ = sp.get_activity_coords(virtual_frame=f2)
#    
#    # 3. Calcula y muestra las intensidades
#    if points:
#        intens = [p[3] for p in points]
#        print(f"✅ puntos: {len(points)} | intensidad prom: {sum(intens)/max(1,len(intens)):.4f} | máx: {max(intens, default=0):.4f}")
#    else:
#        print("❌ 0 puntos detectados. Algo falló en la generación del frame o no hubo movimiento.")
# =================================================================
# TEST RÁPIDO DE CANAL ESTÁTICO (Parvocelular / DoG)
# =================================================================


#if __name__ == "__main__":
#    print("Iniciando test del Canal Estático...")
#    sp = ScreenProcessor()
    
#    # Generamos un frame (si tienes real_screen_mode=True, capturará tu pantalla)
#    # Si es virtual, usará la imagen de prueba que genera tu código.
#    frame = sp.generate_mission_frame() 
#    focus = (0.5, 0.5)  # Foco en el centro
    
#    # Llamamos al nuevo método
#    puntos = sp.get_static_contrast_coords(frame, focus)
    
#    if puntos:
#        intensidades = [p[3] for p in puntos]
#        tipos = set([p[2] for p in puntos])
#        
#        print(f"✅ Formas estáticas detectadas!")
#        print(f"   Puntos activados: {len(puntos)}")
#        print(f"   Intensidad prom:  {sum(intensidades)/len(intensidades):.4f}")
#        print(f"   Intensidad máx:   {max(intensidades):.4f}")
#        print(f"   Tipo de canal:    {tipos} (Debería decir {{1}})")
#    else:
#        print("❌ 0 puntos detectados. El frame está completamente liso o el umbral es muy alto.")    


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

#core/evolution_config.py


"""
Define los perfiles biológicos y la configuración del motor de mutación.
Optimizado para hardware i5-6th Gen.
"""

class PROFILE:
    def __init__(self, name, tau_m, v_thresh, refr_period, energy_drain, conn_radius):
        """
        Representa un estado funcional de una sub-región neuronal.
        """
        self.name = name
        self.tau_m = tau_m             # Constante de tiempo de membrana (ms)
        self.v_thresh = v_thresh       # Umbral de disparo (mV)
        self.refr_period = refr_period # Tiempo de descanso tras disparo (ms)
        self.energy_drain = energy_drain # Costo metabólico por paso
        self.conn_radius = conn_radius   # Radio físico de conexión sináptica (unidades 3D)

# === DICCIONARIO DE PERFILES EVOLUTIVOS ===
# La IA elegirá entre estos perfiles según el éxito de la región.
EVO_MENU = {
    "RAPID_FIRE": PROFILE(
        name="Rapid Fire",
        tau_m=5.0,           # Reacción instantánea
        v_thresh=16.0,       # 🧠 Subido de 12.0 a 16.0 (Requiere más energía para entrar en bucle)
        refr_period=3.5,     # 🧠 Subido de 1.5 a 3.5 (Obliga a la neurona a descansar más entre disparos)
        energy_drain=0.12,   # Castigo metabólico más alto por abusar del disparo   
        conn_radius=15.0     # Enfoque local denso
    ),
    
    "BUFFER": PROFILE(
        name="Working Memory",
        tau_m=80.0,          # Retiene el voltaje mucho tiempo
        v_thresh=25.0,       # Difícil de activar, requiere persistencia
        refr_period=10.0,    # Lento
        energy_drain=0.01,   # Muy económico
        conn_radius=10.0     # Estructura compacta
    ),
    
    "RELAY": PROFILE(
        name="Long-Range Relay",
        tau_m=20.0,          # Estándar biológico
        v_thresh=18.0,
        refr_period=4.0,
        energy_drain=0.03,
        conn_radius=40.0     # Ideal para conectar Vision con Attention
    ),
    
    "INTEGRATOR": PROFILE(
        name="Noise Filter",
        tau_m=40.0,          # Filtra el ruido acumulando señales
        v_thresh=35.0,       # Solo dispara si hay un patrón claro
        refr_period=8.0,
        energy_drain=0.02,
        conn_radius=12.0
    )
}

# === CONFIGURACIÓN DEL MOTOR DE EVOLUCIÓN ===
EVO_SETTINGS = {
    "mutation_rate": 0.02,       # Probabilidad de intento de mutación por ciclo bio
    "evaluation_steps": 100,     # Cuántos pasos dura la "prueba" de un nuevo perfil
    "energy_min_req": 0.75,      # Energía mínima necesaria para intentar un 'Jump'
    "stress_threshold": 0.6,     # Si el estrés supera esto, se busca un Rollback
    "default_profile_key": "RELAY"
}

# Perfil por defecto para la inicialización de la red
DEFAULT_PROFILE = EVO_MENU[EVO_SETTINGS["default_profile_key"]]

def get_profile_by_name(name):
    """Retorna un perfil del menú por su nombre o el default si no existe."""
    for p in EVO_MENU.values():
        if p.name == name:
            return p
    return DEFAULT_PROFILE

# Alias de compatibilidad para evitar romper el resto del código que busca "Profile"
Profile = PROFILE



# core/module.py
# Sistema de puertos y módulos plug-and-play con soporte para coordenadas 3D

import random

class Port:
    def __init__(self, name, module, is_input=True, strength=1.0, pos=None):
        self.name = name
        self.module = module
        self.is_input = is_input
        self.strength = max(0.1, min(2.0, strength))  # límites razonables
        self.value = 0.0
        self.connected_neurons = []  # lista de índices de neuronas
        
        # --- NUEVO: Posición 3D para el visualizador ---
        # Si no se define, se coloca en el centro por defecto (25, 25, 25)
        self.pos = pos if pos is not None else (25.0, 25.0, 25.0)
    
    def to_dict(self):
        """Estado serializable del puerto — sin locks, sin referencias circulares."""
        return {
            "strength": self.strength,
            "value": float(self.value),
            "connected_neurons": [int(i) for i in self.connected_neurons],
        }

    def load_dict(self, data, max_idx=None):
        self.strength = data.get("strength", self.strength)
        self.value = data.get("value", self.value)
        neuronas = data.get("connected_neurons", [])
        if max_idx is not None:
            neuronas = [i for i in neuronas if 0 <= i < max_idx]
        self.connected_neurons = list(neuronas)

    def get_all_ports(self):
        """Retorna una lista con todos los puertos (In y Out)"""
        return list(self.input_ports.values()) + list(self.output_ports.values())
    
    def connect(self, neuron_idx):
        if neuron_idx not in self.connected_neurons:
            self.connected_neurons.append(neuron_idx)

    def disconnect(self, neuron_idx):
        if neuron_idx in self.connected_neurons:
            self.connected_neurons.remove(neuron_idx)

    def mutate(self):
        """Mutación de la fuerza del puerto (plasticidad estructural)"""
        if random.random() < 0.08:  # ~8% chance por ciclo de mutación
            self.strength += random.uniform(-0.12, 0.12)
            self.strength = max(0.1, min(2.0, self.strength))


class Module:
    def __init__(self, name):
        self.name = name
        self.input_ports = {}
        self.output_ports = {}
        self.active = True

    
    def get_state(self):
        """Estado serializable de TODOS los puertos (in + out). Genérico: cualquier
        módulo nuevo lo hereda gratis, sin escribir lógica de guardado propia."""
        return {
            "input_ports":  {name: p.to_dict() for name, p in self.input_ports.items()},
            "output_ports": {name: p.to_dict() for name, p in self.output_ports.items()},
        }

    def restore_state(self, data, max_idx=None):
        """Aplica un estado guardado sobre los puertos YA CREADOS del módulo actual.
        Si un puerto del guardado ya no existe, se ignora. Si hay uno nuevo, 
        queda con su valor por defecto."""
        for name, pdata in data.get("input_ports", {}).items():
            if name in self.input_ports:
                self.input_ports[name].load_dict(pdata, max_idx=max_idx)
                
        for name, pdata in data.get("output_ports", {}).items():
            if name in self.output_ports:
                self.output_ports[name].load_dict(pdata, max_idx=max_idx)


    def add_input_port(self, name, strength=1.0, pos=None):
        """Añade un puerto de entrada con posición opcional para el visualizador"""
        port = Port(name, self, is_input=True, strength=strength, pos=pos)
        self.input_ports[name] = port
        return port

    def add_output_port(self, name, strength=1.0, pos=None):
        """Añade un puerto de salida con posición opcional para el visualizador"""
        port = Port(name, self, is_input=False, strength=strength, pos=pos)
        self.output_ports[name] = port
        return port

    def update(self, net, t):
        """Cada módulo específico (como HeartModule) implementa su lógica aquí"""
        raise NotImplementedError("Debes implementar el método update en la subclase.")

    def mutate(self):
        """Mutación genérica de todos los puertos del módulo"""
        for port in list(self.input_ports.values()) + list(self.output_ports.values()):
            port.mutate()


# core/network.py
from core.evolution_config import EVO_MENU, DEFAULT_PROFILE

import random
import math
import numpy as np
import threading
import heapq
import logging

from config import (
    GRID_SIZE, GRID_DEPTH, RADIUS_3D, NUM_REGIONS,
    V_REST, V_THRESHOLD, V_RESET, TAU_MEMBRANE, REFRACTORY_PERIOD, 
    SYNAPTIC_DELAY_BASE, IGNITION_BOOST_FACTOR, MAX_EVENTS_QUEUE
)

from systems.genetics import GeneticSystem
from core.region import Region
from core.reward_system import RewardSystem
from modules.heart import HeartModule
from modules.vision import VisionModule


class NeuralNetwork:
    
    def __init__(self, n=1000):
        self.max_neurons = 8000
        self.n = n
        self.lock = threading.RLock()

        # === ESTADOS VECTORIZADOS ===
        self.active = np.zeros(self.max_neurons, dtype=bool)
        self.active[:n] = True

        self.membrane_potential = np.full(self.max_neurons, V_REST, dtype=np.float32)
        self.last_update_time = np.zeros(self.max_neurons, dtype=np.float32)
        self.refractory_until = np.zeros(self.max_neurons, dtype=np.float32)
        self.energy = np.ones(self.max_neurons, dtype=np.float32)

        # === FITNESS NEURONAL (Condición física por neurona) ===
        # Rango: 0.5 (atrofiada por desuso) a 2.0 (entrenada)
        # Sube cuando la neurona dispara con recompensa dopaminérgica
        # Baja cuando la neurona está inactiva demasiado tiempo
        # Determina el techo máximo de energía alcanzable
        self.fitness_neuronal = np.ones(self.max_neurons, dtype=np.float32)

        self.fired = np.zeros(self.max_neurons, dtype=np.int32)
        self.fired_acumulado = np.zeros(self.max_neurons, dtype=np.int32)
        self.last_spike = np.zeros(self.max_neurons, dtype=np.float32)

        # Parámetros biológicos
        self.tau_m = np.full(self.max_neurons, TAU_MEMBRANE, dtype=np.float32)
        self.v_thresh = np.full(self.max_neurons, V_THRESHOLD, dtype=np.float32)
        self.refr_period = np.full(self.max_neurons, REFRACTORY_PERIOD, dtype=np.float32)

        # === QUÍMICA ===
        
        self.nt_vector = np.zeros((self.max_neurons, 5), dtype=np.float32)
        self.receptors = self.nt_vector.copy()          # ← Corrección clave
        
        # Agregar el nivel de dopamina global de la red
        self.dopamine_level = 0.0

        # Posiciones
        self.positions = np.zeros((self.max_neurons, 3), dtype=np.float32)
        for i in range(n):
            self.positions[i] = [random.uniform(0, 150), random.uniform(0, 150), random.uniform(0, 100)]

        self.connections = [[] for _ in range(self.max_neurons)]
        self.dna = [None] * self.max_neurons
        for i in range(n):
            self.dna[i] = self._generate_initial_dna()

        # Regiones
        self.regions = {i: Region(i) for i in range(NUM_REGIONS)}
        self.neuron_region = np.full(self.max_neurons, -1, dtype=np.int32)
        self.neuron_subregion = np.full(self.max_neurons, -1, dtype=np.int32)

        initial_regions = [random.randint(0, NUM_REGIONS - 1) for _ in range(n)]
        self.neuron_region[:n] = initial_regions

        # Sub-regiones iniciales
        self.regions[0].create_sub_region(sub_id=0, profile=EVO_MENU["RAPID_FIRE"])
        self.regions[0].create_sub_region(sub_id=1, profile=EVO_MENU["BUFFER"])

        # Módulos
        self.genetics = GeneticSystem()
        self.heart = HeartModule()
        self.vision = VisionModule(net=self)
        self.reward_manager = RewardSystem()
        
        
        self.modules = {
            "heart": self.heart,
            "vision": self.vision,
            "reward_manager": self.reward_manager
        }

        # Eventos
        self.event_queue = []
        self.weights = {}
        self.current_time = 0.0
        self.ignition_active = False

        # Inicialización
        self._apply_initial_genetics()
        self._assign_initial_subregions()
        self._init_core_connections()
        self._init_connections_sparse()
        
        print(f"🚀 NeuralNetwork 1.2.2 | {self.n} neuronas | Ciclo Visión OK")
        
        
    def get_port(self, port_name):
     """Busca un puerto por nombre dentro de todos los módulos registrados."""
     if hasattr(self, 'modules'):
        for module in self.modules.values():
            # Si tus módulos guardan los puertos en un diccionario interno o lista
            if hasattr(module, 'ports') and port_name in module.ports:
                return module.ports[port_name]
            # O si los tenés como atributos directos (como self.out_x)
            for attr_name, attr_val in module.__dict__.items():
                if attr_name.startswith("out_") and hasattr(attr_val, "name") and attr_val.name == port_name:
                    return attr_val
     return None
        

    def _generate_initial_dna(self):
        return [random.uniform(0, 150), random.uniform(0, 150), random.uniform(0, 100),
                random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0.1, 1.0)]

    def _apply_initial_genetics(self):
        for i in range(self.n):
            self.genetics.mutate_new_neuron(self, i)

    def _assign_initial_subregions(self):
        for i in range(self.n):
            r_id = self.neuron_region[i]
            region = self.regions[r_id]
            if region.sub_regions:
                s_id = random.choice(list(region.sub_regions.keys()))
                self.neuron_subregion[i] = s_id
                region.sub_regions[s_id].add_neuron(i)
                prof = region.sub_regions[s_id].current_profile
                self.tau_m[i] = prof.tau_m
                self.v_thresh[i] = prof.v_thresh
                self.refr_period[i] = prof.refr_period

    import math

    def _init_core_connections(self):
        rf_profile = EVO_MENU["RAPID_FIRE"]
        x_count = y_count = z_count = 0
        
        # Definimos el centro del sistema visual (ajusta según tus coordenadas reales)
        # Ejemplo: si tu red es un cubo de 100x100x100, el centro es 50,50,50
        centro_vision = [50.0, 50.0, 50.0] 
        UMBRAL_VISION = 35.0  # Radio de influencia para conexiones visuales
    
        for i in range(self.n):
            if not self.active[i]: continue
    
            # Conexión al corazón (se mantiene igual)
            if random.random() < 0.06:
                self.heart.stress_port.connect(i)
    
            # Cálculo de distancia euclidiana
            pos = self.positions[i]
            dist = math.sqrt(sum((pos[j] - centro_vision[j])**2 for j in range(3)))
    
            # Solo si la neurona está dentro del radio de influencia visual
            if dist < UMBRAL_VISION:
                # Probabilidades ajustadas para los puertos
                r = random.random()
                
                # Fovea (radio)
                if r < 0.20:
                    self.vision.fovea_radius_port.connect(i)
                    z_count += 1
                # Coord X
                elif r < 0.45:
                    self.vision.coord_x_port.connect(i)
                    x_count += 1
                # Coord Y
                elif r < 0.70:
                    self.vision.coord_y_port.connect(i)
                    y_count += 1
    
                # Aplicar perfil de fuego rápido a las neuronas que logran conectar al sistema visual
                if r < 0.70:
                    self.tau_m[i] = rf_profile.tau_m
                    self.v_thresh[i] = rf_profile.v_thresh
                    self.refr_period[i] = rf_profile.refr_period
    
        print(f"🔌 Motores oculares (Euclidianos) → X:{x_count} | Y:{y_count} | Fóvea:{z_count} | Total:{x_count+y_count+z_count}")

    
    def backfill_orphan_connections(self, max_new_per_neuron=8, min_connections=1):
        """
        Cablea retroactivamente las neuronas activas con MENOS de
        min_connections conexiones. Con min_connections=1 se comporta
        como antes (solo huérfanas totales); con un valor mayor también
        repara a las que solo tenían el vínculo heredado del padre
        (systems/growth.py, antes del fix de vecindario real).
        """
        pocas = [i for i in range(self.n) if self.active[i] and len(self.connections[i]) < min_connections]
        if not pocas:
            print(f"🔌 [Backfill] Ninguna neurona por debajo de {min_connections} conexiones — nada que hacer.")
            return

        conectadas = 0
        for i in pocas:
            r_id = self.neuron_region[i]
            s_id = self.neuron_subregion[i]
            
            if r_id in self.regions and s_id in self.regions[r_id].sub_regions:
                radius = self.regions[r_id].sub_regions[s_id].current_profile.conn_radius
            else:
                radius = RADIUS_3D
    
            pos_i = self.positions[i]
            dist = np.linalg.norm(self.positions[:self.n] - pos_i, axis=1)
            candidatos = np.argsort(dist)
            
            agregadas = 0
            for j in candidatos:
                j = int(j)
                if j == i or not self.active[j] or j in self.connections[i]:
                    continue
                if dist[j] > radius:
                    break
                self._connect(i, j, random.uniform(0.3, 0.7))
                agregadas += 1
                if agregadas >= max_new_per_neuron:
                    break
                
            if agregadas > 0:
                conectadas += 1
            
        print(f"🔌 [Backfill] {conectadas}/{len(pocas)} neuronas recibieron conexiones nuevas.")

    def receive_spike(self, target_idx, strength, arrival_time):
            with self.lock:
                if not (0 <= target_idx < self.n) or not self.active[target_idx]:
                    return
                
                qsize = len(self.event_queue)
                
                # Asegúrate de que MAX_EVENTS_QUEUE esté definido arriba (ej. MAX_EVENTS_QUEUE = 20000)
                if qsize > MAX_EVENTS_QUEUE:
                    # Cuanto más saturada la cola, más fuerza exigimos para entrar
                    umbral_dinamico = 0.3 + min(1.5, (qsize - MAX_EVENTS_QUEUE) / 50000.0)
                    if strength < umbral_dinamico:
                        return
                
                heapq.heappush(self.event_queue, (float(arrival_time), int(target_idx), float(strength)))

    def inject_sensory_activity(self, points):
        if not points: return

        with self.lock:
            n_actual = self.n
            pos_copy = self.positions[:n_actual].copy()
            active_copy = self.active[:n_actual].copy()

        events = []
        for p in points:
            x = float(p[0] * 150)
            y = float(p[1] * 150)
            z = float(p[2]) if len(p) > 2 else 95.0
            intensity = float(p[3]) if len(p) > 3 else 7.0

            target = np.array([x, y, z])
            # ANTES: radio 24.0 — dejaba fuera a la banda de lenguaje (mín 36.4)
            # AHORA: 45.0, coherente con RADIUS_3D que ya usás en el resto del código
            nearby = np.where((np.linalg.norm(pos_copy - target, axis=1) < 45.0) & active_copy)[0]

            for idx in nearby:
                events.append((int(idx), intensity))

        if events:
            with self.lock:
                t = self.current_time
                for idx, strength in events:
                    self.receive_spike(idx, strength, t)

    def process_event(self, t_event, idx, strength):
        if not (0 <= idx < self.n) or t_event < self.refractory_until[idx]:
            return

        dt = t_event - self.last_update_time[idx]

        # --- FIX: blindaje contra dt negativo (condición de carrera entre
        # hilos que empujan al event_queue con timestamps de momentos
        # distintos) y contra exponentes extremos que overflowean al
        # castear a float32 en membrane_potential.
        if dt < 0:
            dt = 0.0

        exponente = -dt / self.tau_m[idx]
        exponente = min(700.0, max(-700.0, exponente))  # exp(700) ya roza el límite de float64

        v_decayed = V_REST + (self.membrane_potential[idx] - V_REST) * math.exp(exponente)
        v_now = v_decayed + strength
        self.membrane_potential[idx] = np.clip(v_now, -1e6, 1e6)  # blindaje extra si strength también se desboca
        self.last_update_time[idx] = t_event

        if v_now >= self.v_thresh[idx]:
            self._execute_spike(idx, t_event)

    def _execute_spike(self, idx, t):
        # --- FASE 1: PROTEGIDA ---
        with self.lock:
            self.fired[idx] = 1
            self.last_spike[idx] = float(t)
            self.membrane_potential[idx] = V_RESET
            self.refractory_until[idx] = t + self.refr_period[idx]
    
            # FASE 1: Eliminadas las variables is_coord_x, is_coord_y, is_fovea
            # worker_spiking es el único responsable de los votos motores
            my_nt          = self.nt_vector[idx].copy()
            my_pos         = self.positions[idx].copy()
            my_connections = list(self.connections[idx])

        # --- FASE 2: PROPAGACIÓN PURA ---
        # El motor ocular fue eliminado de acá.
        # worker_spiking acumula votos → update_fovea_physics los consume.
        # (El bloque intermedio de recolección de puertos genéricos fue eliminado
        # para evitar redundancia y mejorar drásticamente el rendimiento).
        for target in my_connections:
            if not self.active[target]: continue
    
            base_weight = self.weights.get((idx, target), 0.5)
    
            affinity        = np.dot(my_nt, self.receptors[target])
            chemical_factor = 0.5 + (affinity * 0.5)
            final_strength  = base_weight * chemical_factor
    
            if getattr(self, 'ignition_active', False):
                final_strength *= IGNITION_BOOST_FACTOR # Asegúrate de que IGNITION_BOOST_FACTOR esté definido
    
            target_pos   = self.positions[target]
            dist         = math.sqrt(sum((my_pos[k] - target_pos[k])**2 for k in range(3)))
            # SYNAPTIC_DELAY_BASE debe estar definido a nivel de módulo/clase
            arrival_time = t + SYNAPTIC_DELAY_BASE + (dist * 0.01) 
    
            self.receive_spike(target, final_strength, arrival_time)
    
    def add_neuron(self, pos=None, region_id=0, parent_idx=None):
         with self.lock:
             if self.n >= self.max_neurons:
                 return None
             idx = self.n

             # Autorreparación
             for attr in ['neuron_region', 'neuron_subregion']:
                 a = getattr(self, attr)
                 if len(a) < self.max_neurons:
                     new_a = np.full(self.max_neurons, -1, dtype=np.int32)
                     new_a[:len(a)] = a
                     setattr(self, attr, new_a)

             self.active[idx] = True
             self.positions[idx] = pos if pos is not None else [75., 75., 50.]
             self.neuron_region[idx] = int(region_id)
             self.energy[idx] = 0.6
             self.membrane_potential[idx] = V_REST
             self.connections[idx] = []
             self.dna[idx] = list(self.dna[parent_idx]) if parent_idx is not None else self._generate_initial_dna()

             self.n += 1
             return idx

    def _init_connections_sparse(self):
        print("🔌 Tejiendo conexiones sparse...")
        for i in range(self.n):
            r_id = self.neuron_region[i]
            s_id = self.neuron_subregion[i]
            radius = self.regions[r_id].sub_regions[s_id].current_profile.conn_radius if s_id != -1 else RADIUS_3D

            for j in range(i+1, self.n):
                if np.linalg.norm(self.positions[i] - self.positions[j]) <= radius:
                    self._connect(i, j, random.uniform(0.3, 0.7))

    def _connect(self, i, j, weight):
        with self.lock:
            if j not in self.connections[i]: self.connections[i].append(j)
            if i not in self.connections[j]: self.connections[j].append(i)
            self.weights[(i, j)] = weight
            self.weights[(j, i)] = weight

    def apply_dopamine_reward(self, visual_success, habituacion=1.0):
        for region in self.regions.values():
            r_val = self.reward_manager.compute_reward(region, self, visual_success, habituacion)
            self.reward_manager.apply_dopamine(region, r_val, self)

        # Sincronizar dopamine_level global con el promedio real de regiones
        if self.regions:
            self.dopamine_level = sum(r.dopamine for r in self.regions.values()) / len(self.regions)

        # === FITNESS NEURONAL: ENTRENAMIENTO ===
        # Si hay éxito visual y dopamina alta, las neuronas activas
        # suben su fitness (se "entrenan"). Si no hay éxito, decae levemente.
        if not hasattr(self, 'fitness_neuronal'):
            return

        dopamina_actual = self.dopamine_level

        if visual_success and dopamina_actual > 0.15:
            # Recompensa de entrenamiento: neuronas que dispararon suben su fitness
            # La ganancia es proporcional a la dopamina y la habituación
            ganancia = 0.002 * dopamina_actual * habituacion
            mascara_activas = self.fired[:self.n].astype(bool) & self.active[:self.n]
            self.fitness_neuronal[:self.n][mascara_activas] += ganancia
        else:
            # Sin éxito: pequeño decaimiento del fitness (desentrenamiento lento)
            decaimiento = 0.0001
            self.fitness_neuronal[:self.n][self.active[:self.n]] -= decaimiento

        # Mantener dentro del rango biológico [0.5, 2.0]
        self.fitness_neuronal[:self.n] = np.clip(
            self.fitness_neuronal[:self.n], 0.5, 2.0
        )
        
        
    
    def inject_inhibitory_signal(self, dolor_multiplier: float):
        """
        Aplica un 'calambre' inhibitorio masivo a la capa motora (Z=60)
        para paralizar los motores que empujan contra la pared.
        """
        # Multiplicador fuerte negativo (ej: -50mV si dolor es máximo)
        fuerza_inhibicion = -50.0 * dolor_multiplier 
        
        with self.lock:
            # OPICIÓN A: Si usas matrices Numpy (Recomendado para SNN rápidas)
            if hasattr(self, 'voltages') and hasattr(self, 'positions'):
                # Busca las neuronas motoras (Capa Z cercana a 60)
                mascara_motora = (self.positions[:, 2] >= 55.0) & (self.positions[:, 2] <= 65.0)
                # Inyecta el voltaje negativo
                self.voltages[mascara_motora] += fuerza_inhibicion
                
            # OPCIÓN B: Si usas una lista de objetos 'Neuron' en Python puro
            elif hasattr(self, 'neurons'):
                for n in self.neurons:
                    if hasattr(n, 'z') and 55.0 <= n.z <= 65.0:
                        n.voltage += fuerza_inhibicion    


#core/region.py


import random
import time
import numpy as np
from typing import Optional, Any
from dataclasses import dataclass
from collections import deque

@dataclass
class Profile:
    """ADN de comportamiento eléctrico para una SubRegión."""
    name: str
    tau_m: float           # Constante de tiempo de membrana (Leaky)
    v_thresh: float        # Umbral de disparo
    refr_period: float     # Periodo refractario (ms)
    energy_drain: float    # Costo metabólico extra por paso (Fase 2)
    conn_radius: float     # Radio de búsqueda para sinapsis

class SubRegion:
    def __init__(self, sub_id: int, parent_id: int, profile: Profile):
        self.sub_id = sub_id
        self.parent_id = parent_id
        self.neurons = []  # Lista de índices de neuronas en NeuralNetwork
        
        # --- ESTADO EVOLUTIVO ---
        self.current_profile = profile
        self.best_profile = profile
        self.backup_profile = None  # Snapshot para Rollback
        
        # --- MONITOREO DE RENDIMIENTO ---
        self.stress_level = 0.0
        self.saturation = 0.0      # % de actividad neuronal reciente
        self.success_history = deque(maxlen=50) 
        
        # --- CONTROL DE MUTACIÓN (Fase 3) ---
        self.last_mutation_time = 0.0
        self.eval_steps_left = 0    # Contador de pasos para Paso 3
        self.is_evaluating = False

    def add_neuron(self, neuron_idx: int):
        if neuron_idx not in self.neurons:
            self.neurons.append(neuron_idx)

    def calculate_stress(self, regional_dopamine: float):
        """Motor de Estrés v1.5: Conversión forzada de tipos para evitar errores de NumPy."""
        import numpy as np
        from collections import deque

        # 1. Cálculo base
        ws, wd = 0.4, 0.6
        dopamine_term = 1.0 - max(0.0, min(1.0, regional_dopamine))
        self.stress_level = (ws * self.saturation) + (wd * dopamine_term)

        # --- REPARACIÓN DINÁMICA DE TIPO ---
        # Si success_history es un array de numpy o no tiene el método append...
        if isinstance(self.success_history, np.ndarray) or not hasattr(self.success_history, "append"):
            # Lo convertimos a una lista clásica de Python y luego a deque
            try:
                # Si es un array de numpy, .tolist() es lo más seguro
                data_list = self.success_history.tolist() if hasattr(self.success_history, "tolist") else list(self.success_history)
            except:
                data_list = []
            
            # Re-inicializamos como deque con límite de 50 para no saturar la RAM
            self.success_history = deque(data_list, maxlen=50)
        # -----------------------------------

        # 2. Ahora el .append() funcionará SIEMPRE
        self.success_history.append(1.0 - self.stress_level)
        
        return self.stress_level

    # --- LÓGICA DE FASE 3: CICLO ESPECULATIVO ---

    def initiate_speculative_jump(self, available_profiles: list, eval_window: int = 100):
        """
        Paso 1: Snapshot Quirúrgico.
        Paso 2: Salto Evolutivo.
        """
        # Guardamos solo el perfil (Snapshot) para posible Rollback
        self.backup_profile = self.current_profile
        
        # Selección reactiva al tipo de estrés
        if self.saturation > 0.8:
            # Si hay saturación, priorizamos perfiles de alta velocidad (RAPID_FIRE)
            candidates = [p for p in available_profiles if p.energy_drain > self.current_profile.energy_drain]
        else:
            # Si el estrés es por dopamina, probamos variedad
            candidates = [p for p in available_profiles if p.name != self.current_profile.name]

        if candidates:
            self.current_profile = random.choice(candidates)
            
        self.eval_steps_left = eval_window
        self.is_evaluating = True
        self.last_mutation_time = time.time()
        
        print(f"🧬 [JUMP] SubRegión {self.sub_id}: Probando {self.current_profile.name} por {eval_window} pasos.")



    def evaluate_jump(self, net: Optional[Any] = None):
        """
        Paso 4: Decisión (Rollback o Commit).
        Actualización física de sinapsis con blindaje contra AttributeError y KeyError de NumPy.
        """
        if not self.is_evaluating:
            return

        # 1. Análisis de éxito
        # Calculamos el promedio histórico para comparar con el rendimiento actual
        avg_historical_success = sum(self.success_history) / len(self.success_history) if self.success_history else 0.5
        current_success = 1.0 - getattr(self, 'stress_level', 0.5)

        # --- Extracción segura de parámetros de perfil ---
        # Blindaje contra perfiles incompletos usando getattr
        curr_rad = getattr(self.current_profile, 'conn_radius', 25.0)
        back_rad = getattr(self.backup_profile, 'conn_radius', 25.0)

        # 2. Lógica de Decisión
        if current_success > avg_historical_success:
            # --- COMMIT: La mutación es beneficiosa ---
            self.best_profile = self.current_profile
            print(f"✅ [COMMIT] SubRegión {self.sub_id}: Perfil {self.current_profile.name} estabilizado.")
            
            # --- ACTUALIZACIÓN FÍSICA (Re-Wiring) ---
            if net and hasattr(net, 'refresh_neuron_connections'):
                if curr_rad != back_rad:
                    print(f"🔌 [Re-Wiring] Radio cambió ({back_rad} -> {curr_rad}). Re-escaneando...")
                    for idx in self.neurons:
                        # FIX: Forzamos int() para evitar KeyError: np.int32(0) en el diccionario de red
                        safe_idx = int(idx)
                        if net.active[safe_idx]:
                            net.refresh_neuron_connections(safe_idx)
        else:
            # --- ROLLBACK: La mutación falló, regresamos al snapshot anterior ---
            print(f"⏪ [ROLLBACK] SubRegión {self.sub_id}: Regresando a {self.backup_profile.name}.")
            self.current_profile = self.backup_profile
            
            # Restauramos el estado de salud regional si es posible
            if hasattr(self, 'energy_buffer'):
                self.energy_buffer = 1.0 

        # 3. Limpieza y Finalización del Ciclo
        self.is_evaluating = False
        self.eval_steps_left = 0 # Asegúrate de que coincida con el nombre de tu contador
        
        # Limpiamos el historial para empezar la medición del nuevo perfil desde cero
        if hasattr(self, 'success_history') and hasattr(self.success_history, 'clear'):
            self.success_history.clear()

class Region:
    def __init__(self, region_id: int):
        self.id = region_id
        self.sub_regions = {} # {sub_id: SubRegion}

        # === ADN REGIONAL (Firma Química) ===
        self.signature = [random.uniform(0.1, 0.5) for _ in range(5)]
        self._normalize_signature()

        # === NEUROMODULADORES ===
        self.dopamine = 0.1
        self.serotonin = 0.5
        self.acetylcholine = 0.4

        # === ESTADO METABÓLICO ===
        self.energy = 1.0
        self.prev_energy = 1.0
        self.is_active = True 

        # === MÉTRICAS DE RENDIMIENTO ===
        self.activity = 0.0
        self.prev_activity = 0.0
        self.output = 0.0
        self.fitness = 1.0 
        self.stress = 0.0
        self.novelty = 0.0
        self.env_interaction = 0.0

        # === ESTRATEGIA Y CONTROL ===
        self.strategy = "balanced"
        self.low_dopamine_counter = 0

    # --- GESTIÓN DE JERARQUÍA ---

    def create_sub_region(self, sub_id: int, profile: Profile) -> SubRegion:
        """Crea un micro-circuito con un perfil genético específico."""
        new_sub = SubRegion(sub_id, self.id, profile)
        self.sub_regions[sub_id] = new_sub
        return new_sub

    def get_avg_subregion_stress(self) -> float:
        if not self.sub_regions:
            return self.stress
        return sum(s.stress_level for s in self.sub_regions.values()) / len(self.sub_regions)

    # --- DINÁMICA QUÍMICA ---

    def _normalize_signature(self):
        total = sum(self.signature) or 1.0
        self.signature = [s / total for s in self.signature]

    def update_signature(self, avg_nt_vector, lr=0.05):
        if avg_nt_vector is None or len(avg_nt_vector) == 0: 
            return
        for i in range(len(self.signature)):
            self.signature[i] += (avg_nt_vector[i] - self.signature[i]) * lr
        self._normalize_signature()

    # --- DINÁMICA METABÓLICA ---

    def update_energy(self, delta_energy: float):
        self.energy += delta_energy
        self.energy *= 0.998  # Decaimiento metabólico basal
        self.energy = max(0.1, min(2.5, self.energy))

    def stabilize_dopamine(self):
        if abs(self.dopamine) > 0.01:
            self.dopamine *= 0.95
        self.dopamine = max(-0.6, min(0.8, self.dopamine))

    def update_strategy(self):
        """Cambia el modo de operación según los recursos y estrés regional."""
        if self.energy < 0.6:
            self.strategy = "conservative"
        elif self.stress > 0.8:
            self.strategy = "defensive"
        elif self.dopamine > 0.4:
            self.strategy = "explorer"
        else:
            self.strategy = "balanced"

    def update_state(
        self,
        activity: float,
        output: float,
        energy_input: float,
        env_signal: float,
        net: Optional[Any] = None
    ) -> None:
        """Ciclo principal de actualización (Metabolismo + Estrés Sub-Regional)."""
        self.prev_activity = self.activity
        self.activity = activity
        self.output = output
        self.prev_energy = self.energy

        # 1. PROCESAMIENTO DE SUB-REGIONES (Fase 2)
        extra_profile_cost = 0.0
        if self.sub_regions and net is not None:
            for sub in self.sub_regions.values():
                # Calcular saturación (actividad real de sus neuronas)
                if sub.neurons:
                    fired_now = sum(1 for idx in sub.neurons if net.fired[idx])
                    sub.saturation = fired_now / len(sub.neurons)
                
                # Actualizar estrés individual basado en la dopamina actual de la región
                sub.calculate_stress(self.dopamine)
                
                # Sumar costo energético del perfil (Rapid_Fire gasta más)
                extra_profile_cost += sub.current_profile.energy_drain

        # 2. COSTO METABÓLICO
        base_cost = abs(self.activity) * 0.05
        mults = {"conservative": 0.5, "explorer": 1.4, "defensive": 0.8, "balanced": 1.0}
        
        # El costo total suma la actividad general y el mantenimiento de perfiles
        activity_cost = (base_cost * mults.get(self.strategy, 1.0)) + (extra_profile_cost * 0.1)
        self.update_energy(energy_input - activity_cost)

        # 3. ACTUALIZACIÓN DE MÉTRICAS REGIONALES
        self.fitness = (self.energy * 0.4) + (self.dopamine * 0.6)
        self.stress = max(0.0, 1.0 - self.energy)
        self.novelty = abs(self.activity - self.prev_activity)
        self.env_interaction = activity * env_signal

        self.update_strategy()

        # 4. DINÁMICA DE NEUROMODULADORES
        delta_e = self.energy - self.prev_energy
        target_dopamine = (delta_e * 1.5) + (self.novelty * 0.4) + (env_signal * 0.2)
        
        self.dopamine += (target_dopamine - self.dopamine) * 0.1
        self.serotonin += ((1.0 - self.stress) - self.serotonin) * 0.05
        self.acetylcholine += ((self.novelty + self.stress) - self.acetylcholine) * 0.1

        self.stabilize_dopamine()

        # Recuperación de depresión dopaminérgica
        if self.dopamine < -0.3:
            self.low_dopamine_counter += 1
            if self.low_dopamine_counter > 30:
                self.dopamine = 0.1
                self.low_dopamine_counter = 0
        else:
            self.low_dopamine_counter = 0
            
    # --- COMPETENCIA INTER-REGIONAL ---
    
    def apply_lateral_inhibition(self, other_regions):
            REGION_INHIBITION_STRENGTH = 0.5
            REGION_DOMINANCE_THRESHOLD = 0.8
    
            if self.activity > REGION_DOMINANCE_THRESHOLD:
                bias_force = (self.activity - REGION_DOMINANCE_THRESHOLD) * REGION_INHIBITION_STRENGTH
                
                for other in other_regions:
                    if other.id == self.id or not other.is_active:
                        continue
                    
                    inhibition_debt = bias_force * 0.15 
                    
                    # FIX: aplicar el mismo clamp que usa update_energy, en vez
                    # de sumar/restar sin límite
                    other.energy = max(0.1, min(2.5, other.energy - inhibition_debt))
                    self.energy  = max(0.1, min(2.5, self.energy + inhibition_debt * 0.5))
                    other.stress = min(1.0, other.stress + (bias_force * 0.2))

    def calculate_regional_gain(self) -> float:
        """Determina la sensibilidad de disparo de la región."""
        gain = (self.energy * 0.7) + (self.dopamine * 0.3)
        return max(0.5, min(2.0, gain))


# core/reward_system.py
import random
import logging

class RewardSystem:
    def __init__(self):
        # === PESOS DE RECOMPENSA (BAJADOS PARA EVITAR SATURACIÓN) ===
        self.w_vision      = 1.2   # Antes 2.5
        self.w_energy      = 0.6
        self.w_env         = 0.4
        self.w_novelty     = 0.8
        self.w_efficiency  = 0.7
        self.w_stress      = 1.5   # Penalización

    def compute_reward(self, region, net, visual_success=0.0, habituacion=1.0):
        """
        Calcula el valor de fitness/recompensa para una región específica.
        Incluye atenuación por habituación/inmovilidad.
        """
        try:
            # 1. DRIVE DE ENERGÍA
            delta_energy = region.energy - region.prev_energy
            drive_energy = (delta_energy * 2.0) + (region.energy * 0.2)

            # 2. NOVEDAD: Exploración de conexiones
            drive_novelty = region.novelty * 2.0

            # 3. ESTRÉS
            drive_stress = -region.stress * 1.5

            # 4. EFICIENCIA
            energy_used = max(0.01, abs(delta_energy))
            efficiency = min(2.0, region.output / energy_used)
            drive_efficiency = efficiency * 1.2

            # --- APLICAR HABITUACIÓN AL ÉXITO VISUAL ---
            # Escalamos el éxito visual antes de sumarlo al total
            visual_component = self.w_vision * visual_success * habituacion

            # --- CÁLCULO DE RECOMPENSA TOTAL ---
            reward = (
                (self.w_energy * drive_energy) +
                (self.w_env * region.env_interaction) +
                (self.w_novelty * drive_novelty) +
                (self.w_efficiency * drive_efficiency) +
                (self.w_stress * drive_stress) +
                visual_component
            )

            # --- SI ESTÁ TOTALMENTE QUIETO (Habituación crítica), PENALIZAR APATÍA ---
            # Si el parche detectó inmovilidad, bajamos la recompensa basal para que no sea rentable ser catatónico
            if habituacion <= 0.05:
                reward -= 0.4  # Penalización por aburrimiento (fuerza el escape metabólico)

            # --- MECANISMOS DE REGULACIÓN ---
            # A. Anti-Coma
            if region.activity < 0.02:
                reward += 0.3
            
            # B. Anti-Caos
            if region.activity > 0.7:
                reward -= 1.2

            # C. Control de Población
            vivas = sum(1 for a in net.active if a)
            if vivas > 500:
                reg_count = net.region_counts[region.id] if hasattr(net, 'region_counts') else 0
                if (reg_count / vivas) > 0.4: 
                    reward -= 0.6

            if region.id == 0 and random.random() < 0.05:
                print(f"DEBUG R0: Reward={reward:.2f} | Energy={region.energy:.2f} | Habituac={habituacion:.2f}")

            return reward

        except Exception as e:
            reg_id = region.id if hasattr(region, 'id') else region
            logging.error(f"Error en compute_reward Región {reg_id}: {e}")
            return 0.0

    def apply_dopamine(self, region, reward, net):
        """
        Transforma reward en Dopamina con recaptación biológica real.
        """
        # Inicializar la propiedad si no existe en la estructura de la región
        if not hasattr(region, 'dopamine'): region.dopamine = 0.0
        
        # 1. RECAPTACIÓN BIOLÓGICA (Clearance): La dopamina vieja decae un 20% antes de sumar la nueva
        # Esto evita que se quede estancada arriba si la recompensa baja
        region.dopamine *= 0.80
        
        # 2. Inyección de la nueva recompensa normalizada
        reward_clipped = max(-1.0, min(1.0, reward))
        
        # Suavizado (Inercia del 10%)
        region.dopamine += (reward_clipped) * 0.1
        region.dopamine = max(0.0, min(2.0, region.dopamine)) # Límites saludables

        # --- LOG DE ÉXITO CORREGIDO ---
        # Ahora printeamos el valor REAL del tanque de dopamina de la región
        if region.dopamine > 0.1:
            print(f"✨ [Dopamina Real] Región {region.id}: {region.dopamine:.2f}")

        # --- CASTIGO METABÓLICO ---
        if reward < -0.6:
            for i in range(net.n):
                if net.neuron_region[i] == region.id and net.active[i]:
                    net.energy[i] -= 0.05
                    
        # --- PREMIO GENÉTICO ---
        region.reproduction_chance = max(0.0, reward) if reward > 0.2 else 0.0


# =============================
# core/spatial.py
# =============================
import random
from config import GRID_SIZE, GRID_DEPTH

class SpatialGrid:
    def __init__(self, n):
        self.positions = [
            (random.randint(0, GRID_SIZE-1),
             random.randint(0, GRID_SIZE-1),
             random.randint(0, GRID_DEPTH-1))
            for _ in range(n)
        ]

    def distance(self, i, j):
        x1,y1,z1 = self.positions[i]
        x2,y2,z2 = self.positions[j]
        return abs(x1-x2) + abs(y1-y2) + abs(z1-z2)

#core/utils.py


import psutil
import os
import time

class SystemMetabolism:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        # Inicializamos para evitar el 0.0 del primer llamado
        self.process.cpu_percent() 
        
    def get_biological_impact(self):
        """
        Calcula qué tan 'estresado' está el hardware.
        Retorna un factor de 0.0 (relax) a 1.0 (crítico).
        """
        # 1. Monitoreo de RAM (Objetivo: 1GB)
        ram_mb = self.process.memory_info().rss / (1024 * 1024)
        # Empieza a estresar a partir de 850MB, llega al máximo en 1024MB
        ram_stress = max(0.0, min(1.0, (ram_mb - 850) / 174))
        
        # 2. Monitoreo de CPU (Objetivo: 15%)
        # Usamos interval=None para no bloquear el hilo principal
        cpu_usage = self.process.cpu_percent() / psutil.cpu_count()
        cpu_stress = max(0.0, min(1.0, (cpu_usage - 15) / 10))
        
        return {
            "ram_mb": ram_mb,
            "cpu_usage": cpu_usage,
            "stress_factor": max(ram_stress, cpu_stress), # El que sea más crítico
            "is_starving": ram_mb > 1024  # ¿Se pasó del GB?
        }

    def get_sleep_time(self, stress_factor):
        """
        Ajusta el delay dinámicamente basado en el estrés.
        """
        if stress_factor > 0.8:
            return 0.05  # Latencia alta: la IA se vuelve 'lenta' por falta de recursos
        if stress_factor > 0.3:
            return 0.01  # Latencia media
        return 0.001     # Latencia normal


# engine/simulation.py

from core.module import Module

import msvcrt
import logging 
 
import re
import threading

def clean_name(name):
    """Limpia cadenas de texto para evitar errores de codificación en la serialización."""
    return re.sub(r'[^a-zA-Z0-9_]', '', str(name))

import math

import time
import heapq
import logging
import numpy as np

import pickle
import os
import random
from config import * # Una sola importación limpia al inicio

from core.evolution_config import EVO_MENU, PROFILE
from core.utils import SystemMetabolism

# === SISTEMAS ===
from systems.growth import GrowthSystem
from systems.metabolism import MetabolismSystem
from systems.regions import RegionSystem
from systems.competition import CompetitionSystem
from systems.plasticity import PlasticitySystem
from systems.physics import PhysicsSystem

from modules.heart import HeartModule
from core.network import NeuralNetwork
from modules.attention import AttentionModule 

class SimulationEngine:
    
    def __init__(self, net=None):
        # 1. Sincronización de Red y Módulos Base
        self.net = net if net is not None else NeuralNetwork()
        self.state = "awake"
        self.is_running = False
        self.step_count = 0
        
        # --- NUEVO: Memoria del Reflejo Sacádico ---
        self.low_dopamine_cycles = 0
        self.is_centering_reflex = False
        
        # Capturador de pantalla para foveación
        from screen_interface import ScreenProcessor
        # Cambiar True por False para activar modo virtual
        self.scr_processor = ScreenProcessor(real_screen_mode=False)
       
        # Asegurarnos de que el diccionario de módulos exista en la red
        if not hasattr(self.net, 'modules'):
            self.net.modules = {}

        # 2. Vincular / Inicializar Módulos (con robustez)
       
        # --- MÓDULO DE VISIÓN ---
        self.vision = getattr(self.net, 'vision', None)
        if self.vision is None:
            from modules.vision import VisionModule
            self.vision = VisionModule(net=self.net)
            self.net.vision = self.vision
            self.net.modules["vision"] = self.vision # Agregado al dict para consistencia
            logging.info("👁️ Módulo de Visión creado desde cero.")
        else:
            self.vision.net = self.net  # <-- SOLUCIÓN: Re-vinculación forzada al despertar
            self.net.modules["vision"] = self.vision
            logging.info("👁️ Módulo de Visión recuperado y enlazado a la red.")

        # Conexión forzada de puertos
        if hasattr(self.vision, 'auto_connect'):
            self.vision.auto_connect(self.net)
            logging.info("🔌 Puertos de Visión conectados.")
        else:
            logging.warning("⚠️ VisionModule no tiene método auto_connect()")

        # --- MÓDULO DEL CORAZÓN ---
        if hasattr(self.net, 'heart') and self.net.heart is not None:
            self.heart = self.net.heart
            self.net.modules["heart"] = self.heart
            logging.info("❤️ Corazón recuperado desde la red.")
        else:
            from modules.heart import HeartModule
            self.heart = HeartModule()
            self.net.heart = self.heart
            self.net.modules["heart"] = self.heart
            logging.info("🌱 Corazón nuevo inicializado.")

        self.heart.auto_connect(self.net)

        # --- MÓDULO DE ATENCIÓN ---
        if hasattr(self.net, 'attention') and self.net.attention is not None:
            self.attention = self.net.attention
            self.attention.net = self.net  # <-- PROTECCIÓN: También re-vinculamos atención
            self.net.modules["attention"] = self.attention
            logging.info("🎯 Módulo de Atención recuperado y enlazado.")
        else:
            from modules.attention import AttentionModule
            self.attention = AttentionModule(net=self.net)
            self.net.attention = self.attention
            self.net.modules["attention"] = self.attention
            logging.info("🎯 Módulo de Atención creado.")


        # Conectarlo siempre, sin importar si es nuevo o recuperado 
        if hasattr(self, 'lang_module') and hasattr(self.lang_module, 'auto_connect'):
            self.lang_module.auto_connect(self.net)
            logging.info("🔌 Puertos de Lenguaje (X, O) conectados a la red.")
            
            # --- NUEVO (Fase B) ---
            if hasattr(self.lang_module, 'assign_buffer_pools'):
                self.lang_module.assign_buffer_pools(self.net)
                logging.info("🧠 Pools de memoria de trabajo (buffer) asignados por letra.")


        from modules.output_module import OutputModule
        self.output_module = OutputModule(umbral_certeza=0.45, output_path="detecciones.json")    

        # =====================================================================
        # --- NUEVO: MÓDULO DE LENGUAJE (Puertos IO de Letras) ---
        # =====================================================================
        if "lenguaje" in self.net.modules and self.net.modules["lenguaje"] is not None:
            self.lang_module = self.net.modules["lenguaje"]
            self.lang_module.net = self.net
            logging.info("🗣️ Módulo de Lenguaje recuperado y enlazado.")
        else:
            try:
                from modules.language_io import LetterIOModule
                self.lang_module = LetterIOModule(net=self.net, symbols=("X", "O"))
                self.net.modules["lenguaje"] = self.lang_module
                self.net.lang_module = self.lang_module # Atributo directo por conveniencia
                logging.info("🗣️ Módulo de Lenguaje (Letras) inicializado.")
            except ImportError:
                logging.warning("⚠️ No se encontró modules.language_io, omitiendo Módulo de Lenguaje.")
        
        # Conectarlo siempre, sin importar si es nuevo o recuperado
        if hasattr(self, 'lang_module') and hasattr(self.lang_module, 'auto_connect'):
            self.lang_module.auto_connect(self.net)
            logging.info("🔌 Puertos de Lenguaje (X, O) conectados a la red.")

        # 3. Sistemas Biológicos
        self.growth = GrowthSystem()
        self.metabolism = MetabolismSystem()
        self.regions_sys = RegionSystem() 
        self.competition = CompetitionSystem()
        self.plasticity = PlasticitySystem()
        self.physics = PhysicsSystem()

        # =====================================================================
        # MEJORA DE INFRAESTRUCTURA COMPARTIDA (Actualizado)
        # =====================================================================
        if self.vision is not None:
            # 1. Validar velocidades e inercia
            if not hasattr(self.vision, 'vel_x'): self.vision.vel_x = 0.0
            if not hasattr(self.vision, 'vel_y'): self.vision.vel_y = 0.0
            if not hasattr(self.vision, 'friction'): self.vision.friction = 0.85
            
            # 2. PROTECCIÓN CRÍTICA: Validar coordenadas espaciales de la fóvea
            # Si el guardado viejo no las tiene, lo centramos por defecto en 0.5
            if not hasattr(self.vision, 'coord_x'): self.vision.coord_x = 0.5
            if not hasattr(self.vision, 'coord_y'): self.vision.coord_y = 0.5
            
            logging.info(f"🛸 [Infraestructura] Estado físico completo en self.vision (X: {self.vision.coord_x} | Y: {self.vision.coord_y})")
        else:
            logging.error("❌ [Infraestructura] No se pudo inicializar la física: self.vision es None.")


    def trigger_letter_training(self, letter):
        import threading
        import time
        import numpy as np
        import cv2
        import logging
        import traceback

        def training_task():
            try:
                logging.info(f"🎓 [Curriculum] Iniciando trial de entrenamiento: Letra '{letter}'")
                self._modo_entrenamiento = True
                
                pos = (0.5, 0.5)
                self.vision.fovea_center_x, self.vision.fovea_center_y = pos
                self.vision.vel_x = self.vision.vel_y = 0.0
                
                activas = set()
                frames = 60
                
                for _ in range(frames):
                    frame = self.scr_processor.generate_letter_frame(letter, pos)
                    coords, _, _ = self.scr_processor.get_activity_coords(focus_pt=pos, virtual_frame=frame)
                    
                    scale = getattr(self.scr_processor, 'scale_factor', 1.0)
                    resized = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1.0 else frame
                    static = self.scr_processor.get_static_contrast_coords(resized, pos)
                    
                    self.vision.update_from_external(coords + static)
                    self.vision.transport_to_net(self.net, self.net.current_time)
                    
                    with self.net.lock:
                        # --- INICIO DEL FIX APLICADO ---
                        ventana_ms = 200.0
                        t_actual = self.net.current_time
                        recientes = np.where(
                            (self.net.last_spike[:self.net.n] > 0) &
                            ((t_actual - self.net.last_spike[:self.net.n]) < ventana_ms) &
                            (np.abs(self.net.positions[:self.net.n, 2] - 45.0) < 8.0)
                        )[0]
                        # --- FIN DEL FIX APLICADO ---
                    
                    activas.update(int(i) for i in recientes)
                    time.sleep(1/30)
                    
                lang = self.net.modules["lenguaje"]
                
                evidencia = lang.input_ports[f"evidencia_{letter}"]
                for idx in activas:
                    if idx not in evidencia.connected_neurons:
                        evidencia.connected_neurons.append(int(idx))
                        
                maestro = lang.output_ports[f"maestro_{letter}"]
                for idx in list(maestro.connected_neurons)[:20]:
                    self.net.receive_spike(idx, 5.0, self.net.current_time)
                    
                if evidencia.value > 0.15:
                    if hasattr(self.net, 'apply_dopamine_reward'):
                        self.net.apply_dopamine_reward(visual_success=True, habituacion=1.0)
                    logging.info(f"✅ Trial '{letter}' exitoso. Evidencia: {evidencia.value:.2f} | Neuronas enlazadas: {len(activas)}")
                else:
                    logging.warning(f"⚠️ Trial '{letter}' débil. Evidencia: {evidencia.value:.2f} | Neuronas enlazadas: {len(activas)}")
                
            except Exception as e:
                # --- NUEVO: ya no dejamos que el hilo muera en silencio ---
                logging.error(f"❌ [Curriculum] Excepción no capturada en trial '{letter}': {e}")
                traceback.print_exc()
            finally:
                # --- NUEVO: garantiza salir de modo entrenamiento pase lo que pase ---
                self._modo_entrenamiento = False
    
        threading.Thread(target=training_task, daemon=True).start()

       
    def recalibrar_musculos_motores(self):
        """
        Fuerza a los puertos motores a conectarse a neuronas reales de la capa Z=60.
        Repara la pérdida de conexiones durante la persistencia.
        """
        with self.net.lock:
            if hasattr(self.net, 'positions'):
                neuronas_z60 = np.where((self.net.positions[:, 2] >= 55.0) & (self.net.positions[:, 2] <= 65.0))[0].tolist()
            else:
                neuronas_z60 = [i for i, n in enumerate(self.net.neurons) if getattr(n, 'z', 60) == 60]
            
            if len(neuronas_z60) < 50:
                logging.warning(f"⚠️ Poca densidad muscular en Z=60 ({len(neuronas_z60)}n). Reparación abortada.")
                return

            random.shuffle(neuronas_z60)
            mitad = len(neuronas_z60) // 2
            
            if hasattr(self, 'vision'):
                logging.info(f"🔧 [Músculos] Forzando cableado de emergencia en Z=60 para {len(neuronas_z60)} neuronas.")
                self.vision.coord_x_port.connected_neurons = set(neuronas_z60[:mitad])
                self.vision.coord_y_port.connected_neurons = set(neuronas_z60[mitad:])
                logging.info(f"✅ Cables restaurados -> X: {len(self.vision.coord_x_port.connected_neurons)} | Y: {len(self.vision.coord_y_port.connected_neurons)}")
       
    def worker_vision(self):
        """Hilo dedicado a la visión a ~30-60 FPS - Fases 1, 2 y 3 (Con Doble Canal Magnocelular/Parvocelular)"""
        target_fps = 30
        frame_time = 1.0 / target_fps

        if not hasattr(self, '_frames_quieto'):  self._frames_quieto = 0
        if not hasattr(self, '_stuck_counter'):  self._stuck_counter = 0
        if not hasattr(self, '_last_fovea_x'):   self._last_fovea_x  = 0.5
        if not hasattr(self, '_last_fovea_y'):   self._last_fovea_y  = 0.5
        if not hasattr(self, 'reflejo_frames'):  self.reflejo_frames  = 0
        if not hasattr(self, 'reflejo_mem_x'):   self.reflejo_mem_x   = 0.0
        if not hasattr(self, 'reflejo_mem_y'):   self.reflejo_mem_y   = 0.0

        while self.is_running:
            start_time = time.time()

            # ------------------------------------------------------------------
            # 0. FIX — AVANZAR EL RELOJ DE LA RED
            # net.current_time estaba congelado en 0.0 para siempre: nada en
            # todo el sistema lo incrementaba. Rompía la cola de eventos, el
            # STDP (dt = last_spike[j]-last_spike[i] siempre 0) y el filtro
            # anti-arranque de Fase A (t < 0.5 nunca se volvía falso).
            # ------------------------------------------------------------------
            with self.net.lock:
                self.net.current_time += frame_time * 1000.0  # a milisegundos

            # ------------------------------------------------------------------
            # 1. ACTUALIZAR FÍSICA (único lugar que escribe coord_x_port.value)
            # ------------------------------------------------------------------
            self.vision.update_fovea_physics(dt=frame_time)

            # ------------------------------------------------------------------
            # 2. LEER POSICIÓN REAL DEL OJO (solo lectura)
            # ------------------------------------------------------------------
            fovea_x = self.vision.fovea_center_x
            fovea_y = self.vision.fovea_center_y
            fovea_r = self.vision.fovea_radius_port.value

            # ------------------------------------------------------------------
            # 3. REFLEJO SACÁDICO DE CENTRADO (via votos)
            # ------------------------------------------------------------------
            if getattr(self, 'is_centering_reflex', False) and hasattr(self, 'vision') and self.vision:
                dx = 0.5 - fovea_x
                dy = 0.5 - fovea_y

                if abs(dx) < 0.02 and abs(dy) < 0.02:
                    self.is_centering_reflex = False
                    self.low_dopamine_cycles = 0
                    logging.info("🎯 [Reflejo] Fóvea centrada. Reiniciando exploración.")
                else:
                    with self.vision._vote_lock:
                        self.vision._vote_x += dx * 50
                        self.vision._vote_y += dy * 50

            # ------------------------------------------------------------------
            # 4 y 5. CAPTURA + INYECCIÓN — GUARD DE ENTRENAMIENTO
            # Si hay un trial de letra en curso, el hilo de trigger_letter_training
            # es el ÚNICO que debe escribir en self.vision.current_activations.
            # Antes, worker_vision seguía inyectando su lienzo en blanco en paralelo
            # y pisaba (race condition) la señal de la letra antes de que llegara
            # a la red — por eso la banda z=37-53 nunca recibía nada.
            # ------------------------------------------------------------------
            if not getattr(self, '_modo_entrenamiento', False):
                import cv2
                frame = self.scr_processor.generate_mission_frame()
            
                coords, score, new_focus = self.scr_processor.get_activity_coords(
                    focus_pt=(fovea_x, fovea_y),
                    focus_radius=fovea_r,
                    virtual_frame=frame
                )
            
                frame_resized = cv2.resize(frame, None, fx=self.scr_processor.scale_factor, fy=self.scr_processor.scale_factor)
                static_coords = self.scr_processor.get_static_contrast_coords(
                    frame_resized,
                    focus_pt=(fovea_x, fovea_y),
                    focus_radius=fovea_r
                )
            
                all_coords = coords + static_coords
            
                if all_coords:
                    self.vision.update_from_external(all_coords)
                    self.vision.transport_to_net(self.net, self.net.current_time)
            else:
                # Durante el trial, mantenemos 'score' válido para no romper la
                # sección 7 (habituación), que sigue leyéndolo más abajo.
                score = getattr(self.vision, 'last_score', 0.0)

            # ------------------------------------------------------------------
            # 5.5. FASE A — SECUENCIAS: esta SIEMPRE corre, con o sin entrenamiento,
            # porque es justo lo que necesitamos medir durante el trial.
            # ------------------------------------------------------------------
            if hasattr(self, 'lang_module') and self.lang_module:
                self.lang_module.update_evidence_values(self.net, self.net.current_time)
                self.lang_module.check_edge_events(self.net, self.net.current_time)

            # ------------------------------------------------------------------
            # 6. NOCICEPCIÓN (via votos)
            # ------------------------------------------------------------------
            dolor_multiplier = 0.0
            reflejo_x        = 0.0
            reflejo_y        = 0.0

            if fovea_x <= 0.10:
                dolor_multiplier = max(dolor_multiplier, (0.10 - fovea_x) / 0.05)
                reflejo_x = +1.0
            elif fovea_x >= 0.90:
                dolor_multiplier = max(dolor_multiplier, (fovea_x - 0.90) / 0.05)
                reflejo_x = -1.0

            if fovea_y <= 0.10:
                dolor_multiplier = max(dolor_multiplier, (0.10 - fovea_y) / 0.05)
                reflejo_y = +1.0
            elif fovea_y >= 0.90:
                dolor_multiplier = max(dolor_multiplier, (fovea_y - 0.90) / 0.05)
                reflejo_y = -1.0

            dolor_multiplier = min(1.0, max(0.0, dolor_multiplier))

            if dolor_multiplier > 0 and self.reflejo_frames == 0:
                if hasattr(self.net, 'dopamine_level'):
                    self.net.dopamine_level = max(0.0, self.net.dopamine_level - (0.5 * dolor_multiplier))

                if hasattr(self.net, 'inject_inhibitory_signal'):
                    self.net.inject_inhibitory_signal(dolor_multiplier * 0.4)

                self.reflejo_frames = 15
                self.reflejo_mem_x  = reflejo_x * 3.0
                self.reflejo_mem_y  = reflejo_y * 3.0
                logging.warning(f"⚡ [Dolor] Espasmo iniciado. Fóvea en ({fovea_x:.2f}, {fovea_y:.2f})")

            if self.reflejo_frames > 0:
                with self.vision._vote_lock:
                    self.vision._vote_x += self.reflejo_mem_x
                    self.vision._vote_y += self.reflejo_mem_y
                self.reflejo_frames -= 1

            # ------------------------------------------------------------------
            # 7. HABITUACIÓN Y EXPLORACIÓN - FASES 3 y 4
            # ------------------------------------------------------------------
            distance_moved = ((fovea_x - self._last_fovea_x)**2 +
                              (fovea_y - self._last_fovea_y)**2) ** 0.5

            if distance_moved > 0.01:
                self._frames_quieto = 0
                self.vision.last_score = score
            else:
                self._frames_quieto += 1

                if not getattr(self, '_modo_entrenamiento', False):
                    if self._frames_quieto > 30:
                        with self.vision._vote_lock:
                            self.vision._vote_x += random.uniform(-1.5, 1.5)
                            self.vision._vote_y += random.uniform(-1.5, 1.5)

                        self.vision.last_score = score * 0.05

                    if self._frames_quieto > 180:
                        nuevo_x = random.uniform(0.15, 0.85)
                        nuevo_y = random.uniform(0.15, 0.85)

                        self.vision.fovea_center_x = nuevo_x
                        self.vision.fovea_center_y = nuevo_y
                        self.vision.vel_x          = 0.0
                        self.vision.vel_y          = 0.0

                        self.vision.last_score  = 0.0
                        self._frames_quieto     = 0

                        logging.info(f"👁️ [Exploración] Sacada a ({nuevo_x:.2f}, {nuevo_y:.2f})")

            self._last_fovea_x = fovea_x
            self._last_fovea_y = fovea_y

            # ------------------------------------------------------------------
            # 8. CONTROL DE RENDIMIENTO (protección térmica i5)
            # ------------------------------------------------------------------
            elapsed    = time.time() - start_time
            sleep_time = frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def worker_spiking(self):
            """Hilo principal de propagación de spikes - Fase 1: Votos musculares (sin escritura directa de puertos)"""
            logging.info("🧠 SpikingThread iniciado - Procesando spikes (Modo Rápido O(1))")
            
            total_spikes = 0
            motor_spikes = 0
            last_debug = time.time()
    
            # Cacheamos los sets de neuronas motoras
            motor_indices = set()
            if hasattr(self, 'vision') and self.vision:
                for port_name in ['coord_x_port', 'coord_y_port', 'fovea_radius_port']:
                    if hasattr(self.vision, port_name):
                        port = getattr(self.vision, port_name)
                        if hasattr(port, 'connected_neurons'):
                            neurons = port.connected_neurons
                            if isinstance(neurons, (set, list, tuple)):
                                motor_indices.update(neurons)
                            else:
                                motor_indices.add(neurons)
    
            while self.is_running:
                batch = []
             
                # === EXTRACCIÓN THREAD-SAFE CON CONTROL TEMPORAL EN LOTES ===
                with self.net.lock:
                    if self.net.event_queue:
                        # FIX: Subimos el tamaño del lote a 1000 para drenar más rápido
                        while self.net.event_queue and len(batch) < 1000:
                            if self.net.event_queue[0][0] <= self.net.current_time:
                                batch.append(heapq.heappop(self.net.event_queue))
                            else:
                                break
    
                # Procesamos el lote FUERA del lock
                if batch:
                     for t_ev, idx, strength in batch:
                         try:
                             self.net.process_event(t_ev, idx, strength)
                             total_spikes += 1
                             
                             # === CORRECCIÓN: ACUMULAR DIRECTO SIN EL IF ===
                             # Cada 'idx' aquí dentro representa actividad real. 
                             # Lo sumamos directamente al acumulador.
                             self.net.fired_acumulado[idx] += 1
                             # ==============================================
    
                             if idx in motor_indices:
                                
                                if hasattr(self, 'vision') and self.vision:
                                    # --- FASE 1: ACUMULACIÓN DE VOTOS (nunca escribe puertos directamente) ---
                                    energia_array = getattr(self.net, 'energy', None)
                                    
                                    if isinstance(energia_array, np.ndarray):
                                        energia_actual = float(energia_array[idx])
                                    else:
                                        energia_actual = 1.0
        
                                    umbral_fatiga = getattr(self.net, 'fatigue_threshold', 0.20)
                                    costo_spike   = getattr(self.net, 'motor_spike_cost', 0.001)
                                    eficiencia    = 1.0 if energia_actual > umbral_fatiga else 0.5
        
                                    # Dirección del músculo: pares empujan +, impares empujan -
                                    direction = 1 if idx % 2 == 0 else -1
                                    voto = direction * eficiencia
        
                                    spike_motor_consumido = False
    
                                    # Eje X: acumular voto (NO tocar coord_x_port.value)
                                    if hasattr(self.vision, 'coord_x_port') and idx in getattr(self.vision.coord_x_port, 'connected_neurons', []):
                                        spike_motor_consumido = True
                                        with self.vision._vote_lock:
                                            self.vision._vote_x += voto
        
                                    # Eje Y: acumular voto (NO tocar coord_y_port.value)
                                    if hasattr(self.vision, 'coord_y_port') and idx in getattr(self.vision.coord_y_port, 'connected_neurons', []):
                                        spike_motor_consumido = True
                                        with self.vision._vote_lock:
                                            self.vision._vote_y += voto
        
                                    # Costo metabólico solo si el spike fue motor
                                    if spike_motor_consumido:
                                        motor_spikes += 1
                                        if isinstance(energia_array, np.ndarray):
                                            self.net.energy[idx] = max(0.0, energia_actual - costo_spike)
        
                                        if eficiencia == 0.5 and total_spikes % 500 == 0:
                                            logging.warning(f"⚠️ [Músculo Neurona {idx}] Fatiga activa. ATP: {energia_actual:.3f}")
                                    # ----------------------------------------------------------------------
        
                         except Exception as e:
                             logging.error(f"Error procesando spike en {idx}: {e}")
                else:
                    time.sleep(0.001)
        
                # === TELEMETRÍA BIO ===
                current_t = time.time()
                if current_t - last_debug > 3.0:
                    with self.net.lock:
                        queue_size = len(getattr(self.net, 'event_queue', []))
        
                    gaze_pos = (0.5, 0.5)
                    if hasattr(self, 'vision') and self.vision:
                        vis = self.vision
                        if hasattr(vis, 'get_current_gaze'):
                            gaze_pos = vis.get_current_gaze()
                        elif hasattr(vis, 'fovea_center_x') and hasattr(vis, 'fovea_center_y'):
                            gaze_pos = (vis.fovea_center_x, vis.fovea_center_y)
        
                    print(f"\n📊 [TELEMETRÍA BIO] Spikes Totales: {total_spikes:,} | Motores: {motor_spikes:,}")
                    print(f"👁️  Posición Fóvea Real: X={gaze_pos[0]:.4f}, Y={gaze_pos[1]:.4f} | Event Queue: {queue_size}")
                    print("-" * 60)
                    last_debug = current_t
                
    def worker_slow(self):
        print("🌱 [SlowBioThread] Ciclos de Metabolismo y Homeostasis por Hardware.")
        last_bio_tick = time.time()
        stress = 0.0
        impact = {"stress_factor": 0.0, "is_starving": False}
        MUTATION_STRESS_THRESHOLD = 0.85
        EVAL_WINDOW = 100

        self.sleep_timer = 0.0
        SLEEP_DURATION_FIXED = 10.0

        while self.is_running:
            now = time.time()
            if now - last_bio_tick >= 2.0:
                try:
                    if hasattr(self.metabolism, 'get_biological_impact'):
                        impact = self.metabolism.get_biological_impact()
                        stress = impact.get("stress_factor", 0.0)
                except Exception:
                    stress = 0.0

                with self.net.lock:
                    try:
                        dopamina_global = getattr(self.net, 'dopamine_level', 0.0)
    
                        indices_activos = np.where(self.net.fired_acumulado[:self.net.n] > 0)[0].tolist()[:200]
                        self.net.fired_acumulado[:self.net.n] = 0
    
                        self.output_module._snapshot_fired = indices_activos
                        self.output_module.evaluate(self.net, self.vision)
    
                        if self.state == "sleep":
                            self.sleep_timer += 2.0
                            if self.sleep_timer < SLEEP_DURATION_FIXED:
                                print(f"💤 [Sueño Profundo] Tiempo restante: {SLEEP_DURATION_FIXED - self.sleep_timer:.1f}s")
                            else:
                                self.state = "drowsy"
                                self.sleep_timer = 0.0
                        else:
                            if dopamina_global > 0.18: nuevo_estado = "awake"
                            elif dopamina_global > 0.14: nuevo_estado = "drowsy"
                            else: nuevo_estado = "sleep"
    
                            if nuevo_estado != self.state:
                                print(f"🧠 [Estado] {self.state.upper()} → {nuevo_estado.upper()} | Dopamina: {dopamina_global:.3f}")
                                self.state = nuevo_estado
    
                        if self.state == "sleep" and hasattr(self, 'output_module'):
                            if self.output_module.memoria_replay:
                                self.output_module.replay_rem(self.net, intensidad=0.25)
                            else:
                                print(f"💤 [REM] Estado Sleep, memoria vacía.")
    
                        env_sig = getattr(self.vision, 'last_activity', 0.1)
                        r_attr = 'neuron_region' if hasattr(self.net, 'neuron_region') else 'region'
                        indices = getattr(self.net, r_attr)
    
                        for region in list(self.net.regions.values()):
                            neurons_in_reg = [i for i in range(self.net.n) if indices[i] == region.id]
                            reg_activity = np.mean([self.net.fired[i] for i in neurons_in_reg]) if neurons_in_reg else 0.0
                            region.update_state(activity=reg_activity, output=getattr(region, 'output', 0.0), energy_input=0.08, env_signal=env_sig, net=self.net)
    
                            for sub in list(region.sub_regions.values()):
                                if sub.is_evaluating:
                                    sub.eval_steps_left -= 1
                                    if sub.eval_steps_left <= 0: sub.evaluate_jump(net=self.net)
                                elif sub.stress_level > MUTATION_STRESS_THRESHOLD:
                                    sub.initiate_speculative_jump(PROFILES, eval_window=EVAL_WINDOW)
    
                        # 5. METABOLISMO (delegado a MetabolismSystem)
                        if hasattr(self.metabolism, 'update'):
                            self.metabolism.update(self.net, self.net.current_time, dt=2000.0, state=self.state)
    
                        # 6. SISTEMAS Y RECOMPENSA (DOPAMINA)
                        if hasattr(self.net, 'update_spatial_regions'): self.net.update_spatial_regions()
                        if not impact.get("is_starving", False):
                            if hasattr(self.growth, 'apply'): self.growth.apply(self.net)
                            if hasattr(self.regions_sys, 'update'): self.regions_sys.update(self.net)
    
                        # --- FIX: STDP nunca corría en ningún hilo ---
                        if hasattr(self.plasticity, 'apply'):
                            self.plasticity.apply(self.net, self.net.current_time)
    
                        # metabolism.apply(...) sigue sin engancharse — ver nota abajo.
    
                        if hasattr(self.net, 'apply_dopamine_reward'):
                            last_score = getattr(self.vision, 'last_score', 0.0)
                            fovea_x, fovea_y = getattr(self.vision, 'fovea_center_x', 0.5), getattr(self.vision, 'fovea_center_y', 0.5)
                            prev_x, prev_y = getattr(self, '_slow_prev_x', fovea_x), getattr(self, '_slow_prev_y', fovea_y)
                            habituacion = 1.0 if math.sqrt((fovea_x - prev_x)**2 + (fovea_y - prev_y)**2) > 0.005 else 0.05
                            self._slow_prev_x, self._slow_prev_y = fovea_x, fovea_y
                            visual_success = last_score > 0.1
                            attn_data = getattr(self.attention, 'detected_coords', None)
                            if self.vision and attn_data is not None:
                                visual_success = visual_success or (self.vision.check_success(attn_data) if hasattr(self.vision, 'check_success') else False)
    
                            self.net.apply_dopamine_reward(visual_success=visual_success, habituacion=habituacion)

                        # --- NUEVO FIX: Reset de net.fired ---
                        # Se hace una sola vez por tick, DESPUÉS de que regiones,
                        # metabolismo y plasticidad ya lo leyeron. Sin esto, el STDP
                        # procesaba una lista infinita de neuronas.
                        self.net.fired[:self.net.n] = 0

                        # ------------------------------------------------------------------
                        # DIAGNÓSTICO DE PUERTOS Y BANDA Z (Fase A y B)
                        # ------------------------------------------------------------------
                        if hasattr(self, 'lang_module') and self.lang_module:
                            if hasattr(self.lang_module, 'debug_pool_status'):
                                self.lang_module.debug_pool_status(self.net)
                            if hasattr(self.lang_module, 'debug_band_activity'):
                                self.lang_module.debug_band_activity(self.net)
                            if hasattr(self.lang_module, 'debug_band_geometry'):
                                self.lang_module.debug_band_geometry(self.net)
                            if hasattr(self.lang_module, 'debug_overlap_evidencia_banda'):  
                                self.lang_module.debug_overlap_evidencia_banda(self.net)        

                    except Exception as e:
                        print(f"⚠️ Error en ciclo SlowBio: {e}")
    
                if hasattr(self.vision, '_dopamine_feedback'):
                    self.vision._dopamine_feedback = getattr(self.net, 'dopamine_level', 0.15)
    
                self.step_count += 1
                last_bio_tick = now
    
            time.sleep(max(0.1, self.metabolism.get_sleep_time(stress) if hasattr(self.metabolism, 'get_sleep_time') else 1.0))
            
    def run(self, screen_processor=None):
        if screen_processor is not None:
            self.scr_processor = screen_processor
        self.is_running = True

        from engine.thread_manager import ThreadManager
        import cv2  # Agregamos cv2 aquí por si no está importado a nivel global

        manager = ThreadManager(self)
        manager.start()

        print("🚀 [Main] Sistema en ejecución | i5-6th Gen Optimized.")
        print("🧠 Hilos de Visión, Spiking y Metabolismo delegados al ThreadManager.")
    
        last_heartbeat = time.time()
    
        try:
            while self.is_running:
                t_now = time.time()
                if t_now - last_heartbeat >= 5.0:
                    with self.net.lock:
                        avg_energy = np.mean([r.energy for r in self.net.regions.values()]) if self.net.regions else 0.0
                        v_x = self.vision.coord_x_port.value if self.vision else 0.5
                        v_y = self.vision.coord_y_port.value if self.vision else 0.5
                        look_pos = [round(v_x, 2), round(v_y, 2)]
                        dopamine = getattr(self.net, 'dopamine_level', 0.0)
                        
                        # --- NUEVO: Cálculo y Persistencia del Fitness ---
                        # Recuperamos el puntaje visual del módulo de visión
                        vision_score = getattr(self.vision, 'last_score', 0.0)
                        
                        # Fórmula de Fitness: Mayor peso a la dopamina/visión (novedad) frente a la energía
                        current_fitness = (dopamine * 2.5) + (vision_score * 1.5) + (avg_energy * 0.5)
                        
                        # Lo guardamos en self.net para que el método save() lo empaquete automáticamente
                        self.net.fitness_score = current_fitness

                    # Actualizamos el print para incluir el Fitness
                    print(f"⏱️ Heartbeat | Red: {self.net.n}n | Energía: {avg_energy:.2f} | Dopamina: {dopamine:.2f} | Fitness: {current_fitness:.2f} | Mirada: {look_pos}")
                    last_heartbeat = t_now

                # =======================================================
                # === FASE 4: CAPTURA DE TECLADO ===
                # =======================================================
                # Dentro del while:
                if msvcrt.kbhit():
                    tecla = msvcrt.getch().decode('utf-8').lower()
                    if tecla == 'x':
                        print("\n🎯 [Intervención] Letra X")
                        self.trigger_letter_training("X")
                    elif tecla == 'o':
                        print("\n🎯 [Intervención] Letra O")
                        self.trigger_letter_training("O")
                    elif tecla == 'q':
                        self.is_running = False
                    elif tecla == 'r':  # NUEVO: recablear evidencia a la banda viva
                        print("\n🔌 [Intervención] Recableo manual de evidencia_X/O")
                        if hasattr(self.lang_module, 'recable_from_live_band'):
                            self.lang_module.recable_from_live_band(self.net)    


        except KeyboardInterrupt:
            print("\n🛑 Interrupción detectada. Deteniendo simulación...")
        except Exception as e:
            print(f"❌ Error en el bucle principal: {e}")
        finally:
            self.is_running = False
            manager.stop()
            if hasattr(self, 'output_module') and self.output_module:
                self.output_module.cerrar_sesion(self.net)
            print("💾 Guardando estado de la red...")
            self.save()

    def save(self, filename="simulation_state"):
        """Guarda la red usando un método híbrido optimizado para bajo consumo de RAM."""
        try:
            print("📦 Iniciando secuencia de guardado (Método Híbrido)...")

            # Claves no serializables que siempre excluimos
            CLAVES_EXCLUIDAS = {
                'lock',          # RLock de la red
                'event_queue',   # Heap de eventos en vuelo
                '_vote_lock',    # Lock del sistema de votos (Fase 1)
            }

            # 1. Extraemos los puertos y módulos activos
            puertos_finales = []
            modulos_activos = {
                'Visión':   self.vision,
                'Atención': self.attention,
                'Corazón':  self.heart,
                'Lenguaje': getattr(self, 'lang_module', None),
                'Reward':   getattr(self, 'reward_system', None)
            }

            for mod_name, mod_obj in modulos_activos.items():
                if not mod_obj: continue
                if hasattr(mod_obj, 'auto_connect'):
                    mod_obj.auto_connect(self.net)

                info_cruda = mod_obj.get_ports_info()
                for p_info in info_cruda:
                    nuevo_puerto = {
                        'name':     clean_name(p_info['name']),
                        'pos':      p_info['pos'],
                        'strength': p_info.get('strength', 1.0),
                        'neurons':  []
                    }

                    real_port    = None
                    p_name_upper = p_info['name'].upper()

                    if mod_name == 'Corazón':
                        if any(k in p_name_upper for k in ['ESTRÉS', 'STRESS']):
                            real_port = getattr(mod_obj, 'stress_port', None)
                        elif any(k in p_name_upper for k in ['ENERGÍA', 'ENERGY']):
                            real_port = getattr(mod_obj, 'energy_port', None)
                        elif any(k in p_name_upper for k in ['PULSE', 'PULSO']):
                            real_port = getattr(mod_obj, 'pulse_port', None)
                    else:
                        partes    = p_info['name'].split()
                        p_id      = partes[-1] if len(partes) > 1 else p_info['name']
                        real_port = (getattr(mod_obj, 'input_ports',  {}).get(p_id) or
                                     getattr(mod_obj, 'output_ports', {}).get(p_id))

                        if not real_port:
                            all_ports = {
                                **getattr(mod_obj, 'input_ports',  {}),
                                **getattr(mod_obj, 'output_ports', {})
                            }
                            for name_key, port_obj in all_ports.items():
                                if p_id in name_key or name_key in p_id:
                                    real_port = port_obj
                                    break

                    if real_port:
                        raw_neurons = getattr(real_port, 'connected_neurons', [])
                        if isinstance(raw_neurons, (set, list, tuple, np.ndarray)):
                            nuevo_puerto['neurons'] = [int(idx) for idx in raw_neurons]
                        elif raw_neurons is not None:
                            try:
                                nuevo_puerto['neurons'] = [int(raw_neurons)]
                            except (TypeError, ValueError):
                                nuevo_puerto['neurons'] = []

                    puertos_finales.append(nuevo_puerto)

            # === Estado exacto de módulos — la fuente real para restaurar wiring.
            modules_state = {}
            with self.net.lock:
                for mod_name, mod_obj in self.net.modules.items():
                    if hasattr(mod_obj, 'get_state') and hasattr(mod_obj, 'input_ports'):
                        modules_state[mod_name] = mod_obj.get_state()

            # 2. SEPARACIÓN HÍBRIDA: Numpy vs Pickle
            arrays_np = {}
            dict_meta = {}

            with self.net.lock:
                for clave, valor in self.net.__dict__.items():

                    # Excluir locks y objetos no serializables
                    if clave in CLAVES_EXCLUIDAS:
                        continue

                    # NUEVO: Excluir módulos dinámicamente usando isinstance
                    if isinstance(valor, Module) or clave == 'modules':
                        continue

                    if isinstance(valor, np.ndarray):
                        arrays_np[clave] = valor
                    else:
                        # Intentar detectar otros locks anidados antes de agregar
                        try:
                            pickle.dumps(valor)
                            dict_meta[clave] = valor
                        except Exception:
                            # Si no se puede serializar, lo saltamos silenciosamente
                            continue

            # Guardamos matrices comprimidas
            np.savez_compressed(f"{filename}_matrices.npz", **arrays_np)

            # 3. Metadata y genética de regiones
            evolution_map = {
                region.id: {
                    sub_id: sub.best_profile.name
                    for sub_id, sub in region.sub_regions.items()
                }
                for region in self.net.regions.values()
            }

            data_final = {
                'network_meta':  dict_meta,
                'evolution_map': evolution_map,
                'ports':         puertos_finales,
                'modules_state': modules_state,
                'step':          self.step_count,
                'current_time':  getattr(self.net, 'current_time', 0),
                'version':       "1.7.0-Fase6-ModulePersistence"
            }

            with open(f"{filename}_meta.pkl", "wb") as f:
                pickle.dump(data_final, f, protocol=4)

            print("-" * 30)
            print("💾 PERSISTENCIA HÍBRIDA EXITOSA")
            cables = sum(1 for p in puertos_finales if p['neurons'])
            print(f"-> Archivos: .npz (Cerebro) + .pkl (Genética/Puertos/Módulos)")
            print(f"-> Puertos con cables útiles: {cables}/{len(puertos_finales)}")
            print("-" * 30)

        except Exception as e:
            print(f"❌ Error crítico en save(): {e}")


    
    def load_state(self, filename="simulation_state"):
        """Revive la red biológica preservando el continuo temporal."""
        file_meta = f"{filename}_meta.pkl"
        file_np   = f"{filename}_matrices.npz"

        import os
        if not os.path.exists(file_meta) or not os.path.exists(file_np):
            print("⚠️ [Persistencia] Archivos no encontrados. Iniciando red virgen.")
            return False

        try:
            import pickle
            import numpy as np
            import threading
            import heapq

            # 1. Leer Metadatos
            with open(file_meta, "rb") as f:
                data = pickle.load(f)

            # Restaurar escalares y estado general
            for k, v in data['network_meta'].items():
                setattr(self.net, k, v)

            self.step_count = data.get('step', 0)

            # 2. Leer e Inyectar Matrices
            np_data = np.load(file_np)
            for k in np_data.files:
                if hasattr(self.net, k):
                    setattr(self.net, k, np_data[k])

            # 3. Reconstruir Perfiles Evolutivos
            evo_map = data.get('evolution_map', {})
            try:
                from core.evolution_config import EVO_MENU
                for r_id, subs in evo_map.items():
                    if int(r_id) in self.net.regions:
                        region = self.net.regions[int(r_id)]
                        for sub_id, profile_name in subs.items():
                            if int(sub_id) in region.sub_regions:
                                prof_obj = EVO_MENU.get(profile_name)
                                if prof_obj:
                                    region.sub_regions[int(sub_id)].current_profile = prof_obj
            except Exception as e:
                print(f"⚠️ Detalle al restaurar perfiles: {e}")

            # =====================================================================
            # 4. DESFIBRILACIÓN (Recrear todo lo que no se puede guardar en pickle)
            # =====================================================================

            # 4.1 Locks de la red
            self.net.lock = threading.RLock()

            # 4.2 Lock de votos del sistema visual (Fase 1)
            if hasattr(self.net, 'vision') and self.net.vision:
                if not hasattr(self.net.vision, '_vote_lock') or \
                   self.net.vision._vote_lock is None:
                    self.net.vision._vote_lock = threading.Lock()
                # Resetear votos acumulados para arrancar limpio
                self.net.vision._vote_x = 0.0
                self.net.vision._vote_y = 0.0
                print("🔒 [Desfibrilación] _vote_lock recreado en VisionModule.")

            # 4.3 Cola de eventos limpia
            self.net.event_queue = []
            heapq.heapify(self.net.event_queue)

            # 4.4 Re-sincronizar relojes de módulos
            if hasattr(self.net, 'vision') and self.net.vision:
                self.net.vision.last_time = self.net.current_time
            if hasattr(self.net, 'heart') and self.net.heart:
                self.net.heart.last_pulse_time = self.net.current_time

            # ==========================================
            # 4.5 ELECTROSHOCK (NUEVO)
            # ==========================================
            # La cola está vacía. Si no inyectamos eventos ahora, la red no arranca.
            print("⚡ [Desfibrilación] Aplicando electroshock inicial para reactivar cascada de spikes...")
            if hasattr(self.net, 'heart') and getattr(self.net.heart, 'pacemaker_neurons', None):
                # Le damos un chispazo a los primeros 20 marcapasos
                for n_idx in list(self.net.heart.pacemaker_neurons)[:20]:
                    heapq.heappush(self.net.event_queue, (self.net.current_time + 0.001, int(n_idx), 5.0))
            else:
                # Fallback: chispazo genérico si no se encuentra el corazón
                for n_idx in range(20):
                    heapq.heappush(self.net.event_queue, (self.net.current_time + 0.001, n_idx, 5.0))
            # =====================================================================
            # 5. RESTAURAR WIRING DE MÓDULOS 
            # =====================================================================
            modules_state = data.get('modules_state', {})
            if modules_state:
                for mod_name, state in modules_state.items():
                    mod_obj = self.net.modules.get(mod_name)
                    if mod_obj is not None and hasattr(mod_obj, 'restore_state'):
                        mod_obj.restore_state(state, max_idx=self.net.n)
                print(f"🔌 [Persistencia] Wiring restaurado: {list(modules_state.keys())}")
            else:
                print("⚠️ [Persistencia] Guardado sin 'modules_state' (versión vieja) — "
                      "los módulos arrancan con el wiring de auto_connect().")

            # --- NUEVO: Recuperar y Mostrar el Fitness Histórico ---
            historical_fitness = getattr(self.net, 'fitness_score', 0.0)    

            print(
                f"✅ [Persistencia] Red revivida en "
                f"T={self.net.current_time:.2f}s | "
                f"Dopamina: {getattr(self.net, 'dopamine_level', 0.0):.2f} | "
                f"Fitness: {historical_fitness:.2f}"
            )
            
            return True
        
        except Exception as e:
            print(f"❌ [Persistencia] Error masivo en load_state: {e}")
            return False

# engine/thread_manager.py
import threading
import time
import logging

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(threadName)s] %(message)s'
)

class ThreadManager:
    def __init__(self, engine):
        """
        Orquestador de la Fase 3/4. 
        Maneja la separación de procesos rápidos (Ignición) y lentos (Metabolismo).
        """
        self.engine = engine
        self.threads = {}
        self._stop_event = threading.Event()
        self.lock = threading.Lock()
        
        # Mapeo directo de los procesos del motor
        self.thread_defs = {
            "VisionThread":  self.engine.worker_vision,
            "SpikingThread": self.engine.worker_spiking,
            "SlowBioThread": self.engine.worker_slow,
        }

    def start(self):
        self.engine.is_running = True
        self._stop_event.clear()

        def wrapper(target, name):
            """Envuelve la ejecución y maneja autorecuperación protegiendo la CPU."""
            while self.engine.is_running and not self._stop_event.is_set():
                try:
                    logging.info(f"🚀 Iniciando ciclo de {name}...")
                    target()
                    
                    if self.engine.is_running:
                        logging.warning(f"⚠️ {name} finalizó inesperadamente sin lanzar errores.")
                        time.sleep(0.5)  # Pausa de seguridad anti-bucle loco
                    else:
                        break  # Apagado controlado
                        
                except Exception as e:
                    logging.error(f"🚨 Error crítico en {name}: {e}", exc_info=True)
                    
                    # Mitigación térmica/recursos para el i5: Evitar reinicios en ráfaga
                    if self.engine.is_running:
                        logging.info(f"♻️ Intentando reanimar {name} en 1.5 segundos...")
                        time.sleep(1.5)
                    else:
                        break

        threads_to_start = []
        with self.lock:
            self.threads.clear() 
            
            for name, target in self.thread_defs.items():
                t = threading.Thread(
                    target=wrapper, 
                    args=(target, name), 
                    name=name, 
                    daemon=True
                )
                self.threads[name] = t
                threads_to_start.append(t)

        # Arrancamos los hilos fuera del lock para evitar condiciones de carrera (Deadlocks)
        for t in threads_to_start:
            t.start()

        logging.info(f"✅ Arquitectura v3.1 iniciada: {len(threads_to_start)} hilos protegidos.")

    def stop(self):
        """Detiene la simulación de forma segura y libera recursos de inmediato."""
        logging.info("🛑 Iniciando protocolo de apagado de hilos...")
        
        self.engine.is_running = False
        self._stop_event.set()  # Despierta hilos durmientes o bloqueados

        with self.lock:
            threads_copy = list(self.threads.items())

        # Margen para que los hilos cierren sus bucles internos
        for name, t in threads_copy:
            if t.is_alive():
                logging.info(f"⌛ Esperando a {name}...")
                t.join(timeout=2.5)  # Margen para vaciar colas de spikes
                if t.is_alive():
                    logging.warning(f"⚠️ {name} no respondió a tiempo, se forzará su cierre vía Daemon.")

        with self.lock:
            self.threads.clear()

        logging.info("✅ Simulación detenida. CPU y memoria liberadas.")

    def monitor_health(self):
        """Verifica si los hilos críticos siguen operativos."""
        if not self.engine.is_running:
            return False
            
        dead_threads = []
        with self.lock:
            for name, t in self.threads.items():
                if not t.is_alive():
                    dead_threads.append(name)
        
        if dead_threads:
            logging.error(f"🚨 Hilos críticos caídos en ejecución: {dead_threads}")
            return False
        return True

    def is_alive(self):
        with self.lock:
            return any(t.is_alive() for t in self.threads.values()) if self.threads else False

    @property
    def active_count(self):
        with self.lock:
            return len([t for t in self.threads.values() if t.is_alive()])

    def get_performance_snapshot(self):
        return {
            "threads_active": self.active_count,
            "engine_running": self.engine.is_running,
            "status": "HEALTHY" if self.monitor_health() else "CRITICAL"
        }


# modules/vision.py


import threading
import math
import numpy as np
import random
import logging
from core.module import Module

class VisionModule(Module):
    def __init__(self, net=None):
        super().__init__("Visión")

        # Puertos principales
        self.coord_x_port      = self.add_input_port("coord_x",      pos=(40.0,  75.0, 35.0))
        self.coord_y_port      = self.add_input_port("coord_y",      pos=(110.0, 75.0, 35.0))
        self.fovea_radius_port = self.add_input_port("fovea_radius", pos=(75.0,  75.0, 35.0))

        # --- FASE 1: Acumuladores de votos musculares (thread-safe) ---
        self._vote_x    = 0.0
        self._vote_y    = 0.0
        self._vote_lock = threading.Lock()
        # ---------------------------------------------------------------

        # Física ocular
        self.fovea_center_x = 0.5
        self.fovea_center_y = 0.5
        self.vel_x          = 0.0
        self.vel_y          = 0.0
        self.motor_mass     = 8.0
        self.friction       = 0.92


        # Límites biológicos
        self.motor_mass_min = 2.0    # Ojo muy ágil (para objetos rápidos)
        self.motor_mass_max = 12.0   # Ojo muy estable (para objetos lentos)
        self.friction_min   = 0.75   # Frena rápido
        self.friction_max   = 0.96   # Mantiene inercia

        self._last_fovea_x = 0.5
        self._last_fovea_y = 0.5

        # Valores iniciales de puertos
        self.coord_x_port.value      = 0.0
        self.coord_y_port.value      = 0.0
        self.fovea_radius_port.value = 0.15

        # Estado
        self.last_activity       = 0.0
        self.last_activity_coords = None
        self.current_activations  = []
        self.is_calibrating       = True
        self.boot_timer           = 0
        self.calibration_duration = 400

    # ------------------------------------------------------------------
    # CONEXIÓN
    # ------------------------------------------------------------------

    def auto_connect(self, net):
        """Conecta los puertos a neuronas cercanas."""
        puertos = [self.coord_x_port, self.coord_y_port, self.fovea_radius_port]
        for port in puertos:
            if len(port.connected_neurons) < 12:
                dist    = np.linalg.norm(net.positions - np.array(port.pos), axis=1)
                closest = np.argsort(dist)[:25]
                for idx in closest:
                    if idx not in port.connected_neurons:
                        port.connected_neurons.append(int(idx))
        print("🔌 [Visión] Puertos auto-conectados.")

    # ------------------------------------------------------------------
    # FÍSICA OCULAR - DUEÑO ÚNICO DE coord_x/y_port.value
    # ------------------------------------------------------------------

    def _compute_centering_force(self, x, y):
        """
        FASE 2: Campo de fuerza de resorte hacia el centro.
        Suave en el medio, fuerte cerca de los bordes.
        """
        force_x = (0.5 - x) * 0.03   # 3% de la distancia al centro
        force_y = (0.5 - y) * 0.03
        return force_x, force_y

    def update_fovea_physics(self, dt=0.016):
        """
        DUEÑO ÚNICO de fovea_center_x/y y coord_x/y_port.value.
        Se llama exclusivamente desde worker_vision.
        Integra Fase 1 (votos) + Fase 2 (resorte central).
        """
        # 1. Leer y resetear votos acumulados por worker_spiking
        with self._vote_lock:
            vote_x = self._vote_x
            vote_y = self._vote_y
            self._vote_x = 0.0
            self._vote_y = 0.0

        # 2. Fuerza muscular desde votos
        force_x = vote_x * 0.01
        force_y = vote_y * 0.01

        # 3. FASE 2: Sumar campo de resorte central
        center_fx, center_fy = self._compute_centering_force(
            self.fovea_center_x,
            self.fovea_center_y
        )
        force_x += center_fx
        force_y += center_fy

        # 4. Física: aceleración, inercia y fricción
        acc_x = force_x / self.motor_mass
        acc_y = force_y / self.motor_mass

        self.vel_x = (self.vel_x + acc_x * dt) * self.friction
        self.vel_y = (self.vel_y + acc_y * dt) * self.friction

        # 5. Clamp de velocidad máxima (evita inercia infinita hacia bordes)
        max_vel    = 0.03
        self.vel_x = max(-max_vel, min(max_vel, self.vel_x))
        self.vel_y = max(-max_vel, min(max_vel, self.vel_y))

        # 6. Mover la fóvea real
        self.fovea_center_x = max(0.05, min(0.95, self.fovea_center_x + self.vel_x))
        self.fovea_center_y = max(0.05, min(0.95, self.fovea_center_y + self.vel_y))

        # 7. Publicar posición en los puertos
        # ÚNICO PUNTO DE ESCRITURA de coord_x/y_port.value en todo el sistema
        self.coord_x_port.value = self.fovea_center_x
        self.coord_y_port.value = self.fovea_center_y
        
        velocidad_actual = math.sqrt(self.vel_x**2 + self.vel_y**2)
    
        if hasattr(self, '_dopamine_feedback'):
            if self._dopamine_feedback < 0.12 and velocidad_actual > 0.01:
                # El ojo se mueve pero no encuentra nada: objetivo escapando
                # Volverse más ágil
                self.motor_mass = max(self.motor_mass_min, 
                                      self.motor_mass * 0.995)
                self.friction   = max(self.friction_min,   
                                      self.friction   * 0.999)
            elif self._dopamine_feedback > 0.20:
                # Trackeando bien: ganar estabilidad gradualmente
                self.motor_mass = min(self.motor_mass_max, 
                                      self.motor_mass * 1.002)
                self.friction   = min(self.friction_max,   
                                      self.friction   * 1.001)


    # ------------------------------------------------------------------
    # REFLEJO DE DOLOR
    # ------------------------------------------------------------------

    def _disparar_reflejo_dolor(self, eje, culpables):
        """
        Hiperpolariza las neuronas motoras responsables del choque
        para apagarlas instantáneamente por sobreestimulación dolorosa.
        """
        puerto = self.coord_x_port if eje == 'x' else self.coord_y_port
        indices_motoras = getattr(puerto, 'connected_neurons', [])

        if not indices_motoras:
            return

        if culpables == 'positivas':
            culpables_list = [idx for idx in indices_motoras if idx % 2 == 0]
        else:
            culpables_list = [idx for idx in indices_motoras if idx % 2 != 0]

        if hasattr(self, 'net') and hasattr(self.net, 'membrane_potential'):
            for idx in culpables_list:
                try:
                    self.net.membrane_potential[idx] = -90.0
                except Exception:
                    pass

        if hasattr(self, 'net') and hasattr(self.net, 'dopamine_level'):
            self.net.dopamine_level = max(0.0, self.net.dopamine_level - 0.15)

        logging.warning(
            f"⚡ [Reflejo Dolor] Choque en Eje {eje.upper()}. "
            f"{len(culpables_list)} neuronas inhibidas a -90mV."
        )

    # ------------------------------------------------------------------
    # CALIBRACIÓN
    # ------------------------------------------------------------------

    def run_calibration_sequence(self):
        if self.boot_timer < self.calibration_duration:
            t = self.boot_timer * 0.05
            self.fovea_center_x      = 0.5 + 0.25 * math.cos(t)
            self.fovea_center_y      = 0.5 + 0.15 * math.sin(2 * t)
            self.fovea_radius_port.value = 0.15 + 0.10 * math.sin(t * 0.5)
            self.last_activity       = 1.0
            self.boot_timer         += 1
        else:
            self.is_calibrating = False

    def update(self, net, t):
        if self.is_calibrating:
            self.run_calibration_sequence()
            return
        self.fovea_radius_port.value = np.clip(self.fovea_radius_port.value, 0.08, 0.45)

    # ------------------------------------------------------------------
    # TRANSPORTE A LA RED
    # ------------------------------------------------------------------

    def transport_to_net(self, net, t):
        """Envía actividad visual a la red. Sin llamadas a dopamina (worker_slow lo hace)."""
        if not hasattr(self, 'current_activations') or not self.current_activations:
            return

        safe_activations = []
        total_intensity  = 0.0

        for act in self.current_activations:
            try:
                pos = act.get('pos', (0.5, 0.5, 60))
                val = act.get('intensity', 1.0)
                safe_activations.append((pos[0], pos[1], pos[2], val))
                total_intensity += val
            except Exception:
                continue

        self.last_activity = total_intensity

        if safe_activations:
            net.inject_sensory_activity(safe_activations)

        # Guardar posición para el siguiente frame
        self._last_fovea_x = self.fovea_center_x
        self._last_fovea_y = self.fovea_center_y

    def update_from_external(self, coords):
        self.last_activity_coords = coords
        self.current_activations = []
        for item in coords:
            try:
                if len(item) >= 4:
                    x, y, p_type, val = float(item[0]), float(item[1]), int(item[2]), float(item[3])
                elif len(item) == 3:  # compatibilidad con el formato viejo
                    x, y, val, p_type = float(item[0]), float(item[1]), float(item[2]), 0
                else:
                    continue
                z = 45.0 if p_type == 1 else 30.0  # separa canales, lejos del corazón (z=60)
                self.current_activations.append({'pos': (x, y, z), 'intensity': val})
            except:
                continue

    # ------------------------------------------------------------------
    # ÉXITO Y PUERTOS
    # ------------------------------------------------------------------

    def check_success(self, attention_coords):
        """Usado por Reward System."""
        if attention_coords is None or len(attention_coords) < 2:
            return False
        try:
            current = np.array([self.fovea_center_x, self.fovea_center_y])
            target  = np.array(attention_coords[:2])
            dist    = np.linalg.norm(current - target)
            fovea_r = getattr(self.fovea_radius_port, 'value', 0.15)
            return dist < (fovea_r * 1.15)
        except Exception:
            return False

    def get_ports_info(self):
        return [
            {
                "name": "V_coord_x",
                "pos":  self.coord_x_port.pos,
                "value": float(self.coord_x_port.value),
                "connected_neurons": list(self.coord_x_port.connected_neurons)
            },
            {
                "name": "V_coord_y",
                "pos":  self.coord_y_port.pos,
                "value": float(self.coord_y_port.value),
                "connected_neurons": list(self.coord_y_port.connected_neurons)
            },
            {
                "name": "V_fovea_radius",
                "pos":  self.fovea_radius_port.pos,
                "value": float(self.fovea_radius_port.value),
                "connected_neurons": list(self.fovea_radius_port.connected_neurons)
            }
        ]


# =============================================================================
# 🧠 FLUJO BIOLÓGICO ACTUALIZADO - Fases 1 y 2
# =============================================================================
# worker_spiking   → detecta spike motor → acumula _vote_x / _vote_y (con _vote_lock)
#
# update_fovea_physics (llamado por worker_vision cada frame):
#   1. Consume y resetea votos
#   2. Suma campo de resorte central (Fase 2): fuerza proporcional a distancia al centro
#   3. Aplica física: aceleración → fricción → clamp de velocidad
#   4. Mueve fovea_center_x/y
#   5. Publica en coord_x/y_port.value  ← ÚNICO PUNTO DE ESCRITURA
#
# worker_vision    → lee fovea_center_x/y para captura de pantalla (nunca escribe puertos)
# worker_slow      → llama apply_dopamine_reward una vez cada 2s (no transport_to_net)
# =============================================================================


# modules/output_module.py

"""
OutputModule - Sistema de salida y clasificación de detecciones
===============================================================

FÓRMULA DE CERTEZA:
    certeza = (dopamina_promedio * 0.4) + (consenso_regiones * 0.4) + (tiempo_fijacion * 0.2)

    - dopamina_promedio  : Promedio de dopamina de todas las regiones activas (0.0 a 1.0)
    - consenso_regiones  : Proporción de regiones con dopamina > 0.20 simultáneamente (0.0 a 1.0)
    - tiempo_fijacion    : Segundos consecutivos mirando la misma zona, normalizado a 30s max.

UMBRAL DE REGISTRO: 0.45

REPLAY REM:
    Durante el estado "sleep" del worker_slow, la red reproduce internamente
    los patrones de activación de las mejores detecciones de la sesión.
    Las detecciones con mayor certeza se replayan más frecuentemente.
    Esto refuerza las sinapsis que llevaron al éxito sin necesidad de ver
    el objeto de nuevo, equivalente al sueño REM biológico.
"""

import json
import math
import time
import logging
import numpy as np
from datetime import datetime


class OutputModule:
    def __init__(self, umbral_certeza=0.45, output_path="detecciones.json"):
        self.umbral_certeza = umbral_certeza
        self.min_umbral = 0.15 # Umbral mínimo de emergencia
        self.output_path    = output_path

        # --- Control de tiempo de fijación ---
        self._tiempo_inicio_fijacion = time.time()
        self._zona_fijacion_x        = 0.5
        self._zona_fijacion_y        = 0.5
        self._max_segundos_ref       = 30.0

        # --- Anti-spam ---
        self._last_detection_x = -1.0
        self._last_detection_y = -1.0
        self._min_dist_nueva   = 0.08

        # ================================================================
        # BUFFER DE MEMORIA PARA REPLAY REM
        # Guarda las mejores detecciones de la sesión con sus patrones
        # de activación neuronal para reproducirlos durante el sueño.
        # ================================================================
        self.memoria_replay = []          # Lista de experiencias exitosas
        self.max_memoria    = 50          # Máximo de experiencias guardadas
        self._replay_idx    = 0           # Índice circular para replay

        self._inicializar_archivo()
        print(f"📋 [OutputModule] Iniciado | Umbral: {umbral_certeza} | "
              f"Archivo: {output_path} | Memoria REM: {self.max_memoria} slots")

    # ------------------------------------------------------------------
    # INICIALIZACIÓN
    # ------------------------------------------------------------------

    def _inicializar_archivo(self):
        sesion = {
            "sesion_inicio": datetime.now().isoformat(),
            "umbral_certeza": self.umbral_certeza,
            "formula_certeza": {
                "descripcion": (
                    "certeza = (dopamina_promedio * 0.4) "
                    "+ (consenso_regiones * 0.4) "
                    "+ (tiempo_fijacion * 0.2)"
                ),
                "dopamina_promedio": (
                    "Promedio de dopamina de todas las regiones activas, "
                    "normalizado al rango real observado (0.0 - 0.5)"
                ),
                "consenso_regiones": (
                    "Proporcion de regiones con dopamina > 0.20 simultaneamente."
                ),
                "tiempo_fijacion": (
                    "Segundos consecutivos con el ojo en la misma zona (radio < 0.03), "
                    "normalizado a 30 segundos maximos."
                )
            },
            "detecciones": []
        }
        try:
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(sesion, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"❌ [OutputModule] Error creando archivo: {e}")

    # ------------------------------------------------------------------
    # CÁLCULO DE CERTEZA
    # ------------------------------------------------------------------

    def _calcular_certeza(self, net, fovea_x, fovea_y):
        dopaminas = [r.dopamine for r in net.regions.values() if r.is_active]
        if not dopaminas:
            return 0.0, {}

        dop_promedio = sum(dopaminas) / len(dopaminas)
        dop_norm     = min(1.0, dop_promedio / 0.5)

        umbral_dop_region = 0.20
        regiones_activas  = sum(1 for d in dopaminas if d > umbral_dop_region)
        consenso          = regiones_activas / len(dopaminas)

        dist_zona = math.sqrt((fovea_x - self._zona_fijacion_x)**2 +
                              (fovea_y - self._zona_fijacion_y)**2)

        if dist_zona < 0.03:
            segundos_fijo = time.time() - self._tiempo_inicio_fijacion
        else:
            self._tiempo_inicio_fijacion = time.time()
            self._zona_fijacion_x        = fovea_x
            self._zona_fijacion_y        = fovea_y
            segundos_fijo                = 0.0

        tiempo_norm = min(1.0, segundos_fijo / self._max_segundos_ref)
        certeza     = (dop_norm * 0.4) + (consenso * 0.4) + (tiempo_norm * 0.2)

        desglose = {
            "dopamina_promedio":  round(dop_promedio, 4),
            "dopamina_norm":      round(dop_norm, 4),
            "consenso_regiones":  round(consenso, 4),
            "regiones_activas":   regiones_activas,
            "total_regiones":     len(dopaminas),
            "segundos_fijos":     round(segundos_fijo, 2),
            "tiempo_fijacion":    round(tiempo_norm, 4)
        }

        return round(certeza, 4), desglose

    # ------------------------------------------------------------------
    # EVALUACIÓN Y REGISTRO
    # ------------------------------------------------------------------

    def _es_zona_nueva(self, x, y):
        dist = math.sqrt((x - self._last_detection_x)**2 +
                         (y - self._last_detection_y)**2)
        return dist >= self._min_dist_nueva

    def evaluate(self, net, vision):
        try:
            fovea_x = getattr(vision, 'fovea_center_x', 0.5)
            fovea_y = getattr(vision, 'fovea_center_y', 0.5)

            certeza, desglose = self._calcular_certeza(net, fovea_x, fovea_y)
            indices_activos = getattr(self, '_snapshot_fired', [])

            # AJUSTE DINÁMICO: Umbral flexible si la memoria está empezando
            # Si tenemos menos de 5 experiencias, bajamos la exigencia para iniciar el REM
            umbral_dinamico = self.min_umbral if len(self.memoria_replay) < 5 else self.umbral_certeza

            # 1. SIEMPRE INTENTAMOS GUARDAR EN MEMORIA SI HAY ACTIVIDAD
            if indices_activos and certeza >= umbral_dinamico:
                experiencia = {
                    "coordenadas": (round(fovea_x, 4), round(fovea_y, 4)),
                    "certeza": certeza,
                    "indices_neuronales": indices_activos,
                    "dopamina_snapshot": round(net.dopamine_level, 4)
                }

                if len(self.memoria_replay) < self.max_memoria:
                    self.memoria_replay.append(experiencia)
                else:
                    idx_min = min(range(len(self.memoria_replay)), key=lambda i: self.memoria_replay[i]["certeza"])
                    if certeza > self.memoria_replay[idx_min]["certeza"]:
                        self.memoria_replay[idx_min] = experiencia

            # 2. FILTRO DE SPAM PARA EL ARCHIVO JSON
            # También usamos el umbral dinámico aquí para que el log/archivo se llene 
            # al menos un poco durante la fase de "aprendizaje inicial"
            if certeza < umbral_dinamico or not self._es_zona_nueva(fovea_x, fovea_y):
                return

            # Si pasa los filtros, escribimos en el JSON
            deteccion = {
                "timestamp": datetime.now().isoformat(),
                "coordenadas": {"x": round(fovea_x, 4), "y": round(fovea_y, 4)},
                "certeza": certeza,
                "desglose": desglose,
                "neuronas_activas": int(sum(net.active[:net.n])),
                "energia_red": round(float(sum(r.energy for r in net.regions.values()) / max(1, len(net.regions))), 4)
            }

            self._escribir_deteccion(deteccion)
            self._last_detection_x = fovea_x
            self._last_detection_y = fovea_y

            # Log informativo
            print(f"📍 [Detección] Certeza: {certeza:.2f} (Umbral: {umbral_dinamico:.2f}) | Mem REM: {len(self.memoria_replay)}/{self.max_memoria}")

        except Exception as e:
            logging.error(f"❌ [OutputModule] Error en evaluate(): {e}")
    # ------------------------------------------------------------------
    # REPLAY REM
    # ------------------------------------------------------------------

    def replay_rem(self, net, intensidad=0.3):
        """
        Reproduce una experiencia exitosa del pasado inyectando spikes
        suaves en las neuronas que participaron en esa detección.

        Llamar desde worker_slow cuando state == 'sleep'.
        La intensidad controla qué tan fuerte es el replay (0.1 a 1.0).
        Las experiencias con mayor certeza se replayan con más fuerza.

        Retorna True si hubo replay, False si no había memoria.
        """
        if not self.memoria_replay:
            return False

        try:
            # Seleccionar experiencia priorizando las de mayor certeza
            # con un poco de aleatoriedad para no repetir siempre la misma
            import random

            # 70% probabilidad de elegir una de las top 5 experiencias
            memoria_ordenada = sorted(
                self.memoria_replay,
                key=lambda x: x["certeza"],
                reverse=True
            )

            if random.random() < 0.70 and len(memoria_ordenada) >= 1:
                top_n = min(5, len(memoria_ordenada))
                experiencia = random.choice(memoria_ordenada[:top_n])
            else:
                experiencia = random.choice(self.memoria_replay)

            indices    = experiencia["indices_neuronales"]
            certeza    = experiencia["certeza"]
            coordenadas = experiencia["coordenadas"]

            if not indices:
                return False

            # Fuerza del replay proporcional a la certeza de la experiencia
            fuerza_base = intensidad * certeza

            # Inyectar spikes suaves en las neuronas del patrón recordado
            t_actual = net.current_time
            inyectadas = 0

            for idx in indices:
                if idx >= net.n or not net.active[idx]:
                    continue

                # Solo inyectar si la neurona no está en periodo refractario
                if t_actual < net.refractory_until[idx]:
                    continue

                # Fuerza con pequeña variación aleatoria para naturalidad
                import random as r
                fuerza = fuerza_base * r.uniform(0.8, 1.2)

                net.receive_spike(idx, fuerza, t_actual)
                inyectadas += 1

            if inyectadas > 0:
                print(f"🧠 [REM] Replay certeza {certeza:.2f} | "
                      f"Zona ({coordenadas[0]:.2f}, {coordenadas[1]:.2f}) | "
                      f"{inyectadas} neuronas reactivadas")

            return inyectadas > 0

        except Exception as e:
            logging.error(f"❌ [OutputModule] Error en replay_rem(): {e}")
            return False

    # ------------------------------------------------------------------
    # ESCRITURA EN JSON
    # ------------------------------------------------------------------

    def _escribir_deteccion(self, deteccion):
        try:
            with open(self.output_path, "r", encoding="utf-8") as f:
                datos = json.load(f)

            datos["detecciones"].append(deteccion)
            datos["total_detecciones"] = len(datos["detecciones"])
            datos["ultima_deteccion"]  = deteccion["timestamp"]

            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(datos, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logging.error(f"❌ [OutputModule] Error escribiendo detección: {e}")

    # ------------------------------------------------------------------
    # CIERRE DE SESIÓN
    # ------------------------------------------------------------------

    def cerrar_sesion(self, net):
        try:
            with open(self.output_path, "r", encoding="utf-8") as f:
                datos = json.load(f)

            datos["sesion_fin"]        = datetime.now().isoformat()
            datos["neuronas_finales"]  = int(net.n)
            datos["total_detecciones"] = len(datos["detecciones"])

            if datos["detecciones"]:
                certezas = [d["certeza"] for d in datos["detecciones"]]
                datos["certeza_promedio"] = round(sum(certezas) / len(certezas), 4)
                datos["certeza_maxima"]   = round(max(certezas), 4)

                mejor = max(datos["detecciones"], key=lambda d: d["certeza"])
                datos["mejor_deteccion"] = {
                    "coordenadas": mejor["coordenadas"],
                    "certeza":     mejor["certeza"],
                    "timestamp":   mejor["timestamp"]
                }

            # Resumen de memoria REM
            datos["memoria_rem"] = {
                "experiencias_guardadas": len(self.memoria_replay),
                "certeza_promedio_rem": round(
                    sum(e["certeza"] for e in self.memoria_replay) /
                    max(1, len(self.memoria_replay)), 4
                ) if self.memoria_replay else 0.0
            }

            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(datos, f, indent=2, ensure_ascii=False)

            print(f"📋 [OutputModule] Sesión cerrada | "
                  f"{datos['total_detecciones']} detecciones | "
                  f"Mejor certeza: {datos.get('certeza_maxima', 0):.2f} | "
                  f"Memoria REM: {len(self.memoria_replay)} experiencias")

        except Exception as e:
            logging.error(f"❌ [OutputModule] Error cerrando sesión: {e}")


# languague_io.py

import numpy as np
from core.module import Module
from core.evolution_config import EVO_MENU

BUFFER_PROFILE = EVO_MENU["BUFFER"]

# --- NUEVO: Perfil RELAY para las conexiones visión->atención ---
# tau_m=20, v_thresh=18 (más fácil de disparar que el default 20)
# energy_drain=0.03 (mucho más bajo que RAPID_FIRE=0.12)
RELAY_PROFILE = EVO_MENU["RELAY"]

class LetterIOModule(Module):
    def __init__(self, net=None, symbols=("X", "O")):
        super().__init__("Lenguaje")
        self.symbols = list(symbols)
        n = len(self.symbols)
        self._buffer_pools = {}                
        self._assigned_buffer_neurons = set()   
        
        # --- Fase A ---
        self._prev_evidence     = {sym: 0.0 for sym in self.symbols}
        self._evidence_cooldown = {sym: False for sym in self.symbols}
        self.edge_threshold     = 0.15
        self.evidence_window_ms = 200.0   # ventana de actividad reciente; ajustable
        self._clean_trial_baseline_t = None
        
        for i, sym in enumerate(self.symbols):
            x = 65.0 + (i * (20.0 / max(1, n - 1))) if n > 1 else 75.0

            self.add_input_port(f"evidencia_{sym}", pos=(x, 75.0, 40.0))
            self.add_output_port(f"maestro_{sym}",  pos=(x, 90.0, 52.0))  # inyecta por receive_spike directo, no depende de distancia
            self.add_input_port(f"buffer_{sym}",    pos=(x, 75.0, 75.0))  # Fase B, no tocado hoy


    def protect_band(self, net, z_range=(37.0, 53.0)):
        """
        A diferencia de recable_from_live_band (que solo actúa sobre
        neuronas que YA demostraron estar vivas), esto le da un piso
        energético + perfil de bajo consumo a TODA la banda de lenguaje,
        para que tengan chance real de sobrevivir hasta recibir su
        primer disparo, en vez de morir de atrofia antes de que les
        llegue señal por primera vez.
        """
        z = net.positions[:net.n, 2]
        en_banda = np.where((z > z_range[0]) & (z < z_range[1]) & net.active[:net.n])[0]
        if len(en_banda) == 0:
            print("🔌 [Protección] Banda vacía — nada que proteger.")
            return

        for idx in en_banda:
            idx = int(idx)
            net.energy[idx] = max(float(net.energy[idx]), 1.0)
            net.tau_m[idx] = RELAY_PROFILE.tau_m
            net.v_thresh[idx] = RELAY_PROFILE.v_thresh
            net.refr_period[idx] = RELAY_PROFILE.refr_period

        print(f"🔌 [Protección] {len(en_banda)} neuronas de la banda reperfiladas y con energía asegurada ≥1.0")        

    
    def recable_from_live_band(self, net, z_range=(37.0, 53.0), max_per_letter=10):
        # --- NUEVO: proteger SIEMPRE primero, pase lo que pase después ---
        self.protect_band(net, z_range=z_range)

        z = net.positions[:net.n, 2]
        en_banda = (z > z_range[0]) & (z < z_range[1]) & net.active[:net.n]
        idx_banda = np.where(en_banda)[0]
        vivas = idx_banda[net.last_spike[idx_banda] > 0].tolist()

        if len(vivas) < len(self.symbols):
            print(f"🔌 [Recable] Solo {len(vivas)} neurona(s) viva(s) — insuficiente "
                  f"para {len(self.symbols)} letras sin que compartan la misma. "
                  f"(La banda ya quedó protegida para la próxima corrida.)")
            return

        import random
        random.shuffle(vivas)
        grupos = np.array_split(vivas, len(self.symbols))

        for sym, grupo in zip(self.symbols, grupos):
            grupo = [int(i) for i in grupo.tolist()][:max_per_letter]
            port = self.input_ports.get(f"evidencia_{sym}")
            if not port or not grupo: continue

            port.connected_neurons = grupo
            print(f"🔌 [Recable] evidencia_{sym} → {len(grupo)} neuronas disjuntas: {grupo}")


    def assign_buffer_pools(self, net, pool_size=12, search_radius=40.0):
        """Reserva un pool FIJO y EXCLUSIVO de neuronas por símbolo."""
        for sym in self.symbols:
            port = self.input_ports[f"buffer_{sym}"]
            port_pos = np.array(port.pos)
            dist  = np.linalg.norm(net.positions[:net.n] - port_pos, axis=1)
            orden = np.argsort(dist)
            
            pool = []
            for idx in orden:
                idx = int(idx)
                if idx in self._assigned_buffer_neurons or not net.active[idx]:
                    continue
                pool.append(idx)
                self._assigned_buffer_neurons.add(idx)
                if len(pool) >= pool_size:
                    break
                    
            self._buffer_pools[sym] = pool
            port.connected_neurons = pool
            
            # Reperfilado eléctrico
            for idx in pool:
                net.tau_m[idx]       = BUFFER_PROFILE.tau_m
                net.v_thresh[idx]    = BUFFER_PROFILE.v_thresh
                net.refr_period[idx] = BUFFER_PROFILE.refr_period
                
            print(f"🧠 [Buffer] Pool '{sym}': {len(pool)} neuronas reperfiladas")
    

    def debug_band_geometry(self, net, z_range=(37.0, 53.0),
                         injection_point=(75.0, 75.0, 40.0), injection_radius=45.0):
        """¿Están estas neuronas siquiera dentro del alcance físico de
        inject_sensory_activity? Mide distancia real, no supone nada."""
        z = net.positions[:net.n, 2]
        en_banda = (z > z_range[0]) & (z < z_range[1]) & net.active[:net.n]
        idx_banda = np.where(en_banda)[0]
        if len(idx_banda) == 0:
            print("🔬 [Diag-Geo] Banda vacía.")
            return

        pos = net.positions[idx_banda]
        target = np.array(injection_point)
        dist = np.linalg.norm(pos - target, axis=1)
        alcanzables = int(np.sum(dist <= injection_radius))

        print(f"🔬 [Diag-Geo] {len(idx_banda)} neuronas en banda | "
              f"dist. prom al punto de inyección={float(np.mean(dist)):.1f} | "
              f"mín={float(np.min(dist)):.1f} | máx={float(np.max(dist)):.1f} | "
              f"dentro del radio de inyección ({injection_radius}): {alcanzables}/{len(idx_banda)}")

    def auto_connect(self, net, radius=20.0):
        for name, port in {**self.input_ports, **self.output_ports}.items():
            if name.startswith("buffer_"):
                continue
            dist = np.linalg.norm(net.positions[:net.n] - np.array(port.pos), axis=1)
            for idx in np.argsort(dist)[:15]:
                if int(idx) not in port.connected_neurons:
                    port.connected_neurons.append(int(idx))

    # --- FUNCIONES FASE A ---

    def begin_clean_trial(self, t):
        """Fija el instante cero para que la evidencia ignore actividad previa."""
        self._clean_trial_baseline_t = float(t)
        for sym in self.symbols:
            port = self.input_ports.get(f"evidencia_{sym}")
            if port is not None:
                port.value = 0.0
        self._prev_evidence = {sym: 0.0 for sym in self.symbols}
        self._evidence_cooldown = {sym: False for sym in self.symbols}

    def end_clean_trial(self):
        """Finaliza el aislamiento de evidencia y elimina el baseline temporal."""
        self._clean_trial_baseline_t = None
        self._prev_evidence = {sym: 0.0 for sym in self.symbols}
        self._evidence_cooldown = {sym: False for sym in self.symbols}
        for sym in self.symbols:
            port = self.input_ports.get(f"evidencia_{sym}")
            if port is not None:
                port.value = 0.0
    
    def update_evidence_values(self, net, t):
        """A.1 — Da vida a evidencia_{sym}.value con filtro de arranque."""
        for sym in self.symbols:
            port = self.input_ports[f"evidencia_{sym}"]
            neurons = port.connected_neurons
            
            # --- Ignorar el primer medio segundo (Falso Positivo) ---
            if not neurons or t < 0.5:
                port.value = 0.0
                continue
            
            idx_arr = np.asarray(neurons, dtype=int)
            idx_arr = idx_arr[idx_arr < net.n]
            if len(idx_arr) == 0:
                port.value = 0.0
                continue
                
            last_spikes = net.last_spike[idx_arr]
            dts = t - last_spikes
            baseline_t = self._clean_trial_baseline_t
            if baseline_t is not None:
                # Solo cuentan spikes producidos desde el inicio del trial.
                valid = (last_spikes >= baseline_t) & (dts >= 0.0) & (dts < self.evidence_window_ms)
                activos = int(np.sum(valid))
            else:
                activos = int(np.sum(dts < self.evidence_window_ms))
            nuevo_valor = float(activos) / len(idx_arr)
            
            # Solo para debuggear: si hay alguito de actividad, pero es menor al umbral anterior,
            # lo imprimimos una sola vez cuando sube (usando un truco rápido)
            if nuevo_valor > 0.02 and nuevo_valor > port.value * 1.5:
                print(f"🔍 [Debug] Evidencia '{sym}' subiendo: {nuevo_valor:.2f} (T={t:.1f})")
                
            port.value = nuevo_valor

    def debug_pool_status(self, net):
        """Diagnóstico: ¿alguna vez dispararon las neuronas conectadas a evidencia_*?"""
        for sym in self.symbols:
            port = self.input_ports.get(f"evidencia_{sym}")
            if not port or not port.connected_neurons:
                print(f"🔬 [Diag] evidencia_{sym}: puerto sin neuronas conectadas")
                continue
                
            # Filtramos solo índices válidos para la red actual
            neurons = [idx for idx in port.connected_neurons if idx < net.n]
            
            # net.last_spike[idx] == 0.0 significa que NUNCA disparó desde que arrancó la red
            # Esto debería ser mucho menos frecuente ahora gracias al nuevo recable_from_live_band
            nunca_dispararon = [idx for idx in neurons if net.last_spike[idx] == 0.0]
            
            print(f"🔬 [Diag] evidencia_{sym}: {len(neurons)} neuronas | "
                  f"{len(nunca_dispararon)} nunca dispararon | "
                  f"índices={neurons[:8]}{'...' if len(neurons) > 8 else ''}")
            
    def debug_band_activity(self, net, z_range=(37.0, 53.0)):
        """¿Hay ALGUNA neurona activa en la banda z de lenguaje? Y en qué
        estado de energía están las que sobreviven ahí."""
        z = net.positions[:net.n, 2]
        en_banda = (z > z_range[0]) & (z < z_range[1]) & net.active[:net.n]
        idx_banda = np.where(en_banda)[0]
        disparadas = idx_banda[net.last_spike[idx_banda] > 0]

        energia_promedio = float(np.mean(net.energy[idx_banda])) if len(idx_banda) > 0 else 0.0
        energia_min = float(np.min(net.energy[idx_banda])) if len(idx_banda) > 0 else 0.0

        print(f"🔬 [Diag-Banda] z∈{z_range}: {len(idx_banda)} neuronas totales | "
              f"{len(disparadas)} dispararon alguna vez | "
              f"energía prom={energia_promedio:.3f} | energía mín={energia_min:.3f} | "
              f"ejemplos={disparadas[:8].tolist()}")      
    
    def debug_overlap_evidencia_banda(self, net, z_range=(37.0, 53.0)):
        """¿Las neuronas que ahora disparan en la banda son las mismas
        que están conectadas a evidencia_X/O, o son poblaciones distintas?"""
        z = net.positions[:net.n, 2]
        en_banda = (z > z_range[0]) & (z < z_range[1]) & net.active[:net.n]
        idx_banda = set(np.where(en_banda)[0].tolist())

        for sym in self.symbols:
            port = self.input_ports.get(f"evidencia_{sym}")
            if not port: continue
            conectadas = set(port.connected_neurons)
            interseccion = conectadas & idx_banda
            print(f"🔬 [Diag-Overlap] evidencia_{sym}: {len(conectadas)} conectadas | "
                  f"{len(interseccion)} están dentro de la banda z diagnosticada")

    def check_edge_events(self, net, t):
        """A.2 — Flanco ascendente sobre evidencia_{sym}.value → dispara
        el buffer de esa letra. Con debounce por histéresis."""
        for sym in self.symbols:
            current  = self.input_ports[f"evidencia_{sym}"].value
            previous = self._prev_evidence[sym]
            cruzo_umbral = previous <= self.edge_threshold < current
            
            if cruzo_umbral and not self._evidence_cooldown[sym]:
                self._fire_buffer_pulse(net, sym, t)
                self._evidence_cooldown[sym] = True
                print(f"⚡ [Secuencia] Evento '{sym}' en T={t:.1f} (evidencia={current:.2f})")
            
            # Se rearma solo cuando la señal cae bien por debajo del umbral,
            # para no disparar varias veces sobre el mismo estímulo sostenido
            if current < self.edge_threshold * 0.5:
                self._evidence_cooldown[sym] = False
                
            self._prev_evidence[sym] = current

    def _fire_buffer_pulse(self, net, sym, t, strength=6.0):
        for idx in self._buffer_pools.get(sym, []):
            if 0 <= idx < net.n and net.active[idx]:
                net.receive_spike(idx, strength, t)

    def get_ports_info(self):
        return [{"name": f"L_{p.name}", "pos": list(p.pos), "strength": p.strength,
                 "connected_neurons": list(p.connected_neurons)}
                for p in {**self.input_ports, **self.output_ports}.values()]


# modules/heart.py
from core.module import Module
import random
import numpy as np

class HeartModule(Module):
    def __init__(self):
        super().__init__("Corazón")
        
        # --- POSICIONAMIENTO CENTRAL (Espacio 150x150x100) ---
        # Elevamos un poco el Z a 60 para separarlo de la visión (Z=5)
        # Separamos X para que los 3 puertos del corazón no se solapen visualmente
        
        # Entrada: Neuronas de estrés y energía
        self.stress_port = self.add_input_port("stress_input", 
                                               strength=0.85, 
                                               pos=(60.0, 75.0, 60.0))
        
        self.energy_port = self.add_input_port("energy_input", 
                                               strength=0.70, 
                                               pos=(90.0, 75.0, 60.0))
        
        # Salida: El pulso se origina en el centro exacto del módulo
        self.pulse_port = self.add_output_port("pulse_output", 
                                               strength=1.2, 
                                               pos=(75.0, 75.0, 60.0))
        
        self.pulse_port.value = 0.0
        self.base_frequency = 1.2
        self.last_pulse_time = 0.0
        
        print("❤️ Corazón v3.1: Puertos configurados en plano Z=60.")

    def update(self, net, t):
        if not self.active:
            return

        stress = 0.0
        energy_pool = 0.0

        # 1. Procesar Estrés (basado en voltaje)
        if self.stress_port.connected_neurons:
            for idx in self.stress_port.connected_neurons:
                if 0 <= idx < net.n:
                    stress += net.get_voltage(idx, t) * self.stress_port.strength
            avg_stress = stress / len(self.stress_port.connected_neurons)
        else:
            avg_stress = 0.0

        # 2. Procesar Energía (basado en metabolismo)
        if self.energy_port.connected_neurons:
            for idx in self.energy_port.connected_neurons:
                if 0 <= idx < net.n:
                    energy_pool += net.energy[idx]
            avg_energy = energy_pool / len(self.energy_port.connected_neurons)
        else:
            avg_energy = 1.0

        # 3. Lógica de Frecuencia Biológica
        # El estrés acelera el corazón, la falta de energía lo ralentiza (bradicardia)
        freq = self.base_frequency * (1.0 + avg_stress * 1.8)
        if avg_energy < 0.3:
            freq *= 0.6 
        
        freq = max(0.4, min(4.5, freq))

        # Decaimiento del valor del puerto
        self.pulse_port.value *= 0.7 

        # 4. Disparo del Latido
        if t - self.last_pulse_time >= (1000.0 / freq):
            self.last_pulse_time = t
            self.pulse_port.value = 1.0 
            
            # El pulso es más vital si la red está en reserva (fuerza 2.0)
            strength = self.pulse_port.strength * (2.0 if avg_energy < 0.4 else 1.5)
            
            injected = 0
            # Protección para el i5: evitamos inyectar si la cola está saturada
            if len(net.event_queue) < 20000:
                for idx in self.pulse_port.connected_neurons:
                    if 0 <= idx < net.n and net.active[idx]:
                        jitter = random.uniform(0.0, 5.0)
                        net.receive_spike(idx, strength, t + jitter)
                        injected += 1

            if random.random() < 0.1: # Reducido log para no ensuciar la consola
                status = "CRÍTICO" if avg_energy < 0.3 else "ESTABLE"
                print(f"❤️ Latido: {freq:.2f}Hz | Spikes: {injected} | Estado: {status}")

    def get_ports_info(self):
        """Envía la info de los 3 puertos al visualizador."""
        ports_info = []
        # Mapeo explícito para asegurar que se lean los 3 atributos
        mapping = [
            (self.stress_port, '❤️ Estrés (IN)'),
            (self.energy_port, '❤️ Energía (IN)'),
            (self.pulse_port, '❤️ Pulso (OUT)')
        ]
        
        for port_obj, label in mapping:
            if port_obj:
                ports_info.append({
                    'name': label,
                    'pos': list(getattr(port_obj, 'pos', [75, 75, 60])),
                    'strength': getattr(port_obj, 'strength', 1.0)
                })
        return ports_info
    
    
    
    def auto_connect(self, net):
        """
        Sistema de conexión por proximidad (Foveación Homeostática).
        El corazón busca neuronas en su radio de influencia para censar y estimular.
        """
        # Radios de acción (puedes ajustarlos para que 'cubran' más o menos red)
        influence_radius = 25.0 
        
        # Limpiamos conexiones previas
        self.stress_port.connected_neurons = []
        self.energy_port.connected_neurons = []
        self.pulse_port.connected_neurons = []

        # Posiciones de los puertos (definidas en tu __init__)
        p_stress = np.array(self.stress_port.pos)
        p_energy = np.array(self.energy_port.pos)
        p_pulse = np.array(self.pulse_port.pos)

        for i in range(net.n):
            pos_i = net.positions[i]
            
            # --- Conexión al Puerto de Estrés ---
            if np.linalg.norm(pos_i - p_stress) < influence_radius:
                self.stress_port.connected_neurons.append(i)
                
            # --- Conexión al Puerto de Energía ---
            if np.linalg.norm(pos_i - p_energy) < influence_radius:
                self.energy_port.connected_neurons.append(i)
                
            # --- Conexión al Puerto de Pulso (Salida) ---
            # Este es el más importante para que el latido afecte a la red
            if np.linalg.norm(pos_i - p_pulse) < (influence_radius * 0.8):
                self.pulse_port.connected_neurons.append(i)

        print(f"❤️ Corazón Auto-Conectado: {len(self.pulse_port.connected_neurons)} neuronas marcapasos detectadas.")


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


# systems/calcium.py

from config import CALCIUM_DECAY, GLOBAL_CALCIUM_DECAY

class CalciumSystem:
    def update(self, net):
        total_fired = sum(net.fired)

        for i in range(net.n):
            net.calcium[i] *= CALCIUM_DECAY
            if net.fired[i]:
                net.calcium[i] += 0.38          # más calcio por spike

        # Calcio global mucho más dinámico
        activity_ratio = total_fired / max(1, net.n)

        # Recuperación depende de la actividad global
        recovery = 0.018 + activity_ratio * 0.035

        # Serotonina alta ayuda a estabilizar (menos oscilación)
        if hasattr(net.regions[0], 'serotonin'):
            avg_serotonin = sum(r.serotonin for r in net.regions) / len(net.regions)
            recovery *= (1.0 - (avg_serotonin - 0.5) * 0.25)

        net.global_calcium *= GLOBAL_CALCIUM_DECAY
        net.global_calcium = min(1.05, net.global_calcium + recovery)

        # Si hay muy poca actividad, el calcio global se recupera más lento
        if activity_ratio < 0.08:
            net.global_calcium *= 0.92


#systems/competition.py




import random

class CompetitionSystem:
    def __init__(self):
        # Bajamos la tasa base para que no mueran por "viejo" tan rápido
        self.base_death_rate = 0.0002 
        self.energy_death_threshold = 0.25 # Umbral más bajo
        self.max_deaths_per_step = 1 # Solo una por paso máximo para evitar colapsos

    def compete(self, net):
        if net.n == 0: return 0
        deaths = 0
        
        safe_n = min(len(net.active), len(net.region), len(net.energy))

        # Mezclamos el orden de chequeo para que no mueran siempre las primeras de la lista
        indices = list(range(safe_n))
        random.shuffle(indices)

        for i in indices:
            if not net.active[i]: continue

            r_idx = net.region[i]
            if r_idx < 0 or r_idx >= len(net.regions): continue

            region = net.regions[r_idx]
            neuron_energy = net.energy[i]

            # --- LÓGICA DE SUPERVIVENCIA CORREGIDA ---
            # Si la dopamina es positiva, la probabilidad de muerte baja drásticamente
            # Si es negativa, aumenta el riesgo, pero no es una sentencia automática
            
            death_prob = self.base_death_rate
            
            # Penalización por baja energía (el factor más importante)
            if neuron_energy < self.energy_death_threshold:
                death_prob += 0.05 
            
            # Penalización por Dopamina negativa (entorno hostil)
            if region.dopamine < 0:
                death_prob += abs(region.dopamine) * 0.01
            else:
                # Recompensa por Dopamina positiva (entorno sano)
                death_prob *= 0.1 

            # --- PROTECCIÓN PARA REDES PEQUEÑAS ---
            # Si hay menos de 50 neuronas en la región, no permitimos muertes aleatorias
            # Solo mueren si se quedan sin energía total.
            neuronas_en_region = sum(1 for j in range(safe_n) if net.active[j] and net.region[j] == r_idx)
            if neuronas_en_region < 50 and neuron_energy > 0.1:
                continue

            # Ejecutar sentencia
            if random.random() < death_prob:
                net.active[i] = False
                net.energy[i] = 0.0
                deaths += 1
                if deaths >= self.max_deaths_per_step: break

        if deaths > 0 and random.random() < 0.1: # Informar con 10% de probabilidad
            print(f"  ⚰️  Competencia: {deaths} neuronas eliminadas (Población: {sum(net.active)})")
        
        return deaths


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
            net.potential[i] += total_signal * 0.2          # antes era * 0.8

            # Boost energético mucho más suave (opcional pero recomendado)
            net.energy[i] += total_signal * 0.01            # antes era * 0.02

            # Input externo aleatorio residual (ruido muy bajo)
            if random.random() < 0.005:
                net.potential[i] += 0.5


# systems/genetics.py
from core.evolution_config import EVO_MENU, PROFILE
import random
import numpy as np
from config import GRID_SIZE  # Importación necesaria para normalizar correctamente

class GeneticSystem:
    def __init__(self):
        self.mutation_rate = 0.05
        self.crossover_prob = 0.7
        # NUEVO: Fuerza del instinto de conexión vertical
        self.tropism_strength = 0.3 

    def mutate_new_neuron(self, net, idx, parent_idx=None):
        """
        Refina el ADN y establece la firma química (Excitatoria vs Inhibitoria).
        v3.5: Codificación de identidad funcional.
        """
        # --- HERENCIA Y MUTACIÓN ---
        if parent_idx is not None:
            parent_dna = net.dna[parent_idx]
            new_dna = []
            for gene in parent_dna:
                if random.random() < self.mutation_rate:
                    mutation = random.uniform(-0.15, 0.15)
                    new_dna.append(np.clip(gene + mutation, -1.0, 1.0))
                else:
                    new_dna.append(gene)
            net.dna[idx] = new_dna
        else:
            # Si no hay padre, el ADN base ya fue generado en __init__
            pass
        
        dna = net.dna[idx]
        
        # --- EXPRESIÓN GENÉTICA (ADN -> QUÍMICA) ---
        gene_polaridad = dna[3]
        gene_metabolismo = dna[4]
        gene_z_bias = dna[5] if len(dna) > 5 else random.uniform(0.1, 1.0)
        
        # NT 0: Glutamato (Excitación)
        excitatory = max(0.05, gene_polaridad) if gene_polaridad > 0 else 0.05
        # NT 1: GABA (Inhibición)
        inhibitory = abs(gene_polaridad) if gene_polaridad < 0 else 0.05
        
        raw_nt = [
            excitatory,                      # NT 0: Excitación
            inhibitory,                      # NT 1: Inhibición Lateral
            abs(gene_metabolismo),           # NT 2: Sensibilidad a Dopamina
            abs(gene_polaridad * gene_metabolismo), # NT 3: Estabilidad Sináptica
            gene_z_bias                      # NT 4: Tropismo (Crecimiento)
        ]

        # Normalización del vector químico
        total = sum(raw_nt) or 1.0
        net.nt_vector[idx] = [val / total for val in raw_nt]

        # --- CONFIGURACIÓN DE RECEPTORES ---
        if inhibitory > excitatory:
            net.receptors[idx] = [0.8, 0.1, 0.4, 0.2, 0.5] # Fuerte respuesta al Glutamato
        else:
            net.receptors[idx] = [0.3, 0.6, 0.4, 0.2, 0.5] # Fuerte respuesta al GABA (Auto-regulación)
    
    def apply_evolution_step(self, net):
        """
        Evolución basada en el éxito de la Mirada.
        Las neuronas que ayudan a mover la mirada desde 0.50 sobreviven más.
        """
        fitness_scores = []
        
        # Calculamos el éxito de la red (distancia Mirada vs Foco)
        try:
            eye_x, eye_y = net.vision.current_eye_pos if hasattr(net, 'vision') else (0.5, 0.5)
            foci = net.vision.last_foci if hasattr(net, 'vision') else [(0.5, 0.5)]
            
            # Recompensa por reducir el error del 0.50
            global_reward = 0.1
            if foci:
                # 🔥 CORRECCIÓN: Normalización dinámica usando GRID_SIZE
                dist = np.sqrt((eye_x - foci[0][0]/GRID_SIZE)**2 + (eye_y - foci[0][1]/GRID_SIZE)**2)
                global_reward = max(0.1, 1.0 - dist)
        except:
            global_reward = 0.1

        for i in range(net.n):
            if net.active[i]:
                # FITNESS = Energía + Eficiencia + (Dopamina Regional * Recompensa Global)
                r_idx = net.region[i]
                reg_dopamine = net.regions[r_idx].dopamine if r_idx < len(net.regions) else 0
                
                score = (net.energy[i] * 0.5) + (reg_dopamine * global_reward * 0.5)
                fitness_scores.append((i, score))
        
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        top_count = max(1, len(fitness_scores) // 10)
        winners = [idx for idx, score in fitness_scores[:top_count]]

        if winners:
            self._horizontal_transfer(net, winners)

    def _horizontal_transfer(self, net, winners):
        """Transferencia de rasgos: Las neuronas 'ciegas' aprenden de las que 'ven'."""
        for _ in range(10):
            source = random.choice(winners)
            target = random.randint(0, net.n - 1)
            
            if net.active[target] and source != target:
                # El target absorbe el ADN del ganador
                for g in range(len(net.dna[target])):
                    if random.random() < 0.2:
                        net.dna[target][g] = (net.dna[target][g] * 0.8) + (net.dna[source][g] * 0.2)
                
                # --- AQUÍ APLICAMOS LA MUTACIÓN FÍSICA ---
                self._sync_neuron_physics(net, target)

    def _sync_neuron_physics(self, net, idx):
        """
        Sincroniza los parámetros físicos (tau_m, v_thresh, refr_period) 
        con el ADN actual de la neurona.
        """
        dna = net.dna[idx]
        polaridad = dna[3] # Usamos el gen 3 como selector
        
        # Seleccionamos perfil basado en el gen de polaridad
        if polaridad > 0.5:
            perfil = EVO_MENU["RAPID_FIRE"]
        elif polaridad < -0.5:
            perfil = EVO_MENU["BUFFER"]
        else:
            perfil = EVO_MENU["RELAY"]
            
        # Actualizamos los vectores de estado de la red
        net.tau_m[idx] = perfil.tau_m
        net.v_thresh[idx] = perfil.v_thresh
        net.refr_period[idx] = perfil.refr_period




#systems/Global_workspace.py

class GlobalWorkspace:
    def update(self, net):
        best_r = None
        best_score = -999

        # Buscar la región más dominante
        for i, r in enumerate(net.regions):
            score = r.dopamine + r.energy

            if score > best_score:
                best_score = score
                best_r = i

        net.global_workspace["region"] = best_r
        net.global_workspace["signal"] = best_score

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


#systems/homoestasis.py

from config import TARGET_ACTIVITY, HOMEOSTASIS_RATE
import numpy as np

class HomeostasisSystem:
    def apply(self, net):
        # Medimos la actividad real sin interferir con el vector principal
        global_activity = np.mean(net.fired) if isinstance(net.fired, np.ndarray) else (sum(net.fired) / max(1, net.n))
        emergency_boost = 2.0 if global_activity < 0.05 else 1.0 # Más sensible si cae al 5%

        # Detectar dinámicamente el nombre exacto del mapa de regiones
        r_attr = 'neuron_region' if hasattr(net, 'neuron_region') else 'region'
        indices = getattr(net, r_attr)

        for i in range(net.n):
            if not net.active[i]: continue

            r_id = indices[i]
            if r_id not in net.regions: continue
            
            region = net.regions[r_id]
            current_activity = net.fired[i]

            # Objetivo dinámico influenciado por químicos
            dynamic_target = TARGET_ACTIVITY + (getattr(region, 'dopamine', 0.0) * 0.03)
            
            # Velocidad de ajuste con el boost de emergencia y acetilcolina protegida
            ach = getattr(region, 'acetylcholine', 0.0)
            adjustment_speed = HOMEOSTASIS_RATE * (0.5 + ach) * emergency_boost

            # Ajuste asimétrico (sube rápido, baja lento)
            if current_activity == 0:
                net.excitability[i] += adjustment_speed * dynamic_target * 1.5
            else:
                net.excitability[i] -= adjustment_speed * (1.0 - dynamic_target) * 0.4

            # Límites de seguridad basados en la dopamina regional
            max_exc = 3.0 + getattr(region, 'dopamine', 0.0)
            min_exc = 0.15
            net.excitability[i] = max(min_exc, min(max_exc, net.excitability[i]))

# systems/inhibition.py
# Sistema de Inhibición GABA - Equilibrio E/I

import random

class InhibitionSystem:
    def __init__(self):
        self.base_gaba = 0.22                    # fuerza base de inhibición
        self.gaba_decay = 0.915                  # decaimiento de sensibilidad

    def apply(self, net):
        """
        Aplica inhibición GABA de forma realista:
        - Más fuerte cuando hay alta actividad (evita sobre-excitación)
        - Modulada por serotonina (estabilidad)
        - Afecta tanto el potencial como la excitabilidad
        """
        for i in range(net.n):
            if not net.active[i]:
                continue

            region = net.regions[net.region[i]]

            # Fuerza de GABA depende de:
            # - Serotonina alta = más inhibición (estabilidad)
            # - Actividad reciente = más inhibición (control de picos)
            gaba_strength = self.base_gaba

            gaba_strength += (region.serotonin - 0.5) * 0.25      # serotonina aumenta inhibición
            gaba_strength += net.fired[i] * 0.18                  # más fuerte después de spike

            # Sensibilidad individual de la neurona
            gaba_strength *= net.gaba_sensitivity[i]

            # Aplicar inhibición al potencial
            net.potential[i] -= gaba_strength * 0.85

            # Reducir ligeramente la excitabilidad (efecto acumulativo)
            net.excitability[i] *= (0.97 + region.serotonin * 0.03)

            # Clamp suave
            net.excitability[i] = max(0.15, min(2.7, net.excitability[i]))

        # Decaimiento lento de sensibilidad GABA (adaptación)
        for i in range(net.n):
            if net.active[i]:
                net.gaba_sensitivity[i] *= self.gaba_decay
                net.gaba_sensitivity[i] = max(0.35, net.gaba_sensitivity[i])



# systems/metabolism.py
import numpy as np

class MetabolismSystem:
    def __init__(self):
        # Umbrales críticos
        self.death_threshold = 0.15      # Si baja de aquí, la neurona muere
        self.atrophy_limit = 8000.0      # ms de silencio antes de considerar atrofia severa
        print("🧬 Sistema Metabólico v3.2 (Selección Natural) inicializado.")

    
    
    
    
    def get_status(self, net):
        """Devuelve un resumen de salud de la red."""
        active_count = np.sum(net.active)
        avg_energy = np.mean(net.energy[net.active]) if active_count > 0 else 0
        return f"Salud: {avg_energy:.2f} | Vivas: {active_count}"
    
    
    
            
    def update(self, net, t, dt, state="awake"):
        if net.n == 0: return

        r_attr = 'neuron_region' if hasattr(net, 'neuron_region') else 'region'
        region_indices = getattr(net, r_attr)

        heart_pulse = 0.0
        if hasattr(net, 'heart') and net.heart.active:
            heart_pulse = getattr(net.heart.pulse_port, 'value', 0.0)
        global_nutrient = heart_pulse * 0.02

        # Parámetros por estado
        if state == "awake":
            base_cost = 0.0012
            recovery  = 0.0025
        elif state == "drowsy":
            base_cost = 0.0008
            recovery  = 0.004
        else:  # sleep
            base_cost = 0.0003
            recovery  = 0.009

        # Verificar si el sistema de fitness está disponible
        tiene_fitness = hasattr(net, 'fitness_neuronal')

        for i in range(net.n):
            if not net.active[i]: continue

            # 1. CONSUMO Y ATROFIA
            time_since_spike = t - net.last_spike[i]
            atrophy_factor   = 1.0
            if time_since_spike > self.atrophy_limit:
                atrophy_factor = min(3.0, 1.0 + (time_since_spike - self.atrophy_limit) / 5000.0)

            spike_cost    = 0.035 * net.fired[i]
            net.energy[i] -= (base_cost * atrophy_factor) + spike_cost

            # 2. RECUPERACIÓN
            r_idx          = region_indices[i]
            zone_efficiency = 1.0
            if r_idx != -1 and r_idx in net.regions:
                region = net.regions[r_idx]
                zone_efficiency = 1.0 + (region.acetylcholine * 1.5)

            net.energy[i] += (recovery + global_nutrient) * zone_efficiency

            # 3. MUERTE POR INANICIÓN
            if net.energy[i] < self.death_threshold:
                net.active[i] = False
                net.energy[i] = 0.0
                if hasattr(net, 'region_counts') and r_idx < len(net.region_counts):
                    net.region_counts[r_idx] -= 1
                continue

            # 4. TECHO DINÁMICO POR FITNESS NEURONAL
            # Una neurona entrenada (fitness=2.0) puede tener hasta 2x más energía
            # Una neurona atrofiada (fitness=0.5) tiene techo muy bajo
            if tiene_fitness:
                techo = float(net.fitness_neuronal[i])

                # Durante el sueño, el techo sube un 20% extra para consolidación
                if state == "sleep":
                    techo = min(2.0, techo * 1.2)

                net.energy[i] = min(techo, net.energy[i])

                # === DECAIMIENTO DE FITNESS POR INACTIVIDAD ===
                # Si la neurona lleva mucho tiempo sin disparar, pierde condición física
                if time_since_spike > self.atrophy_limit * 2:
                    decaimiento_fitness = 0.00005 * atrophy_factor
                    net.fitness_neuronal[i] = max(0.5, net.fitness_neuronal[i] - decaimiento_fitness)
            else:
                # Fallback si fitness no existe: techo fijo original
                net.energy[i] = min(1.1, net.energy[i])

        if hasattr(net, 'heart'):
            net.heart.pulse_port.value *= 0.6

    def get_sleep_time(self, stress_factor):
        return 1.0 + (stress_factor * 2.0)    


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



# systems/neurotransmitter.py

import random
from config import VESICLE_RELEASE_PROB

class NeurotransmitterSystem:
    def modulate(self, net, i, j, signal):
        if net.vesicles[j] <= 0:
            return 0.0

        # Acetilcolina aumenta la probabilidad de liberación (atención)
        ach_factor = 0.6 + 0.8 * net.regions[net.region[j]].acetylcholine

        release_prob = VESICLE_RELEASE_PROB * ach_factor
        if random.random() > release_prob:
            return 0.0

        net.vesicles[j] -= 1

        affinity = sum(a * b for a, b in zip(net.nt_vector[j], net.receptors[i]))
        calcium_factor = 0.5 + net.calcium[j]

        return signal * (0.5 + affinity) * calcium_factor

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
            if net.region[i] == net.region[j]:
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
            
                region_i.energy -= ENERGY_LEARN_COST * 0.02

        # 2. --- PLASTICIDAD ESTRUCTURAL (Crecimiento Dirigido) ---
        active_indices = [i for i, a in enumerate(net.active) if a]
    
        for i in random.sample(active_indices, min(len(active_indices), 100)):
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


#system/prediction.py

class PredictionSystem:
    def __init__(self):
        self.prev_energy = 0.0

    def update(self, net):
        # Valor actual
        current = net.self_state.get("energy", 0.0)

        # 🔮 Predicción = último valor conocido
        predicted = self.prev_energy

        # ❌ Error real
        error = abs(current - predicted)

        net.self_state["prediction_error"] = error

        # 🧠 Guardar para el próximo paso
        self.prev_energy = current


#systems/regions.py


import numpy as np
from config import NUM_REGIONS

class RegionSystem:
    def __init__(self):
        self.iteration = 0
        self.extinction_threshold = 0.2  # Si el fitness cae de aquí, peligro
        self.learning_rate_sig = 0.01    # Qué tan rápido cambia la identidad regional

    
    def update(self, net):
        self.iteration += 1
        
        # 1. Recopilar datos por región
        region_data = {r.id: {"potentials": [], "nt_vector": []} for r in net.regions.values()}
        
        for i in range(net.n):
            if net.active[i]:
                # --- CAMBIO AQUÍ: net.region -> net.neuron_region ---
                rid = net.neuron_region[i] 
                
                if rid in region_data:
                    region_data[rid]["potentials"].append(net.membrane_potential[i])
                    # --- CAMBIO AQUÍ: net.nt_vector -> net.nt_vectors (plural) ---
                    region_data[rid]["nt_vector"].append(net.nt_vector[i])
                    
                    
        # 2. Actualizar cada región individualmente
        active_list = [r for r in net.regions.values() if r.is_active]
        
        for r in active_list:
            data = region_data[r.id]
            avg_act = np.mean(data["potentials"]) if data["potentials"] else 0.0
            avg_nt = np.mean(data["nt_vector"], axis=0) if data["nt_vector"] else None
            
            if avg_nt is not None:
                r.update_signature(avg_nt, lr=self.learning_rate_sig)

            r.update_state(
                activity=float(avg_act),
                output=float(avg_act * 0.5), 
                energy_input=net.heart.energy_port.value * 0.1,
                env_signal=net.heart.pulse_port.value
            )

        # --- NUEVO: 2.5 COMPETENCIA LATERAL (Foco Atencional) ---
        # Hacemos que las regiones peleen por los recursos antes de la extinción
        for r in active_list:
            if r.activity > 0.1: # Solo las regiones con pulso pueden intentar dominar
                r.apply_lateral_inhibition(active_list)

        # 3. COMPETENCIA Y EXTINCIÓN (Cada 100 pasos)
        if self.iteration % 100 == 0:
            self._process_competition(net)
    

    def _process_competition(self, net):
        """Las regiones con bajo fitness pierden sus neuronas."""
        active_regions = [r for r in net.regions.values() if r.is_active]
        if len(active_regions) <= 2: 
            return # Mantener un mínimo de diversidad

        # --- SOLUCIÓN AL ATTRIBUTE ERROR ---
        # Calculamos el conteo actual de neuronas por región al vuelo.
        # Esto asegura que net.region_counts siempre exista y esté actualizado.
        unique, counts = np.unique(net.neuron_region, return_counts=True)
        net.region_counts = dict(zip(unique, counts))
        # ------------------------------------

        for r in active_regions:
            # Usamos .get(r.id, 0) para evitar errores si la región está totalmente vacía
            neuron_count = net.region_counts.get(r.id, 0)
            
            # Si una región es muy ineficiente o está casi vacía
            if r.fitness < self.extinction_threshold or neuron_count < 5:
                print(f"💀 Región {r.id} en extinción. Fitness: {r.fitness:.2f} | Neuronas: {neuron_count}")
                self._extinguish_region(net, r)

    def _extinguish_region(self, net, r_id):
     """Manejo seguro de extinción y re-colonización"""
     try:
        # 1. Verificación de existencia (Vital para diccionarios)
        if r_id not in net.regions:
            return 

        print(f"💀 Extinguiendo Región {r_id} (Fitness insuficiente)...")
        
        # 2. Reasignación de neuronas antes de borrar
        # Buscamos todas las neuronas que pertenecían a esta región
        neuronas_huerfanas = [i for i, reg in enumerate(net.neuron_regions) if reg == r_id]
        
        for i in neuronas_huerfanas:
            # Intentamos asignarlas a la región con mayor afinidad química
            new_reg_id = net._assign_region_by_affinity(i)
            net.neuron_regions[i] = new_reg_id
            
        # 3. Borrado final del objeto región
        del net.regions[r_id]
        
     except Exception as e:
        print(f"⚠️ Error crítico en extinción de Región {r_id}: {e}")
        # Opcional: podrías forzar un volcado de log aquí si el error persiste



#systems/self_system.py


class SelfSystem:
    def update(self, net):
        if net.n == 0:
            return

        total_energy = 0.0
        total_activity = 0.0
        total_stress = 0.0

        for i in range(net.n):
            if not net.active[i]:
                continue

            total_energy += net.energy[i]
            total_activity += net.fired[i]
            total_stress += abs(net.potential[i])

        n = max(1, net.n)

        net.self_state["energy"] = total_energy / n
        net.self_state["activity"] = total_activity / n
        net.self_state["stress"] = total_stress / n

        # Coherencia = qué tan sincronizada está la red
        variance = sum((x - net.self_state["activity"])**2 for x in net.fired) / n
        net.self_state["coherence"] = 1.0 - variance


        
         
                                                                                                                                                                                        