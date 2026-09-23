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