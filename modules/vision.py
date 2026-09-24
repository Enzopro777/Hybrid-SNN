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
        """Conecta puertos una sola vez sin depender de radios espaciales.

        Los candidatos se toman de neuronas activas que ya forman parte de
        la red sináptica; el vínculo del puerto es estable y no se recalcula
        por proximidad en cada ciclo.
        """
        candidatos = [
            i for i in range(net.n)
            if net.active[i] and not getattr(net, 'heart_mask', np.zeros(net.max_neurons, dtype=bool))[i]
            and len(net.connections[i]) > 0
        ]
        if not candidatos:
            print("⚠️ [Visión] No hay candidatos sinápticos para conectar puertos.")
            return

        puertos = [self.coord_x_port, self.coord_y_port, self.fovea_radius_port]
        paso = max(1, len(candidatos) // len(puertos))
        for p_idx, port in enumerate(puertos):
            if len(port.connected_neurons) >= 12:
                continue
            inicio = p_idx * paso
            seleccion = candidatos[inicio:inicio + 12]
            for idx in seleccion:
                port.connect(int(idx))
        print("🔌 [Visión] Puertos conectados a neuronas sinápticas (sin radio).")

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
            
            # --- INSTRUMENTACIÓN CAUSAL: salidas independientes por bloque ---
            _probe_outputs = {}

            # --- TRANSDUCTOR (Bloque 1): normaliza amplitud ---
            if getattr(self, '_transductor', None) is not None:
                safe_activations = self._transductor.process(safe_activations)
                _probe_outputs["transductor"] = list(safe_activations)

            # Guardar output del Transductor: BordesV y BordesH deben recibir
            # la misma señal base (píxeles normalizados z=30,45), no la señal
            # ya acumulada con bordes de otros detectores.
            transductor_out = safe_activations

            # --- BORDES VERTICALES (Bloque 2a): gradiente horizontal → z=70 ---
            if getattr(self, '_edge_detector_v', None) is not None:
                edge_v = self._edge_detector_v.process(transductor_out)
                _probe_outputs["bordes_v"] = list(edge_v) if edge_v else []
                if edge_v:
                    safe_activations = safe_activations + edge_v

            # --- BORDES HORIZONTALES (Bloque 2b): gradiente vertical → z=72 ---
            if getattr(self, '_edge_detector_h', None) is not None:
                edge_h = self._edge_detector_h.process(transductor_out)
                _probe_outputs["bordes_h"] = list(edge_h) if edge_h else []
                if edge_h:
                    safe_activations = safe_activations + edge_h

            # --- DIAGONALES (Bloque 2c): gradientes diag → z=74 (barra) z=76 (slash) ---
            # Recibe transductor_out igual que V y H — independiente del frame acumulado.
            if getattr(self, '_diag_detector', None) is not None:
                diags = self._diag_detector.process(transductor_out)
                _probe_outputs["diagonales"] = list(diags) if diags else []
                if diags:
                    safe_activations = safe_activations + diags

            # --- COMBINADOR DE ESQUINAS (Bloque 3): co-ocurrencia V+H → z=80 ---
            # Recibe TODAS las activaciones (incluye z=70 y z=72) y filtra por z.
            if getattr(self, '_combinador_esquina', None) is not None:
                esquinas = self._combinador_esquina.process(safe_activations)
                _probe_outputs["esquinas"] = list(esquinas) if esquinas else []
                if esquinas:
                    safe_activations = safe_activations + esquinas

            # --- CURVAS (Bloque 2d): curvatura local → z=78 ---
            # Opera sobre transductor_out (misma base que V, H, Diag).
            if getattr(self, '_curva_detector', None) is not None:
                curvas = self._curva_detector.process(transductor_out)
                _probe_outputs["curvas"] = list(curvas) if curvas else []
                if curvas:
                    safe_activations = safe_activations + curvas

            # --- SIMETRÍA (Bloque 2e): simetría bilateral → z=82, radial → z=84 ---
            if getattr(self, '_simetria_detector', None) is not None:
                sims = self._simetria_detector.process(transductor_out)
                _probe_outputs["simetria"] = list(sims) if sims else []
                if sims:
                    safe_activations = safe_activations + sims

            # --- JUNCTIONS (Bloque 3b): cruces de bordes → z=86 ---
            # Recibe TODAS las activaciones (filtra por z internamente).
            if getattr(self, '_junction_detector', None) is not None:
                juncs = self._junction_detector.process(safe_activations)
                _probe_outputs["junctions"] = list(juncs) if juncs else []
                if juncs:
                    safe_activations = safe_activations + juncs

            # --- DETECTOR DE LETRAS (Bloque 5): features → spikes en evidencia ---
            # FIX v1.5.9: verificar alineación de z entre CurvaDetector y LetraDetector (una vez)
            if not getattr(self, '_curv_z_checked', False):
                self._curv_z_checked = True
                _curv_det = getattr(self, '_curva_detector', None)
                _letra_det = getattr(self, '_letra_detector', None)
                if _curv_det is not None and _letra_det is not None:
                    _cz_out = float(getattr(_curv_det, 'output_z', -1))
                    _cz_in  = float(getattr(_letra_det, 'curv_z', -1))
                    _ztol   = float(getattr(_letra_det, 'z_tol', 1.0))
                    if abs(_cz_out - _cz_in) > _ztol:
                        import logging
                        logging.error(
                            f"[Vision] DESALINEACIÓN Z: CurvaDetector.output_z={_cz_out} "
                            f"vs LetraDetector.curv_z={_cz_in} (z_tol={_ztol}). "
                            f"curv_per_bv SIEMPRE SERÁ 0.0. Corregir simulation.py."
                        )
                    else:
                        import logging
                        logging.info(
                            f"[Vision] Alineación Z OK: CurvaDetector.output_z={_cz_out} "
                            f"== LetraDetector.curv_z={_cz_in} (z_tol={_ztol})"
                        )
            if getattr(self, '_letra_detector', None) is not None:
                letra_out = self._letra_detector.process(safe_activations, net=net, t=t)
                _probe_outputs["letras"] = list(letra_out) if letra_out else []

            # Guardar salidas para la sonda. La observación se hará después,
            # cuando worker_vision haya actualizado los puertos de evidencia.
            self._last_probe_outputs = _probe_outputs

            net.inject_sensory_activity(safe_activations)

        # Limpiar activaciones después de procesar para evitar que el mismo
        # frame se reinyecte si update_from_external no es llamado en el ciclo siguiente.
        self.current_activations = []

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
    # TELEMETRÍA DE VISIÓN — lectura, no control
    # ------------------------------------------------------------------

    def get_fovea_telemetry(self):
        """Snapshot ligero y estable de la mirada real. No modifica estado."""
        return {
            "x": float(np.clip(self.fovea_center_x, 0.0, 1.0)),
            "y": float(np.clip(self.fovea_center_y, 0.0, 1.0)),
            "radius": float(np.clip(getattr(self.fovea_radius_port, "value", 0.15), 0.0, 0.5)),
            "vx": float(self.vel_x),
            "vy": float(self.vel_y),
            "speed": float(math.sqrt(self.vel_x ** 2 + self.vel_y ** 2)),
        }

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