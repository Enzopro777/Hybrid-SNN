# engine/simulation.py

from core.module import Module
from systems.trial_commit_journal import TrialCommitJournal

try:
    import msvcrt
except ImportError:  # entorno Linux/CI
    msvcrt = None
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

# Fase 3: límite defensivo para colas de eventos en el motor.
MAX_EVENTS_QUEUE = int(globals().get("MAX_EVENTS_QUEUE", 25000) or 25000)

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
from modules.curriculum import VocabularyCurriculum


from modules.monitor_block import MonitorBlock, restore_blocks_from_state
from modules.transductor_block import TransductorBlock
from modules.detector_bordes_v import VerticalEdgeDetectorBlock
from modules.detector_bordes_h import HorizontalEdgeDetectorBlock
from modules.combinador_esquina import EsquinaBlock
from modules.detector_diagonales import DiagonalDetectorBlock
from modules.detector_curvas import CurvaDetectorBlock
from modules.detector_simetria import SimetriaDetectorBlock
from modules.combinador_junction import JunctionBlock

from modules.detector_letras import LetraDetectorBlock
from modules.causal_probe import CausalProbe
from config import CAUSAL_TRIAL_FRAME_DT_MS

class SimulationEngine:
    
    def __init__(self, net=None):
        # 1. Sincronización de Red y Módulos Base
        self.net = net if net is not None else NeuralNetwork()
        self.state = "awake"
        self.is_running = False
        self.step_count = 0

        # Hilos de entrenamiento: se rastrean para impedir que un daemon siga
        # escribiendo en stdout mientras Python destruye el intérprete.
        self._training_threads = set()
        self._training_threads_lock = threading.Lock()
        self._training_stop_event = threading.Event()
        self._shutdown_in_progress = False
        self._trial_commit_journal = (
            TrialCommitJournal(str(globals().get("TRIAL_COMMIT_JOURNAL_PATH", "trial_commit_1.6.29f.json")))
            if bool(globals().get("TRIAL_COMMIT_JOURNAL_ENABLED", True)) else None
        )

        # Instrumentación causal: observacional, sin crear neuronas ni
        # modificar pesos/conexiones/umbrales.
        self.causal_probe = CausalProbe(
            net=self.net,
            output_path="causal_probe_results.json",
            max_events=4000,
            flush_every=100,
        )
        self.net.causal_probe = self.causal_probe

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
                self.lang_module = LetterIOModule(net=self.net, symbols=tuple(LETTER_VOCABULARY))
                self.net.modules["lenguaje"] = self.lang_module
                self.net.lang_module = self.lang_module # Atributo directo por conveniencia
                logging.info("🗣️ Módulo de Lenguaje (Letras) inicializado.")
            except ImportError:
                logging.warning("⚠️ No se encontró modules.language_io, omitiendo Módulo de Lenguaje.")
        
        # Conectarlo siempre, sin importar si es nuevo o recuperado
        if hasattr(self, 'lang_module') and hasattr(self.lang_module, 'auto_connect'):
            self.lang_module.auto_connect(self.net)
            logging.info("🔌 Puertos de Lenguaje (X, O) conectados a la red.")
            
            # --- NUEVO FIX 2A: Asignar pools de buffer si existen neuronas suficientes ---
            if hasattr(self.lang_module, 'assign_buffer_pools'):
                self.lang_module.assign_buffer_pools(self.net)
                logging.info("🧠 Pools de buffer asignados al inicializar.")
            # -----------------------------------------------------------------------------

        # Garantizar que todos los punteros apunten a la misma instancia de LetterIOModule
        if hasattr(self, 'lang_module') and self.lang_module:
            self.net.modules["lenguaje"] = self.lang_module
            self.net.lang_module = self.lang_module
            print(f"🔗 [Sync] lang_module unificado. id={id(self.lang_module)}", flush=True)

        # =====================================================================
        # --- NUEVO: CURRÍCULA ADAPTATIVA DE VOCABULARIO (Etapa 2) ---
        # =====================================================================
        # Nace siempre con valores por defecto acá; si había una corrida
        # guardada, load_state() la re-hidrata más tarde con lo que haya en
        # net.curriculum_state (ver comentario en load_state para el porqué
        # del orden: este __init__ corre ANTES de que main.py llame a
        # sim.load_state()).
        self.curriculum = VocabularyCurriculum(
            vocabulary=LETTER_VOCABULARY,
            starter_letters=CURRICULUM_STARTER_LETTERS,
            window=CURRICULUM_WINDOW,
            mastery_evidence=CURRICULUM_MASTERY_EVIDENCE,
            mastery_success_rate=CURRICULUM_MASTERY_SUCCESS_RATE,
            mastery_margin=CURRICULUM_MASTERY_MARGIN,
            min_trials_to_unlock=CURRICULUM_MIN_TRIALS_TO_UNLOCK,
            weak_letter_boost=CURRICULUM_WEAK_LETTER_BOOST,
            balanced_mode=CURRICULUM_BALANCED_MODE,
            max_trial_imbalance=CURRICULUM_MAX_TRIAL_IMBALANCE,
        )
        self.curriculum_autoplay = CURRICULUM_AUTOPLAY_DEFAULT
        self._last_autoplay_trial = 0.0
        logging.info(f"🎓 [Currícula] Vocabulario objetivo: {LETTER_VOCABULARY} | "
                     f"arrancando con: {self.curriculum.unlocked}")

        # =====================================================================
        # --- NUEVO: CURRÍCULA DE SECUENCIAS / PALABRAS (Etapa 3) ---
        # =====================================================================
        # VocabularyCurriculum es genérica sobre qué es un "símbolo": no le
        # importa si son letras o bigramas, así que la reusamos tal cual en
        # vez de escribir una clase nueva. Igual que con self.curriculum,
        # nace acá con defaults y load_state() la rehidrata después si había
        # progreso guardado (net.word_curriculum_state).
        self.word_curriculum = VocabularyCurriculum(
            vocabulary=WORD_VOCABULARY,
            starter_letters=WORD_STARTER_WORDS,
            window=WORD_CURRICULUM_WINDOW,
            mastery_evidence=WORD_MASTERY_EVIDENCE,
            mastery_success_rate=WORD_MASTERY_SUCCESS_RATE,
            min_trials_to_unlock=WORD_MIN_TRIALS_TO_UNLOCK,
            weak_letter_boost=WORD_WEAK_LETTER_BOOST,
        )
        self.word_curriculum_autoplay = WORD_CURRICULUM_AUTOPLAY_DEFAULT
        self._last_word_autoplay_trial = 0.0
        if hasattr(self, 'lang_module') and hasattr(self.lang_module, 'configure_words'):
            self.lang_module.configure_words(WORD_VOCABULARY)
        logging.info(f"🔤 [Secuencia] Vocabulario de palabras objetivo: {WORD_VOCABULARY} | "
                     f"arrancando con: {self.word_curriculum.unlocked}")

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

        # =====================================================================
        # --- ARQUITECTURA DE BLOQUES: Bloque Monitor ---
        # El primer bloque de la nueva arquitectura. Se inyecta en la banda
        # z=37-53 (zona de lenguaje) y reemplaza el diagnóstico externo.
        # Si hay un estado guardado, load_state() lo re-hidrata después.
        # =====================================================================
        if "monitor_lenguaje" not in self.net.modules:
            self.monitor_block = MonitorBlock(zone_z=(37.0, 53.0), n_immortal=8, log_every=50)
            self.monitor_block.inject(self.net)
        else:
            # Ya existía (recuperado de persistencia)
            self.monitor_block = self.net.modules["monitor_lenguaje"]
            logging.info("🧱 [MonitorBlock] Recuperado desde persistencia.")

        # --- ARQUITECTURA DE BLOQUES: Bloque Transductor ---
        # Normaliza la señal visual antes de inyectarla en la red.
        # No tiene neuronas inmortales: opera sobre listas Python.
        # Si hay un estado guardado, load_state() lo re-hidrata después.
        if "transductor_vision" not in self.net.modules:
            self.transductor = TransductorBlock(zone_z=(10.0, 25.0))   # separado de banda de lenguaje (37-53)
            self.transductor.register(self.net)
        else:
            self.transductor = self.net.modules["transductor_vision"]

        self.vision._transductor = self.transductor
        self.vision._causal_probe = self.causal_probe

        
        # --- Detector de bordes verticales (Bloque 2a, Paso 3) ---
        if "detector_bordes_v" not in self.net.modules:
            self.edge_detector_v = VerticalEdgeDetectorBlock(
                zone_z=(65.0, 80.0),
                output_z=70.0,
            )
            self.edge_detector_v.register(self.net)
        else:
            self.edge_detector_v = self.net.modules["detector_bordes_v"]
            logging.info("🔍 [BordesV] Recuperado desde persistencia.")
        self.vision._edge_detector_v = self.edge_detector_v

        # --- Detector de bordes horizontales (Bloque 2b, Paso 4) ---
        if "detector_bordes_h" not in self.net.modules:
            self.edge_detector_h = HorizontalEdgeDetectorBlock(
                zone_z=(80.0, 92.0),
                output_z=72.0,
            )
            self.edge_detector_h.register(self.net)
        else:
            self.edge_detector_h = self.net.modules["detector_bordes_h"]
            logging.info("🔍 [BordesH] Recuperado desde persistencia.")
        self.vision._edge_detector_h = self.edge_detector_h

        # --- Detector de diagonales (Bloque 2c) ---
        if "detector_diagonales" not in self.net.modules:
            self.diag_detector = DiagonalDetectorBlock(
                zone_z=(100.0, 110.0),
                barra_z=74.0,
                slash_z=76.0,
            )
            self.diag_detector.register(self.net)
        else:
            self.diag_detector = self.net.modules["detector_diagonales"]
            logging.info("╲╱ [Diag] Recuperado desde persistencia.")
        self.vision._diag_detector = self.diag_detector

        # --- Combinador de esquinas (Bloque 3, Paso 4) ---
        if "combinador_esquina" not in self.net.modules:
            self.combinador_esquina = EsquinaBlock(
                zone_z=(92.0, 100.0),
                v_z=70.0,
                h_z=72.0,
                output_z=80.0,
                corner_threshold=0.20,
            )
            self.combinador_esquina.register(self.net)
        else:
            self.combinador_esquina = self.net.modules["combinador_esquina"]
            logging.info("🔲 [Esquinas] Recuperado desde persistencia.")
        self.vision._combinador_esquina = self.combinador_esquina
        # ----------------------------------------------------------

        # --- DETECTOR DE CURVAS (Bloque 2d): curvatura local → z=78 ---
        if "detector_curvas" not in self.net.modules:
            self.curva_detector = CurvaDetectorBlock(
                zone_z=(110.0, 120.0),
                output_z=78.0,
            )
            self.curva_detector.register(self.net)
        else:
            self.curva_detector = self.net.modules["detector_curvas"]
            logging.info("〇 [Curvas] Recuperado desde persistencia.")
        self.vision._curva_detector = self.curva_detector

        # --- DETECTOR DE SIMETRÍA (Bloque 2e): simetría bilateral/radial → z=82,84 ---
        if "detector_simetria" not in self.net.modules:
            self.simetria_detector = SimetriaDetectorBlock(
                zone_z=(120.0, 130.0),
                sym_v_z=82.0,
                sym_h_z=84.0,
            )
            self.simetria_detector.register(self.net)
        else:
            self.simetria_detector = self.net.modules["detector_simetria"]
            logging.info("🪞 [Simetría] Recuperado desde persistencia.")
        self.vision._simetria_detector = self.simetria_detector

        # --- COMBINADOR DE JUNCTIONS (Bloque 3b): cruces de bordes → z=86 ---
        if "combinador_junction" not in self.net.modules:
            self.junction_detector = JunctionBlock(
                zone_z=(130.0, 140.0),
                junction_z=86.0,
            )
            self.junction_detector.register(self.net)
        else:
            self.junction_detector = self.net.modules["combinador_junction"]
            logging.info("✚ [Junctions] Recuperado desde persistencia.")
        self.vision._junction_detector = self.junction_detector
        # ----------------------------------------------------------

        # --- DETECTOR DE LETRAS (Bloque 5): scores discriminantes → z=37-53 ---
        if "detector_letras" not in self.net.modules:
            self.letra_detector = LetraDetectorBlock(
                zone_z=(55.0, 65.0),
                letra_threshold=0.55,
                inject_strength=6.0,
                curv_z=78.0,    # FIX v1.5.9: explícito para coincidir con CurvaDetector.output_z
            )
            self.letra_detector.register(self.net)
        else:
            self.letra_detector = self.net.modules["detector_letras"]
            logging.info("🔤 [Letras] Recuperado desde persistencia.")

        self.letra_detector._lang_module = self.lang_module
        self.vision._letra_detector = self.letra_detector

        # v1.9: evidencia_* permanece observacional y sin cableado neuronal.

    def _wait_event_queue_backpressure(self, high_watermark=22000, low_watermark=10000, timeout_s=8.0):
        """Espera drenaje sin borrar eventos de entrada.

        La versión anterior podía eliminar eventos sensoriales por mirar solo
        ``strength``. Ahora el origen está etiquetado dentro del evento y el
        alivio selectivo solo elimina cascadas sinápticas débiles ya listas.
        """
        deadline = time.time() + float(timeout_s)
        relief_done = False
        while self.is_running and not self._training_stop_event.is_set():
            with self.net.lock:
                qsize = len(getattr(self.net, "event_queue", []))
            if qsize <= int(high_watermark):
                return True

            if not relief_done and time.time() >= deadline - float(timeout_s) * 0.5:
                try:
                    removed = self.net.shed_weak_ready_synaptic_events(max_remove=5000, strength_threshold=0.35)
                    if removed:
                        logging.warning("🧹 [Trial] Alivio selectivo: eliminados %d eventos sinápticos débiles; entradas sensoriales preservadas.", removed)
                except Exception as exc:
                    logging.debug(f"[Trial] Alivio selectivo omitido: {exc}")
                relief_done = True

            if time.time() >= deadline:
                try:
                    with self.net.lock:
                        self.net.event_queue_stats["backpressure_timeouts"] += 1
                        q_final = len(self.net.event_queue)
                    logging.warning("⚠️ [Trial] Cola alta sin drenaje completo (%d); se continúa sin vaciar eventos de entrada.", q_final)
                except Exception:
                    pass
                return True
            time.sleep(0.01)
        return False

    def _wait_for_evidence_processing(self, idx_ev, target_t, timeout_s=1.0):
        """Espera a que al menos una neurona de evidencia haya sido procesada."""
        if idx_ev is None or len(idx_ev) == 0:
            return False
        deadline = time.time() + float(timeout_s)
        target_t = float(target_t)
        idx_ev = np.asarray(idx_ev, dtype=int)
        while self.is_running and not self._training_stop_event.is_set():
            with self.net.lock:
                last_sp = self.net.last_spike[idx_ev].copy()
            if np.any(last_sp >= target_t):
                return True
            if time.time() >= deadline:
                return False
            time.sleep(0.005)
        return False

    def trigger_letter_training(self, letter):
        import threading
        import time
        import numpy as np
        import cv2
        import logging
        import traceback

        # --- FIX CLAUDE: Watchdog para evitar ráfagas concurrentes y destrabar trials colgados ---
        if getattr(self, '_shutdown_in_progress', False) or self._training_stop_event.is_set() or not self.is_running:
            logging.info(f"⛔ [Training] Ignorado '{letter}': cierre de simulación en curso.")
            return

        tiempo_actual = time.time()
        if getattr(self, '_modo_entrenamiento', False):
            # No liberar el semáforo por tiempo: un trial puede tardar más de
            # 5 s cuando el Event Queue está bajo carga. Liberarlo aquí permitía
            # dos trials simultáneos y mezclaba evidencia.
            print(f"⛔ [Trial] Trial en curso, ignorando '{letter}'", flush=True)
            return
        
        self._trial_start_timestamp = tiempo_actual
        # --- FIX: el flag se marca ACÁ, síncrono, antes de lanzar el hilo ---
        self._modo_entrenamiento = True
        # -----------------------------------------------------------------------------------------

        def training_task():
            probe_trial_open = False
            authoritative_prediction = None
            authoritative_certainty = 0.0
            authoritative_source = None
            termination_reason = "incomplete_exit"
            trial_complete = False
            # IA 1.6.27: la cantidad de frames debe existir antes de abrir
            # el contrato del trial y antes de notificarla al módulo de lenguaje.
            # En 1.6.25 el valor se asignaba más abajo, provocando UnboundLocalError
            # y dejando todos los trials en 0/60.
            frames = 60
            try:
                logging.info(f"🎓 [Curriculum] Iniciando trial de entrenamiento: Letra '{letter}'")
                try:
                    self.causal_probe.begin_trial(
                        label=f"letter:{letter}",
                        ground_truth=letter,
                        t=getattr(self.net, "current_time", 0.0),
                        expected_frames=60,
                        vision=self.vision,
                        target_xy=(0.5, 0.5),
                    )
                    probe_trial_open = True
                except Exception as probe_exc:
                    logging.debug(f"[CausalProbe] Apertura de trial omitida: {probe_exc}")
                
                lang = self.net.modules.get("lenguaje")
                if lang:
                    # --- FIX B: protect_band y precharge selectivo ANTES de abrir
                    # la ventana de medición causal. Ocurre fuera de begin_clean_trial
                    # por lo que no contamina la medición. Solo eleva la energía de
                    # las neuronas de evidencia_* para que puedan disparar al recibir
                    # señal visual; no inyecta spikes ni altera pesos/sinapsis.
                    if hasattr(lang, 'protect_band'):
                        try:
                            lang.protect_band(self.net, z_range=(37.0, 53.0))
                        except Exception as _pb_exc:
                            logging.debug(f"[Trial] protect_band omitido: {_pb_exc}")
                    # Precharge selectivo: solo neuronas conectadas a evidencia_*.
                    for sym in lang.symbols:
                        _port = lang.input_ports.get(f"evidencia_{sym}")
                        if _port is None:
                            continue
                        for _nidx in _port.connected_neurons:
                            if 0 <= _nidx < self.net.n and self.net.active[_nidx]:
                                self.net.energy[_nidx] = max(float(self.net.energy[_nidx]), 1.2)
                    # --- Fin FIX B --------------------------------------------------

                    # Durante el trial los puertos quedan FIJOS. No se recablea,
                    # no se reperfilan pools ni se inyectan spikes artificiales.
                    for sym in lang.symbols:
                        port = lang.input_ports.get(f"evidencia_{sym}")
                        if port is not None:
                            port.value = 0.0
                    # FIX 1.5.6: señal para que el corazón pause durante el trial
                    self.net._causal_trial_active = True
                    if hasattr(lang, 'begin_clean_trial'):
                        lang.begin_clean_trial(self.net.current_time, net=self.net, target=letter)
                    if hasattr(lang, 'set_trial_frame_expectation'):
                        lang.set_trial_frame_expectation(frames)
                    lang._prev_evidence = {sym: 0.0 for sym in lang.symbols}
                    lang._evidence_cooldown = {sym: False for sym in lang.symbols}
                    print(f"🧪 [Trial] Modo limpio: protect_band+precharge aplicados antes; "
                          f"conexiones congeladas durante ventana causal.", flush=True)
                    # FIX 1.4.10: diagnóstico profundo de sources al inicio del trial
                    if hasattr(lang, 'get_readout_telemetry'):
                        try:
                            import numpy as _np
                            _srcs = _np.asarray(lang.get_readout_source_indices(), dtype=int)
                            _srcs = _srcs[_srcs < self.net.n]
                            _n_src = len(_srcs)
                            if _n_src:
                                _active_src   = int(_np.sum(self.net.active[_srcs]))
                                _ever_fired   = int(_np.sum(self.net.last_spike[_srcs] > 0))
                                _energy_mean  = float(_np.mean(self.net.energy[_srcs]))
                                _vthresh_mean = float(_np.mean(self.net.v_thresh[_srcs]))
                                _vmem_mean    = float(_np.mean(self.net.membrane_potential[_srcs]))
                                # Conexiones entrantes: cuántas de las fuentes tienen al menos 1 presynaptic
                                # (no hay lista de entradas directa; aproximamos contando cuántos sources
                                #  aparecen como TARGET en las connections de otras neuronas — muy caro,
                                #  así que solo contamos cuántos tienen connections salientes > 0)
                                _with_out_conns = int(sum(1 for i in _srcs if len(self.net.connections[int(i)]) > 0))
                                _pool_sizes = {s: len(lang._readout_pools.get(s, [])) for s in lang.symbols}
                                _pool_vthresh = {}
                                for _s in lang.symbols:
                                    _p = lang._readout_pools.get(_s, [])
                                    if _p:
                                        _pool_vthresh[_s] = round(float(self.net.v_thresh[_p[0]]), 1)
                                print(
                                    f"🔬 [Readout-init] sources: total={_n_src} | activas={_active_src} | "
                                    f"alguna_vez_disparó={_ever_fired} | con_conexiones_salientes={_with_out_conns} | "
                                    f"energy_mean={_energy_mean:.3f} | v_thresh_mean={_vthresh_mean:.1f} | "
                                    f"vmem_mean={_vmem_mean:.3f}",
                                    flush=True
                                )
                                print(
                                    f"🔬 [Readout-init] pools: sizes={_pool_sizes} | v_thresh={_pool_vthresh}",
                                    flush=True
                                )
                            else:
                                print("🔬 [Readout-init] WARN: readout sin fuentes registradas", flush=True)
                        except Exception as _re:
                            print(f"🔬 [Readout-init] Error diagnóstico: {_re}", flush=True)
                else:
                    logging.warning(f"⚠️ [Trial] No se encontró el módulo de lenguaje al iniciar.")
                
                pos = (0.5, 0.5)
                self.vision.fovea_center_x, self.vision.fovea_center_y = pos
                self.vision.vel_x = self.vision.vel_y = 0.0
                
                activas = set()
                evidencia_disparadas = set()

                # --- FIX LOCK: capturar puerto UNA sola vez antes del loop, sin lock.
                # El puerto no cambia durante el trial. Así el loop no necesita
                # pedir net.lock para acceder al módulo de lenguaje en cada frame.
                lang_pre = self.net.modules.get("lenguaje")
                ev_indices = {}
                for sym in getattr(lang_pre, "symbols", []) if lang_pre else []:
                    port = lang_pre.input_ports.get(f"evidencia_{sym}")
                    if port and port.connected_neurons:
                        ev_indices[sym] = np.asarray([i for i in port.connected_neurons if 0 <= i < self.net.n], dtype=int)
                idx_readout_sources = np.asarray(
                    lang_pre.get_readout_source_indices() if lang_pre and hasattr(lang_pre, 'get_readout_source_indices') else [], dtype=int
                )

                with self.net.lock:
                    t_inicio_trial = self.net.current_time
                    if hasattr(self.net, "begin_causal_trial"):
                        self.net.begin_causal_trial(f"letter:{letter}:{getattr(self.causal_probe, '_trial_counter', 0)}", t_inicio_trial)

                for frame_i in range(frames):
                    frame_wall_started = time.perf_counter()
                    # 1.6.21: clock gate. Un frame lógico avanza exactamente
                    # CAUSAL_TRIAL_FRAME_DT_MS de tiempo neuronal, independientemente
                    # de cuánto tarde Python/cola en preparar el frame.
                    with self.net.lock:
                        frame_sim_started = float(self.net.current_time)
                        frame_t_target = float(t_inicio_trial) + (frame_i + 1) * float(CAUSAL_TRIAL_FRAME_DT_MS)
                        if hasattr(self.net, "begin_causal_frame"):
                            self.net.begin_causal_frame(frame_i, frame_t_target, int(CAUSAL_EVENT_ATTEMPT_BUDGET_PER_FRAME))
                        if float(self.net.current_time) < frame_t_target:
                            self.net.current_time = frame_t_target
                        frame_t = float(self.net.current_time)
                    try:
                        queue_before_frame = int((self.net.get_event_queue_stats() or {}).get("queue_size", 0))
                    except Exception:
                        queue_before_frame = None
                    if self._training_stop_event.is_set() or not self.is_running:
                        termination_reason = "simulation_shutdown"
                        print(f"🛑 [Trial] Cancelado durante '{letter}' por cierre de simulación.", flush=True)
                        return

                    # Nunca perder un frame: el Probe y el ejecutor deben ver
                    # exactamente la misma cantidad (60/60).
                    if not self._wait_event_queue_backpressure():
                        termination_reason = "queue_backpressure_timeout"
                        raise RuntimeError(
                            f"Event Queue no pudo recuperar capacidad antes de frame {frame_i + 1}/{frames}"
                        )

                    frame  = self.scr_processor.generate_letter_frame(letter, pos)
                    coords, _, _ = self.scr_processor.get_activity_coords(focus_pt=pos, virtual_frame=frame)

                    scale   = getattr(self.scr_processor, 'scale_factor', 1.0)
                    resized = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1.0 else frame
                    static  = self.scr_processor.get_static_contrast_coords(resized, pos)

                    if frame_i == 0 or frame_i == frames - 1:
                        print(f"🩺 [Diag-Trial] Frame {frame_i}/{frames}: "
                              f"{len(coords)} puntos movimiento + {len(static)} estáticos "
                              f"= {len(coords) + len(static)} totales", flush=True)

                    self.vision.update_from_external(coords + static)
                    self.vision.transport_to_net(self.net, frame_t)

                    # Esperar a que SpikingThread procese al menos una neurona
                    # del pool objetivo antes de medir evidencia. Sin esto,
                    # last_spike suele ir detrás de la inyección y el Probe ve 0.
                    idx_wait = np.concatenate(list(ev_indices.values())) if ev_indices else np.asarray([], dtype=int)
                    self._wait_for_evidence_processing(idx_wait, frame_t, timeout_s=0.12)

                    # v1.6.14: competencia intratrial del readout con dominancia clara.
                    if lang is not None and hasattr(lang, 'apply_lateral_pool_inhibition'):
                        try:
                            lang.apply_lateral_pool_inhibition(self.net, t=float(self.net.current_time))
                        except Exception as _lat_exc:
                            logging.debug(f"[Readout] inhibición lateral omitida: {_lat_exc}")

                    # Actualizar inmediatamente la vista lógica de evidencia.
                    # Los spikes pueden seguir en cola, así que esta lectura se
                    # toma como telemetría intermedia; la medición neuronal real
                    # se consolida al final después del grace period.
                    if lang_pre is not None:
                        lang_pre.update_evidence_values(self.net, frame_t)

                    probe_trial = getattr(self, "causal_probe", None)
                    if probe_trial is not None:
                        try:
                            try:
                                queue_after_frame = int((self.net.get_event_queue_stats() or {}).get("queue_size", 0))
                            except Exception:
                                queue_after_frame = None
                            probe_trial.observe_frame(
                                t=frame_t,
                                outputs=getattr(self.vision, "_last_probe_outputs", {}),
                                net=self.net,
                                wall_started=frame_wall_started,
                                wall_finished=time.perf_counter(),
                                simulation_started=frame_sim_started,
                                queue_before=queue_before_frame,
                                queue_after=queue_after_frame,
                                vision=self.vision,
                            )
                        except Exception as exc:
                            logging.debug(f"[CausalProbe] trial observe omitido: {exc}")

                    if lang_pre is not None and hasattr(lang_pre, 'note_trial_frame'):
                        lang_pre.note_trial_frame(frame_i + 1)

                    # --- FIX LOCK: solo copiamos los arrays numpy, sin lógica adentro.
                    # El lock se libera en microsegundos, sin bloquear SlowBioThread.
                    with self.net.lock:
                        snap_spike  = self.net.last_spike[:self.net.n].copy()
                        snap_pos_z  = self.net.positions[:self.net.n, 2].copy()
                        snap_active = self.net.active[:self.net.n].copy()

                    # Todo el procesamiento fuera del lock con las copias locales
                    if idx_readout_sources.size:
                        recientes_src = idx_readout_sources[(snap_spike[idx_readout_sources] > t_inicio_trial) & snap_active[idx_readout_sources]]
                        activas.update(int(i) for i in recientes_src)
                    for _sym, _idx in ev_indices.items():
                        if _idx.size:
                            evidencia_disparadas.update(int(i) for i in _idx[snap_spike[_idx] > t_inicio_trial])

                    time.sleep(0.005)

                # --- Evaluación final: competencia entre todos los pools ---
                lang = self.net.modules.get("lenguaje")
                if lang is None:
                    termination_reason = "missing_language_module"
                    print("❌ [Trial] No se encontró módulo 'lenguaje' al finalizar", flush=True)
                    return
                idx_wait = np.concatenate(list(ev_indices.values())) if ev_indices else np.asarray([], dtype=int)
                if idx_wait.size:
                    self._wait_for_evidence_processing(idx_wait, t_inicio_trial, timeout_s=1.0)
                try:
                    lang.wait_for_trial_readout_settle(self.net, timeout_s=lang.readout_settle_timeout_s)
                except Exception as _settle_exc:
                    logging.debug(f"[Readout] Espera de asentamiento omitida: {_settle_exc}")

                now_t = float(self.net.current_time)
                # IA 1.6.28: después de que la percepción terminó y el readout
                # se asentó, ejecutar workspace latente sobre el snapshot cortical.
                # No genera eventos ni altera Event Queue.
                latent_workspace_result = {}
                if hasattr(lang, 'run_latent_workspace_reasoning'):
                    try:
                        latent_workspace_result = lang.run_latent_workspace_reasoning(self.net)
                        if latent_workspace_result:
                            _lp = latent_workspace_result.get('stage_predictions', [])
                            _ls = latent_workspace_result.get('stage_scores', {})
                            print(
                                f"🧠↻ [LatentWorkspace] steps={latent_workspace_result.get('steps', 0)} "
                                f"| pred={'→'.join('None' if x is None else str(x) for x in _lp)} "
                                f"| converged={latent_workspace_result.get('converged', False)} "
                                f"| Δlast={latent_workspace_result.get('state_delta_last', 0.0):.4f} "
                                f"| authoritative={latent_workspace_result.get('authoritative', False)}",
                                flush=True,
                            )
                    except Exception as _latent_exc:
                        logging.debug(f"[LatentWorkspace] omitido: {_latent_exc}")
                # IA 1.6.27: el clasificador del trial debe leer el readout neuronal
                # aprendible, no el puerto observacional evidencia_* del detector.
                # Dejamos ambos canales medidos por separado.
                # v1.2: la clasificación autoritativa no cae silenciosamente
                # al detector heurístico. Eso mezclaba percepción con readout y
                # hacía parecer causal una predicción que el readout no produjo.
                if hasattr(lang, 'readout_scores') and getattr(lang, '_readout_ready', False):
                    scores = lang.readout_scores(self.net, t=now_t, window_ms=lang.evidence_window_ms)
                else:
                    scores = {sym: 0.0 for sym in getattr(lang, 'symbols', [])}

                try:
                    from config import LATENT_WORKSPACE_AUTHORITATIVE
                except Exception:
                    LATENT_WORKSPACE_AUTHORITATIVE = False

                trial_rt = lang.get_trial_readout_activity() if hasattr(lang, 'get_trial_readout_activity') else {}
                readout_active = bool(
                    int(trial_rt.get('source_spikes', 0)) > 0 or
                    sum(int(v) for v in trial_rt.get('pool_spikes', {}).values()) > 0
                )
                authoritative_source = "learned_readout" if readout_active else "learned_readout_inactive"

                # 1.6.28b: la autoridad latente se resuelve aquí y no se vuelve a
                # sobrescribir después. Por defecto continúa desactivada.
                if LATENT_WORKSPACE_AUTHORITATIVE and latent_workspace_result.get('active'):
                    latent_scores = latent_workspace_result.get('best_scores') or latent_workspace_result.get('stage_scores', {}).get(
                        latent_workspace_result.get('best_stage') or f"LATENT-{latent_workspace_result.get('steps', 0)}", {})
                    if any(float(v) > 0.0 for v in latent_scores.values()):
                        scores = {sym: float(latent_scores.get(sym, 0.0)) for sym in lang.symbols}
                        authoritative_source = "latent_workspace"

                if not readout_active or not scores:
                    scores = {sym: 0.0 for sym in getattr(lang, 'symbols', [])}
                    prediction = None
                    certainty = 0.0
                    total_score = 0.0
                else:
                    total_score = sum(max(0.0, float(v)) for v in scores.values())
                    ordered = sorted(scores.items(), key=lambda kv: (-float(kv[1]), kv[0]))
                    prediction = ordered[0][0] if ordered else None
                    top = max(0.0, float(ordered[0][1])) if ordered else 0.0
                    second = max(0.0, float(ordered[1][1])) if len(ordered) > 1 else 0.0
                    certainty = ((top - second) / (top + second + 1e-9)) if top > 0 else 0.0
                    # v1.7: abstención sólo cuando no existe separación medible.
                    # Evita que una diferencia minúscula entre pools mantenga
                    # prediction=null durante todos los trials.
                    if top <= 1e-9 or (abs(top - second) <= 1e-12 and top <= 0.15):
                        prediction = None
                        certainty = 0.0
                # v1.8: corrección de semántica de evaluación. Una predicción
                # correcta no deja de ser correcta solo porque el margen sea
                # pequeño. La certeza se conserva como métrica independiente
                # para saber cuándo la red además está segura.
                prediction_correct = bool(
                    readout_active and prediction == letter and total_score > 0.0
                )
                confident_prediction = bool(
                    prediction_correct and certainty >= 0.05
                )
                exito = prediction_correct
                authoritative_prediction = prediction
                authoritative_certainty = certainty
                print(
                    f"🧠 [Readout-end] GT={letter} | pred={prediction} | "
                    + ", ".join(f"{k}:{v:.3f}" for k,v in scores.items())
                    + f" | source={authoritative_source} | "
                    f"{'✅ CORRECTO' if prediction_correct else '❌ INCORRECTO'} "
                    f"| confianza={'✅' if confident_prediction else 'baja'}"
                    , flush=True
                )
                # FIX 1.4.9: diagnóstico del readout al final del trial
                if lang is not None and hasattr(lang, 'get_readout_telemetry'):
                    try:
                        _rt_end = lang.get_readout_telemetry(self.net)
                        print(
                            f"🔬 [Readout-end] sources_fired={_rt_end.get('readout_sources_fired',0)}/{_rt_end.get('readout_sources_total',0)} | "
                            + " | ".join(
                                f"{s}: pool_spikes={_rt_end.get(f'readout_{s}_pool_spikes',0)} "
                                f"w_mean={_rt_end.get(f'readout_{s}_weight_mean',0):.3f} "
                                f"(Δ={_rt_end.get(f'readout_{s}_weight_delta',0):+.3f}) "
                                f"V_mem={_rt_end.get(f'readout_{s}_pool_v_mean',0):.2f}"
                                for s in getattr(lang, "symbols", [])
                            ),
                            flush=True
                        )
                    except Exception as _re:
                        print(f"🔬 [Readout-end] Error diagnóstico: {_re}", flush=True)

                changed = 0
                trial_complete = (frame_i + 1) >= frames
                if trial_complete and hasattr(lang, 'apply_readout_learning'):
                    active_sources_for_learning = trial_rt.get("source_indices", []) or list(activas)
                    changed = lang.apply_readout_learning(self.net, active_sources_for_learning, letter, prediction=prediction)
                    print(f"🧬 [Readout-learning] changed={changed} | source_spikes={trial_rt.get('source_spikes',0)} | pool_total={sum(int(v) for v in trial_rt.get('pool_spikes',{}).values())} | pred={prediction or 'none'}", flush=True)

                latent_learning = {}
                if hasattr(lang, 'learn_latent_workspace_trial'):
                    try:
                        latent_learning = lang.learn_latent_workspace_trial(letter, completed=trial_complete)
                        if latent_learning:
                            print(
                                f"🧠↻ [Latent-learning] applied={latent_learning.get('learning_applied', False)} "
                                f"| decoderΔ={latent_learning.get('decoder_sum_abs_delta', 0.0):.5f} "
                                f"| recurrentΔ={latent_learning.get('recurrent_sum_abs_delta', 0.0):.5f} "
                                f"| contextΔ={latent_learning.get('context_sum_abs_delta', 0.0):.5f}",
                                flush=True,
                            )
                    except Exception as _latent_learn_exc:
                        logging.debug(f"[Latent-learning] omitido: {_latent_learn_exc}")

                if trial_complete:
                    termination_reason = "completed"

                # v1.6.17: congelar la fuente causal de verdad del trial
                # inmediatamente antes de entregarla al Probe. Esto evita que
                # ``observe_frame`` (que ocurre por frame) y el asentamiento
                # final describan acumuladores distintos.
                trial_readout_snapshot = (
                    lang.snapshot_trial_readout()
                    if hasattr(lang, 'snapshot_trial_readout') else
                    lang.get_trial_readout_activity() if hasattr(lang, 'get_trial_readout_activity') else {}
                )
                if trial_readout_snapshot:
                    print(
                        f"🔒 [Readout-snapshot] pools={trial_readout_snapshot.get('pool_spikes', {})} "
                        f"total={trial_readout_snapshot.get('pool_spikes_total', 0)} "
                        f"unique={trial_readout_snapshot.get('pool_unique', {})} "
                        f"consistent={trial_readout_snapshot.get('pool_ledger_consistent', False)}",
                        flush=True,
                    )

                # v1.9: un trial de lectura no debe recompensar/deprimir toda la red biológica.
                # La plasticidad del readout ya recibe la señal target/competidor.
                # La dopamina global queda reservada para tareas visuales/ambientales.

                if hasattr(self, 'curriculum'):
                    evidencia_real = float(scores.get(letter, 0.0))
                    self.curriculum.record_trial(letter, evidencia_real, exito, complete=trial_complete, margin=authoritative_certainty)
                    self.net.curriculum_state = self.curriculum.to_dict()
                    print(f"🎓 [Currícula] Trial registrado: {self.curriculum.status()}", flush=True)

                if exito:
                    maestro = lang.output_ports.get(f"maestro_{letter}")
                    if maestro:
                        for idx in list(maestro.connected_neurons)[:20]:
                            self.net.receive_spike(idx, 5.0, self.net.current_time)
                print(f"🧬 [Readout] pesos actualizados={changed} | fuentes activas={len(activas)}", flush=True)

                # Cruce con detector de letras — ground truth
                if hasattr(self, 'letra_detector') and self.letra_detector is not None:
                    self.letra_detector.record_ground_truth(letter)
                
            except BaseException as e:
                termination_reason = f"exception:{type(e).__name__}"
                print(f"❌ [Curriculum] Excepción en trial '{letter}': {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
            finally:
                if probe_trial_open:
                    try:
                        # Registrar en CausalProbe la misma predicción autorizada que
                        # se evaluó para el trial. No volver a leer evidencia_* aquí:
                        # esos puertos son observacionales y podrían ocultar un readout
                        # neuronal correcto o introducir un valor stale al cerrar.
                        pred = authoritative_prediction
                        certainty = float(authoritative_certainty)
                        source = authoritative_source
                        summary = self.causal_probe.end_trial(
                            prediction=pred,
                            certainty=certainty,
                            t=getattr(self.net, "current_time", 0.0),
                            prediction_source=source,
                            termination_reason=termination_reason,
                            readout_activity=(
                                lang.get_last_trial_readout()
                                if hasattr(lang, 'get_last_trial_readout') else
                                trial_readout_snapshot if 'trial_readout_snapshot' in locals() else None
                            ),
                        )
                        if summary and summary.get("candidate"):
                            logging.info(
                                "🧪 [CausalProbe] Candidato: %s",
                                summary["candidate"],
                            )
                    except Exception as probe_exc:
                        logging.debug(f"[CausalProbe] Cierre de trial omitido: {probe_exc}")
                try:
                    lang_cleanup = self.net.modules.get("lenguaje")
                    if hasattr(self.net, "end_causal_trial"):
                        self.net.end_causal_trial()
                    else:
                        self.net._causal_trial_active = False
                    if lang_cleanup is not None and hasattr(lang_cleanup, 'end_clean_trial'):
                        lang_cleanup.end_clean_trial()
                except Exception as cleanup_exc:
                    logging.debug(f"[Trial] Limpieza de baseline omitida: {cleanup_exc}")
                if (self._trial_commit_journal is not None and trial_complete and lang is not None):
                    try:
                        snap = (lang.get_last_trial_readout() if hasattr(lang, 'get_last_trial_readout') else {}) or {}
                        latent = snap.get("latent_workspace", {}) if isinstance(snap, dict) else {}
                        self._trial_commit_journal.commit(
                            trial_id=f"letter:{letter}:{getattr(lang, '_latent_trial_sequence', 0)}",
                            sequence=int(getattr(lang, '_latent_trial_sequence', 0)),
                            label=str(letter),
                            current_time=float(getattr(self.net, 'current_time', 0.0)),
                            frames=int(getattr(lang, '_trial_frame_count', 0)),
                            completed=True,
                            trial_mode=str(getattr(lang, '_latent_trial_mode', 'train')),
                            learning_steps=int(getattr(getattr(lang, '_latent_workspace', None), 'learning_steps', 0)),
                            latent_summary={
                                "h0_prediction": latent.get("h0_prediction"),
                                "accepted_stage": latent.get("accepted_stage", "H0"),
                                "raw_best_stage": latent.get("raw_best_stage", latent.get("best_stage", "H0")),
                                "halt_reason": latent.get("halt_reason"),
                            },
                            readout_summary={
                                "prediction": snap.get("prediction_source"),
                                "pool_spikes": snap.get("pool_spikes", {}),
                            },
                        )
                    except Exception as journal_exc:
                        logging.debug(f"[TrialCommit] No se pudo escribir frontera durable: {journal_exc}")
                self._modo_entrenamiento = False
    
        self._start_training_thread(training_task, name=f"LetterTraining-{letter}")

    def trigger_word_training(self, word):
        """Etapa 3 — trial de secuencia: muestra letra1, un hueco en blanco,
        y letra2, en la MISMA posición fija (0.5, 0.5) que ya usan los
        trials de letra individual (Etapa 1/Generalización con posición
        variable queda para después). Comparte el watchdog y el flag
        _modo_entrenamiento con trigger_letter_training a propósito: son el
        mismo recurso (la red no puede estar en dos trials a la vez), así
        que deben turnarse por el mismo semáforo.

        v1.6.9: el propio loop del trial consume explícitamente
        lang.check_edge_events() después de cada frame. Esto evita depender
        del worker_vision, que durante _modo_entrenamiento no actualiza la
        vía observacional normal."""
        import threading
        import time
        import numpy as np
        import cv2
        import logging
        import traceback

        if len(word) != 2:
            print(f"⚠️ [Trial-Palabra] '{word}' no es un bigrama válido, se ignora.", flush=True)
            return
        l1, l2 = word[0], word[1]

        if getattr(self, '_shutdown_in_progress', False) or self._training_stop_event.is_set() or not self.is_running:
            logging.info(f"⛔ [Training] Ignorado '{word}': cierre de simulación en curso.")
            return

        tiempo_actual = time.time()
        if getattr(self, '_modo_entrenamiento', False):
            # No liberar el semáforo por tiempo: un trial puede tardar más de
            # 5 s bajo carga del Event Queue. Liberarlo mezclaba trials.
            print(f"⛔ [Trial] Trial en curso, ignorando palabra '{word}'", flush=True)
            return

        self._trial_start_timestamp = tiempo_actual
        # --- FIX: mismo motivo que en trigger_letter_training, ver comentario ahí ---
        self._modo_entrenamiento = True

        def training_task():
            try:
                logging.info(f"🎓 [Curriculum] Iniciando trial de SECUENCIA: '{word}' ({l1} → {l2})")

                lang = self.net.modules.get("lenguaje")
                if not lang:
                    logging.warning("⚠️ [Trial-Palabra] No se encontró el módulo de lenguaje al iniciar.")
                    return

                lang.auto_connect(self.net)
                lang._prev_evidence = {sym: 0.0 for sym in lang.symbols}
                lang._evidence_cooldown = {sym: False for sym in lang.symbols}
                lang.reset_sequence_state()
                print(f"🔄 [Trial-Palabra] Estado reseteado para '{word}'", flush=True)

                pos = (0.5, 0.5)
                self.vision.fovea_center_x, self.vision.fovea_center_y = pos
                self.vision.vel_x = self.vision.vel_y = 0.0

                with self.net.lock:
                    t_inicio_trial = self.net.current_time

                # Guion del trial: letra1 x N frames, hueco en blanco, letra2 x N frames.
                guion = ([l1] * WORD_TRIAL_FRAMES_PER_LETTER
                         + [None] * WORD_TRIAL_GAP_FRAMES
                         + [l2] * WORD_TRIAL_FRAMES_PER_LETTER)

                for letra_frame in guion:
                    if self._training_stop_event.is_set() or not self.is_running:
                        print(f"🛑 [Trial-Palabra] Cancelado durante '{word}' por cierre de simulación.", flush=True)
                        return
                    if letra_frame is not None:
                        frame = self.scr_processor.generate_letter_frame(letra_frame, pos)
                    else:
                        frame = self.scr_processor.generate_mission_frame()

                    coords, _, _ = self.scr_processor.get_activity_coords(focus_pt=pos, virtual_frame=frame)
                    scale = getattr(self.scr_processor, 'scale_factor', 1.0)
                    resized = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1.0 else frame
                    static = self.scr_processor.get_static_contrast_coords(resized, pos)

                    frame_t = self.net.current_time
                    self.vision.update_from_external(coords + static)
                    self.vision.transport_to_net(self.net, frame_t)

                    # v1.6.9: durante secuencias se actualiza la evidencia neural
                    # separadamente y luego se consumen los flancos del detector
                    # heurístico para registrar el orden O->X / X->O.
                    lang.update_evidence_values(self.net, frame_t)
                    lang.check_edge_events(self.net, frame_t)

                    time.sleep(1 / 30)

                with self.net.lock:
                    t_fin_trial = self.net.current_time

                lang.update_sequence_evidence(self.net, t_fin_trial,
                                               window_start=t_inicio_trial,
                                               max_gap_ms=WORD_MAX_GAP_MS)
                lang.debug_event_log()

                port = lang.input_ports.get(f"secuencia_{word}")
                evidencia_real = float(port.value) if port else 0.0
                exito = evidencia_real > 0.05

                print(f"🩺 [Trial-Palabra-end] secuencia_{word}={evidencia_real:.3f} "
                      f"| orden esperado {l1}→{l2}", flush=True)

                if exito:
                    if hasattr(self.net, 'apply_dopamine_reward'):
                        self.net.apply_dopamine_reward(visual_success=True, habituacion=1.0)
                    print(f"✅ Trial-Palabra '{word}' exitoso. Evidencia de orden: {evidencia_real:.2f}", flush=True)

                    maestro = lang.output_ports.get(f"maestro_{word}")
                    if maestro:
                        for idx in list(maestro.connected_neurons)[:20]:
                            self.net.receive_spike(idx, 5.0, self.net.current_time)
                else:
                    print(f"⚠️ Trial-Palabra '{word}' débil. Evidencia de orden: {evidencia_real:.2f}", flush=True)

                if hasattr(self, 'word_curriculum'):
                    self.word_curriculum.record_trial(word, evidencia_real, exito)
                    self.net.word_curriculum_state = self.word_curriculum.to_dict()

            except Exception as e:
                logging.error(f"❌ [Curriculum] Excepción no capturada en trial-palabra '{word}': {e}")
                traceback.print_exc()
            finally:
                self._modo_entrenamiento = False

        self._start_training_thread(training_task, name=f"WordTraining-{word}")

    def _start_training_thread(self, target, name="TrainingThread"):
        """Inicia un hilo de entrenamiento rastreado y no-daemon.

        Los trials antiguos eran daemon y podían sobrevivir al apagado. Eso
        permitía que un `print()` ocurriera mientras CPython estaba en
        finalización, provocando `_enter_buffered_busy`.
        """
        holder = {}

        def runner():
            try:
                target()
            finally:
                thread_obj = holder.get("thread")
                if thread_obj is not None:
                    with self._training_threads_lock:
                        self._training_threads.discard(thread_obj)

        t = threading.Thread(target=runner, name=name, daemon=False)
        holder["thread"] = t
        with self._training_threads_lock:
            self._training_threads.add(t)
        t.start()
        return t

    def _stop_training_threads(self, timeout=10.0):
        """Cancela y espera cooperativamente los trials antes del guardado."""
        self._training_stop_event.set()
        self._modo_entrenamiento = False
        with self._training_threads_lock:
            threads = list(self._training_threads)
        if not threads:
            logging.info("✅ No quedan hilos de entrenamiento pendientes.")
            return True
        logging.info(f"🛑 Cancelando {len(threads)} hilo(s) de entrenamiento antes del guardado...")
        deadline = time.time() + max(1.0, float(timeout))
        while time.time() < deadline:
            vivos = [t for t in threads if t.is_alive()]
            if not vivos:
                break
            for t in vivos:
                t.join(timeout=0.10)
        with self._training_threads_lock:
            vivos = [t.name for t in self._training_threads if t.is_alive()]
        if vivos:
            logging.error(f"❌ No se puede garantizar un guardado consistente: hilos activos {vivos}")
            return False
        logging.info("✅ Todos los hilos de entrenamiento terminaron antes del guardado.")
        return True

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
            if bool(getattr(self.net, "_causal_trial_active", False)):
                # 1.6.21: LetterTraining owns the simulation clock and the visual
                # frame stream. No motor/reflex/exploration tick may advance time or
                # mutate the fovea while the causal trial is active.
                time.sleep(frame_time)
                continue

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
            # TEMPORAL: verificar identidad de objetos
            if hasattr(self, 'lang_module') and self.lang_module:
                modulo_en_net = self.net.modules.get("lenguaje")
                mismo_objeto = (self.lang_module is modulo_en_net)
                if not mismo_objeto:
                    print(f"🚨 [IDENTIDAD] lang_module y net.modules['lenguaje'] son DISTINTOS", flush=True)
                    print(f"   sim.lang_module id={id(self.lang_module)}", flush=True)
                    print(f"   net.modules id={id(modulo_en_net)}", flush=True)
                    
                    # CORRECCIÓN AUTOMÁTICA VISUAL: usar el que tiene los spikes
                    port_sim = self.lang_module.input_ports.get("evidencia_X")
                    port_net = modulo_en_net.input_ports.get("evidencia_X") if modulo_en_net else None
                    print(f"   evidencia_X en sim={port_sim.value:.4f if port_sim else 'N/A'}", flush=True)
                    print(f"   evidencia_X en net={port_net.value:.4f if port_net else 'N/A'}", flush=True)

            # CORRECCIÓN: Usar siempre la referencia incrustada en self.net
            lang = self.net.modules.get("lenguaje")
            if lang and not getattr(self, '_modo_entrenamiento', False):
                # Durante el trial, trigger_letter_training es el único lugar que
                # actualiza evidencia y flancos. Así no hay doble consumidor ni
                # frames extra en el Probe.
                lang.update_evidence_values(self.net, self.net.current_time)

            # La sonda debe leer la evidencia DESPUÉS de actualizar los puertos,
            # no dentro de Vision.transport_to_net() antes de ese paso.
            probe = getattr(self.vision, "_causal_probe", None)
            if probe is not None and not getattr(self, "_modo_entrenamiento", False):
                try:
                    probe.observe_frame(
                        t=self.net.current_time,
                        outputs=getattr(self.vision, "_last_probe_outputs", {}),
                        net=self.net,
                    )
                except Exception as exc:
                    logging.debug(f"[CausalProbe] observe_frame omitido: {exc}")

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
                readout_q = getattr(self.net, 'readout_event_queue', [])
                general_q = self.net.event_queue
                if readout_q or general_q:
                    qsize_now = len(readout_q) + len(general_q)
                    # Bajo carga normal usamos lotes pequeños; cuando hay
                    # backlog, aumentamos gradualmente para recuperar sin
                    # bloquear demasiado tiempo al i5.
                    batch_limit = 1000
                    # IA 1.6.25: durante el trial causal la cola ya usa
                    # agregación por destino+bucket; no hacemos coalescing por
                    # fuerza máxima porque podía borrar transiciones válidas.
                    if qsize_now > 23000:
                        batch_limit = 4000
                        # Bajo presión severa, colapsar eventos listos duplicados
                        # por destino antes de procesarlos. Esto reduce cascadas
                        # sin tocar eventos futuros y mantiene la entrada sensorial.
                        try:
                            self.net.coalesce_ready_event_backlog(max_events=8000)
                            qsize_now = len(self.net.event_queue)
                        except Exception as _coal_exc:
                            logging.debug(f"[Queue] coalesce omitido: {_coal_exc}")
                    elif qsize_now > 20000:
                        batch_limit = 6000
                    elif qsize_now > 5000:
                        batch_limit = 2000

                    # v1.4: drenaje prioritario del canal readout. La clasificación
                    # no debe esperar detrás de la cascada biológica general.
                    while getattr(self.net, 'readout_event_queue', None) and len(batch) < batch_limit:
                        rq = self.net.readout_event_queue
                        if rq[0][0] <= self.net.current_time:
                            batch.append(heapq.heappop(rq))
                            self.net.event_queue_stats["readout_processed"] += 1
                        else:
                            break
                    while self.net.event_queue and len(batch) < batch_limit:
                        if self.net.event_queue[0][0] <= self.net.current_time:
                            batch.append(heapq.heappop(self.net.event_queue))
                        else:
                            break

            # Procesamos el lote FUERA del lock
            if batch:
                 for event in batch:
                     try:
                         if len(event) >= 5:
                             t_ev, event_seq, idx, strength, origin = event[:5]
                         else:
                             t_ev, idx, strength, origin = event
                             event_seq = None
                     except ValueError:
                         t_ev, idx, strength = event[:3]
                         origin = event[3] if len(event) >= 4 else None
                         event_seq = None
                     try:
                         if hasattr(self.net, 'consume_causal_aggregated_event'):
                             event = self.net.consume_causal_aggregated_event(event)
                             if len(event) >= 5:
                                 t_ev, event_seq, idx, strength, origin = event[:5]
                         self.net.process_event(t_ev, idx, strength, origin=origin, event_seq=event_seq)
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
                qstats = self.net.get_event_queue_stats() if hasattr(self.net, 'get_event_queue_stats') else {
                    "queue_size": queue_size, "ready_count": 0, "future_count": 0
                }
                print(
                    f"👁️  Posición Fóvea Real: X={gaze_pos[0]:.4f}, Y={gaze_pos[1]:.4f} | "
                    f"Event Queue: {qstats.get('queue_size', queue_size)} | "
                    f"listos={qstats.get('ready_count', 0)} futuros={qstats.get('future_count', 0)}"
                )
                if getattr(self, '_modo_entrenamiento', False) and queue_size >= 14000:
                    print("🧯 [Trial] Governor de cola activo: reduciendo propagación débil.", flush=True)
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
        # 1.6.21-BIOFIX: la biología puede seguir siendo OBSERVADA durante un
        # trial causal aunque sus mecanismos mutadores/interventores permanezcan
        # aislados. Este contador evita confundir "no intervenir" con "no medir".
        self._causal_bio_observation_count = 0

        while self.is_running:
            now = time.time()
            if now - last_bio_tick >= 2.0:
                if bool(getattr(self.net, "_causal_trial_active", False)):
                    # 1.6.21-BIOFIX: ventana causal limpia. REM, evolución,
                    # metabolismo, recompensa y STDP global siguen fuera del trial.
                    # Pero la biología NO desaparece del log: hacemos una lectura
                    # estrictamente observacional del estado real, sin mutar red,
                    # regiones, dopamina, energía, reloj ni cola.
                    try:
                        with self.net.lock:
                            active_mask = self.net.active[:self.net.n]
                            active_count = int(np.count_nonzero(active_mask))
                            if active_count:
                                energy_active = self.net.energy[:self.net.n][active_mask]
                                bio_energy_mean = float(np.mean(energy_active))
                                bio_energy_min = float(np.min(energy_active))
                                bio_energy_max = float(np.max(energy_active))
                            else:
                                bio_energy_mean = bio_energy_min = bio_energy_max = 0.0

                            region_dopamines = [
                                float(getattr(r, "dopamine", 0.0))
                                for r in self.net.regions.values()
                            ]
                            if region_dopamines:
                                bio_dop_mean = float(np.mean(region_dopamines))
                                bio_dop_min = float(np.min(region_dopamines))
                                bio_dop_max = float(np.max(region_dopamines))
                            else:
                                bio_dop_mean = bio_dop_min = bio_dop_max = float(
                                    getattr(self.net, "dopamine_level", 0.0)
                                )

                            bio_dop_global = float(getattr(self.net, "dopamine_level", 0.0))
                            bio_state = str(getattr(self, "state", "unknown"))
                            heart_pulse = float(getattr(
                                getattr(getattr(self.net, "heart", None), "pulse_port", None),
                                "value", 0.0
                            ))
                            bio_time = float(getattr(self.net, "current_time", 0.0))

                        self._causal_bio_observation_count += 1
                        print(
                            "🧬 [BIO-OBS] Trial causal (solo lectura) | "
                            f"t={bio_time:.1f}ms | estado={bio_state.upper()} | "
                            f"Dopamina global={bio_dop_global:.3f} | "
                            f"regiones μ/min/max={bio_dop_mean:.3f}/{bio_dop_min:.3f}/{bio_dop_max:.3f} | "
                            f"Energía μ/min/max={bio_energy_mean:.3f}/{bio_energy_min:.3f}/{bio_energy_max:.3f} | "
                            f"vivas={active_count} | pulso={heart_pulse:.3f} | "
                            "intervención=0",
                            flush=True,
                        )
                    except Exception as e:
                        print(f"⚠️ [BIO-OBS] No se pudo leer el snapshot biológico: {e}", flush=True)

                    last_bio_tick = now
                    time.sleep(0.05)
                    continue
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
    
                        if (self.state == "sleep" and hasattr(self, 'output_module')
                                and not getattr(self, '_modo_entrenamiento', False)):
                            if self.output_module.memoria_replay:
                                self.output_module.replay_rem(self.net, intensidad=0.25)
                            else:
                                print(f"💤 [REM] Estado Sleep, memoria vacía.")
                        elif self.state == "sleep" and getattr(self, '_modo_entrenamiento', False):
                            print("🧪 [Trial] REM suspendido durante ventana limpia.", flush=True)
    
                        env_sig = getattr(self.vision, 'last_activity', 0.1)
                        r_attr = 'neuron_region' if hasattr(self.net, 'neuron_region') else 'region'
                        # --- FIX: vectorizado. Antes era un loop Python puro sobre
                        # las n neuronas (x5 regiones) tomado CON net.lock; con la red
                        # en 15k+ neuronas eso ocupaba el lock el tiempo suficiente
                        # como para trabar los trials de letra (que piden el mismo
                        # lock ~2 veces por frame, 60 frames por trial).
                        indices_arr = np.asarray(getattr(self.net, r_attr))[:self.net.n]
                        fired_arr = self.net.fired[:self.net.n]
    
                        for region in list(self.net.regions.values()):
                            mask = indices_arr == region.id
                            reg_activity = float(np.mean(fired_arr[mask])) if np.any(mask) else 0.0
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
                        # DESPUÉS:
                        lang_diag = self.net.modules.get("lenguaje")
                        if lang_diag:
                            if hasattr(lang_diag, 'debug_pool_status'):
                                lang_diag.debug_pool_status(self.net)
                            if hasattr(lang_diag, 'debug_band_activity'):
                                lang_diag.debug_band_activity(self.net)
                            if hasattr(lang_diag, 'debug_band_geometry'):
                                lang_diag.debug_band_geometry(self.net)
                            if hasattr(lang_diag, 'debug_overlap_evidencia_banda'):
                                lang_diag.debug_overlap_evidencia_banda(self.net)       

                        # --- ARQUITECTURA DE BLOQUES: update del Monitor ---
                        # Sustituye progresivamente los debug_* externos.
                        # Por ahora conviven; cuando el monitor sea estable
                        # se eliminan los debug_* de arriba.
                        monitor = self.net.modules.get("monitor_lenguaje")
                        if monitor:
                            monitor.update(self.net, self.net.current_time)

                        # Corazón: ahora se llama explícitamente desde worker_slow
                        # (en v3.1 su update() existía pero nunca se invocaba)
                        if hasattr(self, 'heart') and self.heart:
                            self.heart.update(self.net, self.net.current_time)

                    except Exception as e:
                        print(f"⚠️ Error en ciclo SlowBio: {e}")
    
                if hasattr(self.vision, '_dopamine_feedback'):
                    self.vision._dopamine_feedback = getattr(self.net, 'dopamine_level', 0.15)
    
                self.step_count += 1
                last_bio_tick = now
    
            time.sleep(max(0.1, self.metabolism.get_sleep_time(stress) if hasattr(self.metabolism, 'get_sleep_time') else 1.0))

    def worker_curriculum(self):
        """Hilo de currícula adaptativa (Etapa 2).

        Con autoplay apagado (default) este hilo no hace nada más que
        dormir: el entrenamiento sigue siendo 100% manual por teclado,
        exactamente como antes. Con autoplay prendido (tecla 'c' en vivo,
        o CURRICULUM_AUTOPLAY_DEFAULT=True en config.py), elige sola qué
        letra entrenar a continuación —dándole más peso a las que peor
        van— y dispara el mismo trigger_letter_training que ya usa el
        modo manual, respetando el watchdog de trials concurrentes."""
        print("🎓 [CurriculumThread] Listo | autoplay inicial: "
              f"{'ON' if self.curriculum_autoplay else 'OFF'} (alternar con 'c')", flush=True)
        while self.is_running:
            if self.curriculum_autoplay and not getattr(self, '_modo_entrenamiento', False):
                ahora = time.time()
                if ahora - self._last_autoplay_trial >= CURRICULUM_AUTOPLAY_COOLDOWN_S:
                    letra = self.curriculum.next_letter()
                    if letra:
                        print(f"🎓 [Currícula-Auto] Entrenando '{letra}' | {self.curriculum.status()}", flush=True)
                        self.trigger_letter_training(letra)
                    self._last_autoplay_trial = ahora

            # --- Etapa 3: autoplay de secuencias, semáforo independiente
            # (tecla 'w'), pero respeta el mismo _modo_entrenamiento que las
            # letras porque comparten la misma red y el mismo scr_processor.
            if getattr(self, 'word_curriculum_autoplay', False) and not getattr(self, '_modo_entrenamiento', False):
                ahora = time.time()
                if ahora - self._last_word_autoplay_trial >= WORD_AUTOPLAY_COOLDOWN_S:
                    palabra = self.word_curriculum.next_letter()
                    if palabra:
                        print(f"🔤 [Secuencia-Auto] Entrenando '{palabra}' | {self.word_curriculum.status()}", flush=True)
                        self.trigger_word_training(palabra)
                    self._last_word_autoplay_trial = ahora

            time.sleep(0.5)

    def run(self, screen_processor=None):
        if screen_processor is not None:
            self.scr_processor = screen_processor
        self.is_running = True
        self._shutdown_in_progress = False
        self._training_stop_event.clear()

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
                    if hasattr(self, 'curriculum'):
                        print(f"🎓 [Currícula] {self.curriculum.status()}")
                    if hasattr(self, 'word_curriculum'):
                        print(f"🔤 [Secuencia] {self.word_curriculum.status()}")
                    last_heartbeat = t_now

                    # NUEVO: Diagnóstico de cola muerta y electroshock
                    with self.net.lock:
                        queue_size = len(self.net.event_queue)
                        if queue_size == 0:
                            # Verificar si el corazón sigue latiendo
                            t_ultimo_pulso = getattr(self.heart, 'last_pulse_time', 0)
                            t_actual = getattr(self.net, 'current_time', 0)
                            silencio_ms = t_actual - t_ultimo_pulso
                            
                            # Imprimir advertencia silenciosa si la cola está en 0
                            if silencio_ms > 1000:  # Mostrar alerta si lleva más de 1 segundo en silencio
                                print(f"⚠️ [Cola Muerta] Queue=0 | Último pulso hace {silencio_ms:.0f}ms | T_red={t_actual:.0f}", flush=True)
                            
                            # Electroshock de emergencia si lleva más de 2 segundos sin actividad (Antes 5000)
                            if silencio_ms > 2000 and not bool(getattr(self.net, "_causal_trial_active", False)):
                                print(f"⚡ [Resucitación] Inyectando electroshock...", flush=True)
                                import heapq  # Aseguramos la importación localmente
                                
                                # Ya estamos dentro del lock, encolamos directamente
                                # --- FIX Bug 2: incluir también neuronas de la banda z=37-53 ---
                                # El electroshock anterior solo tocaba pulse_port[:20] (z≈60),
                                # lejos de la banda de lenguaje. Ahora también inyecta ahí.
                                
                                # Neuronas del corazón (z≈60) — igual que antes
                                heart_targets = list(self.heart.pulse_port.connected_neurons)[:20]
                                
                                # Neuronas de la banda de lenguaje z=37-53
                                pos_z  = self.net.positions[:self.net.n, 2]
                                active = self.net.active[:self.net.n]
                                lang_band_targets = [
                                    i for i in range(self.net.n)
                                    if active[i] and 37.0 <= pos_z[i] <= 53.0
                                ]
                                # Tomar hasta 30 neuronas de la banda (priorizando inmortales si hay monitor)
                                monitor_obj = self.net.modules.get("monitor_lenguaje")
                                if monitor_obj and hasattr(monitor_obj, 'immortal_neurons'):
                                    inmortales = [i for i in monitor_obj.immortal_neurons if i < self.net.n and active[i]]
                                    resto = [i for i in lang_band_targets if i not in set(inmortales)]
                                    import random as _random
                                    lang_band_targets = inmortales + _random.sample(resto, min(22, len(resto)))
                                else:
                                    import random as _random
                                    lang_band_targets = _random.sample(lang_band_targets, min(30, len(lang_band_targets)))
                                
                                all_targets = list(dict.fromkeys(heart_targets + lang_band_targets))
                                print(f"⚡ [Resucitación] Targets: {len(heart_targets)} corazón + {len(lang_band_targets)} banda-lenguaje = {len(all_targets)} total", flush=True)
                                
                                for n_idx in all_targets:
                                    n_idx = int(n_idx)
                                    heapq.heappush(
                                        self.net.event_queue,
                                        (self.net.current_time + 0.001, n_idx, 8.0)
                                    )
                                    # Recargar energía para romper el ciclo vicioso
                                    self.net.energy[n_idx] = max(self.net.energy[n_idx], 0.8)

                # =======================================================
                # === FASE 4: CAPTURA DE TECLADO ===
                # =======================================================
                # Dentro del while:
                if msvcrt is not None and msvcrt.kbhit():
                    raw_tecla = msvcrt.getch()
                    if raw_tecla == b"\x03":
                        print("\n🛑 Ctrl+C detectado. Deteniendo simulación...", flush=True)
                        self.is_running = False
                        self._training_stop_event.set()
                        continue
                    tecla = raw_tecla.decode('utf-8', errors='ignore').lower()
                    letra_tecla = tecla.upper()

                    # --- Etapa 2: cualquier letra del vocabulario activo, no solo X/O ---
                    if letra_tecla in getattr(self.lang_module, 'symbols', []):
                        print(f"\n🎯 [Intervención] Letra {letra_tecla}")
                        self.trigger_letter_training(letra_tecla)
                    elif tecla == 'p':  # resumen/flush de instrumentación causal
                        resumen = self.causal_probe.summarize()
                        print(
                            f"\n🧪 [CausalProbe] trials={resumen['trials_total']} | "
                            f"accuracy={resumen['accuracy']:.2f} | "
                            f"candidatos={len(resumen['causal_candidates'])}"
                            , flush=True
                        )
                        self.causal_probe.flush()
                    elif tecla == 'q':
                        self.is_running = False
                    elif tecla == 'r':  # recablear evidencia a la banda viva
                        print("\n🔌 [Intervención] Recableo manual de evidencia")
                        if hasattr(self.lang_module, 'recable_from_live_band'):
                            self.lang_module.recable_from_live_band(self.net)
                    elif tecla == 'c':  # NUEVO: alternar currícula automática
                        self.curriculum_autoplay = not self.curriculum_autoplay
                        estado = "ON" if self.curriculum_autoplay else "OFF"
                        print(f"\n🎓 [Currícula] Autoplay → {estado}", flush=True)
                    elif tecla == 'v':  # NUEVO: ver estado del vocabulario
                        print(f"\n🎓 [Currícula] {self.curriculum.status()}", flush=True)
                        print(f"🔤 [Secuencia] {self.word_curriculum.status()}", flush=True)
                    elif tecla == 's':  # NUEVO Etapa 3: entrenar la próxima palabra sugerida
                        palabra = self.word_curriculum.next_letter()
                        if palabra:
                            print(f"\n🎯 [Intervención] Palabra '{palabra}'", flush=True)
                            self.trigger_word_training(palabra)
                    elif tecla == 'w':  # NUEVO Etapa 3: alternar autoplay de secuencias
                        self.word_curriculum_autoplay = not self.word_curriculum_autoplay
                        estado = "ON" if self.word_curriculum_autoplay else "OFF"
                        print(f"\n🔤 [Secuencia] Autoplay → {estado}", flush=True)
                    elif tecla in '1234' and int(tecla) <= len(WORD_VOCABULARY):
                        # Disparo directo de una palabra específica por índice,
                        # útil para probar puntualmente el par OX/XO invertido.
                        palabra = WORD_VOCABULARY[int(tecla) - 1]
                        print(f"\n🎯 [Intervención] Palabra fija '{palabra}'", flush=True)
                        self.trigger_word_training(palabra)


        except KeyboardInterrupt:
            print("\n🛑 Interrupción detectada. Deteniendo simulación...")
        except Exception as e:
            print(f"❌ Error en el bucle principal: {e}")
        finally:
            self._shutdown_in_progress = True
            self.is_running = False
            self._training_stop_event.set()
            manager_ok = manager.stop(timeout_s=10.0)
            training_ok = self._stop_training_threads(timeout=10.0)
            if not manager_ok or not training_ok:
                logging.error("❌ Guardado abortado: todavía hay hilos activos sobre la red.")
                return
            if hasattr(self, 'output_module') and self.output_module:
                self.output_module.cerrar_sesion(self.net)
            try:
                self.causal_probe.flush()
            except Exception:
                pass
            print("💾 Guardando estado de la red...", flush=True)
            save_ok = self.save()
            if not save_ok:
                logging.error("❌ El guardado no terminó correctamente.")

    def save(self, filename="simulation_state"):
        """Guarda la red sin modificar topología ni colas runtime."""
        try:
            if getattr(self, 'thread_manager', None) is not None and self.thread_manager.is_alive():
                logging.error("❌ Guardado rechazado: hay hilos de simulación activos.")
                return False
            print("📦 Iniciando secuencia de guardado (Método Híbrido)...")

            # Claves no serializables que siempre excluimos
            CLAVES_EXCLUIDAS = {
                'lock',          # RLock de la red
                'causal_probe', # telemetría runtime; su propia JSON persiste aparte
                'event_queue',   # Heap de eventos en vuelo
                'readout_event_queue',  # Canal de readout: siempre runtime-only
                'potential',          # alias de membrane_potential; no duplicar en el .npz
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
                # Guardar es una operación observacional: no debe modificar la topología.
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

            # Guardamos matrices comprimidas en temporal y hacemos replace atómico.
            final_np = f"{filename}_matrices.npz"
            tmp_np = f"{final_np}.tmp.npz"
            np.savez_compressed(tmp_np, **arrays_np)
            os.replace(tmp_np, final_np)

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
                'version':       "1.7.0-Fase6-ModulePersistence",
                'blocks': {
                    block_id: block.to_dict()
                    for block_id, block in self.net.modules.items()
                    if hasattr(block, 'to_dict') and hasattr(block, 'immortal_neurons')
                }
            }

            final_meta = f"{filename}_meta.pkl"
            tmp_meta = f"{final_meta}.tmp"
            with open(tmp_meta, "wb") as f:
                pickle.dump(data_final, f, protocol=4)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            os.replace(tmp_meta, final_meta)

            print("-" * 30)
            print("💾 PERSISTENCIA HÍBRIDA EXITOSA")
            cables = sum(1 for p in puertos_finales if p['neurons'])
            print(f"-> Archivos: .npz (Cerebro) + .pkl (Genética/Puertos/Módulos)")
            print(f"-> Puertos con cables útiles: {cables}/{len(puertos_finales)}")
            print("-" * 30, flush=True)
            return True

        except Exception as e:
            print(f"❌ Error crítico en save(): {e}", flush=True)
            return False


    
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

            # --- FASE 3: RESCATE DE SUPERVIVENCIA ---
            # Se ejecuta justo después de restaurar arrays y antes de despertar
            # el motor/SlowBio.
            if not hasattr(self.net, 'dopamine_level'):
                self.net.dopamine_level = 0.5
            if getattr(self.net, 'dopamine_level', 0.0) < 0.5:
                # No forzamos a la red a un nivel artificial alto: 0.5 es sólo
                # el nivel basal mínimo de arranque de esta fase.
                self.net.dopamine_level = 0.5

            if hasattr(self.net, '_apply_phase3_survival_guards'):
                self.net._apply_phase3_survival_guards()

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

            # 4.0 Reconstruir máscara física del marcapasos si el guardado viejo
            # no la contenía correctamente.
            if not hasattr(self.net, 'heart_mask') or len(self.net.heart_mask) != self.net.max_neurons:
                self.net.heart_mask = np.zeros(self.net.max_neurons, dtype=bool)
            self.net.heart_mask[:self.net.n] = (
                (self.net.positions[:self.net.n, 2] >= 37.0) &
                (self.net.positions[:self.net.n, 2] <= 53.0)
            )

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
            # FIX 1.5.6: Vaciar la cola guardada completamente al restaurar.
            # Una cola grande persistida es la causa raíz del "Event Queue sigue en 24000".
            # Solo inyectamos el electroshock después, que es lo mínimo para arrancar.
            self.net.event_queue = []
            heapq.heapify(self.net.event_queue)
            self.net.readout_event_queue = []
            heapq.heapify(self.net.readout_event_queue)
            print("🧹 [Restauración] Event Queue + Readout Queue vaciadas para arranque limpio.", flush=True)

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
                    self.net.receive_spike(int(n_idx), 5.0, self.net.current_time + 0.001, origin="electroshock")
            else:
                # Fallback: chispazo genérico si no se encuentra el corazón
                for n_idx in range(20):
                    self.net.receive_spike(int(n_idx), 5.0, self.net.current_time + 0.001, origin="electroshock")
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
                # v1.6.19: reconciliación obligatoria después de restaurar módulos.
                # Evita que un snapshot histórico sustituya la representación cortical.
                lang = self.net.modules.get("lenguaje")
                if lang is not None and hasattr(lang, "auto_connect"):
                    lang.auto_connect(self.net)
                    allowed = set(int(i) for i in getattr(lang, "_readout_sources", []))
                    roles = getattr(self.net, "neuron_roles", {}) or {}
                    if not allowed or not all(roles.get(i) == "cortical_representation" for i in allowed):
                        raise RuntimeError("Readout inválido tras persistencia: fuente fuera de cortical_representation")
                    self.net.readout_allowed_sources = allowed
                    logging.info("🛡️ [Persistencia] Readout reconciliado: %d fuentes corticales.", len(allowed))
            else:
                print("⚠️ [Persistencia] Guardado sin 'modules_state' (versión vieja) — "
                      "los módulos arrancan con el wiring de auto_connect().")

            # =====================================================================
            # 6. CURRÍCULA DE VOCABULARIO (Etapa 2)
            # =====================================================================
            # self.curriculum ya existe (se crea siempre en __init__, con
            # valores por defecto, ANTES de que main.py llame a load_state).
            # network_meta acaba de restaurar net.curriculum_state más arriba
            # (paso 1) si esta corrida ya había entrenado letras antes; acá
            # lo usamos para que self.curriculum "recuerde" ese progreso en
            # vez de arrancar de cero.
            if hasattr(self, 'curriculum'):
                estado_guardado = getattr(self.net, 'curriculum_state', None)
                if estado_guardado:
                    self.curriculum.load_dict(estado_guardado)
                    print(f"🎓 [Persistencia] Currícula restaurada: {self.curriculum.status()}", flush=True)
                else:
                    print(f"🎓 [Persistencia] Sin progreso previo de currícula — "
                          f"arrancando con: {self.curriculum.unlocked}", flush=True)

            # =====================================================================
            # 6.b CURRÍCULA DE SECUENCIAS / PALABRAS (Etapa 3) — mismo patrón
            # =====================================================================
            if hasattr(self, 'word_curriculum'):
                estado_guardado_w = getattr(self.net, 'word_curriculum_state', None)
                if estado_guardado_w:
                    self.word_curriculum.load_dict(estado_guardado_w)
                    print(f"🔤 [Persistencia] Currícula de secuencias restaurada: "
                          f"{self.word_curriculum.status()}", flush=True)
                else:
                    print(f"🔤 [Persistencia] Sin progreso previo de secuencias — "
                          f"arrancando con: {self.word_curriculum.unlocked}", flush=True)

            # --- ARQUITECTURA DE BLOQUES: restaurar desde persistencia ---
            # --- ARQUITECTURA DE BLOQUES: restaurar desde persistencia ---
            if "blocks" in data:
                restored = restore_blocks_from_state(data["blocks"], self.net)
                for block_id, block in restored.items():
                    setattr(self, block_id.replace("-", "_"), block)
                logging.info(f"🧱 [Bloques] {len(restored)} bloques restaurados.")

                if "transductor_vision" in restored and hasattr(self, 'vision') and self.vision:
                    self.transductor = restored["transductor_vision"]
                    self.vision._transductor = self.transductor
                    # Corregir zone_z si el pkl guardado tenía el valor viejo (20,50)
                    if self.transductor.zone_z == (20.0, 50.0):
                        self.transductor.zone_z = (10.0, 25.0)
                        print("🔧 [Transductor] zone_z corregida de (20,50) -> (10,25) al restaurar.", flush=True)
                    logging.info("🔌 [Transductor] Reconectado a VisionModule tras restaurar.")

                if "detector_bordes_v" in restored and hasattr(self, 'vision') and self.vision:
                    self.edge_detector_v = restored["detector_bordes_v"]
                    self.vision._edge_detector_v = self.edge_detector_v
                    logging.info("🔍 [BordesV] Reconectado a VisionModule tras restaurar.")

                if "detector_bordes_h" in restored and hasattr(self, 'vision') and self.vision:
                    self.edge_detector_h = restored["detector_bordes_h"]
                    self.vision._edge_detector_h = self.edge_detector_h
                    logging.info("🔍 [BordesH] Reconectado a VisionModule tras restaurar.")

                if "detector_diagonales" in restored and hasattr(self, 'vision') and self.vision:
                    self.diag_detector = restored["detector_diagonales"]
                    self.vision._diag_detector = self.diag_detector
                    logging.info("╲╱ [Diag] Reconectado a VisionModule tras restaurar.")

                if "combinador_esquina" in restored and hasattr(self, 'vision') and self.vision:
                    self.combinador_esquina = restored["combinador_esquina"]
                    self.vision._combinador_esquina = self.combinador_esquina
                    logging.info("🔲 [Esquinas] Reconectado a VisionModule tras restaurar.")

                if "detector_curvas" in restored and hasattr(self, 'vision') and self.vision:
                    self.curva_detector = restored["detector_curvas"]
                    self.vision._curva_detector = self.curva_detector
                    logging.info("〇 [Curvas] Reconectado a VisionModule tras restaurar.")

                if "detector_simetria" in restored and hasattr(self, 'vision') and self.vision:
                    self.simetria_detector = restored["detector_simetria"]
                    self.vision._simetria_detector = self.simetria_detector
                    logging.info("🪞 [Simetría] Reconectado a VisionModule tras restaurar.")

                if "combinador_junction" in restored and hasattr(self, 'vision') and self.vision:
                    self.junction_detector = restored["combinador_junction"]
                    self.vision._junction_detector = self.junction_detector
                    logging.info("✚ [Junctions] Reconectado a VisionModule tras restaurar.")
                    
            # --- NUEVO: Recuperar y Mostrar el Fitness Histórico ---
            historical_fitness = getattr(self.net, 'fitness_score', 0.0)    

            # v1.2: el readout ya no se restaura como exemption global del governor.
            # Solo retiramos posibles exemptions heredadas de checkpoints antiguos.
            if not hasattr(self.net, 'governor_exempt_targets'):
                self.net.governor_exempt_targets = set()
            lang_mod = self.net.modules.get("lenguaje")
            if lang_mod is not None:
                srcs = getattr(lang_mod, '_readout_sources', [])
                pools = getattr(lang_mod, '_readout_pools', {})
                for idx in srcs:
                    self.net.governor_exempt_targets.discard(int(idx))
                for sym_pool in pools.values():
                    for idx in sym_pool:
                        self.net.governor_exempt_targets.discard(int(idx))

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