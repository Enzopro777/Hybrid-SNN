# =============================
# === config.py - v1.3.0 (Fase 4: Persistencia) ===
# =============================

# --- PARÁMETROS BASE ---
NUM_NEURONS = 10000
TIME_STEPS = 200
GRID_SIZE = 50
RADIUS = 3
NUM_REGIONS = 5
NUM_NEUROTRANSMITTERS = 10
LEARNING_RATE = 0.05

# --- CAPACIDAD MÁXIMA DE LA RED ---
# Antes esto vivía hardcodeado como self.max_neurons=8000 dentro de
# NeuralNetwork.__init__ (core/network.py), sin relación con el NUM_NEURONS
# de arriba (que de hecho no lo usa nadie en el código, es un resto viejo).
# Este SÍ está conectado: se pasa a NeuralNetwork(max_neurons=MAX_NEURONS) y
# ahí determina el tamaño de TODOS los arrays pre-alocados (posiciones,
# energía, conexiones, etc.) y el techo de GrowthSystem — más allá de este
# número, add_neuron() deja de agregar neuronas nuevas, silenciosamente.
MAX_NEURONS = 10000

# --- READOUT SNN V1.6.17: representación cortical + readout competitivo ---
# Los eventos del FeatureBus llegan con strengths ~2.8..6.8. El pool debe
# integrar y discriminar, no disparar ante cada evento.
READOUT_SYNAPTIC_GAIN = 1.10
READOUT_ROUTING_TEMPERATURE = 0.35
READOUT_ROUTING_MIN_BUDGET = 2.80
READOUT_ROUTING_MAX_BUDGET = 4.00
READOUT_POOL_VTHRESH = 1.55
READOUT_POOL_TAU_M = 55.0
READOUT_LEARNING_RATE = 0.045
READOUT_COMPETITOR_RATE = 0.025
READOUT_ROW_CENTERING = 0.06
READOUT_BALANCE_MIN = 0.75
READOUT_BALANCE_MAX = 1.50
READOUT_SCORE_TEMPERATURE = 0.65
CORTICAL_REPRESENTATION_NEURONS = 96
CORTICAL_REPRESENTATION_TOP_K = 8
CORTICAL_REPRESENTATION_TAU_MS = 55.0
CORTICAL_REPRESENTATION_FATIGUE = 0.65
# 1.6.21: codificación latente dispersa sobre el vector completo FeatureBus.
CORTICAL_REPRESENTATION_SPARSE_INPUTS = 36
CORTICAL_REPRESENTATION_TEMPORAL_BLEND = 0.78
CORTICAL_REPRESENTATION_SCORE_THRESHOLD = 0.50
CORTICAL_REPRESENTATION_REPEAT_PENALTY = 0.06
# Competencia real entre unidades corticales: evita que 1-3 fuentes
# reaparezcan en casi todos los frames. No codifica clases.
CORTICAL_INTRA_TRIAL_INHIBITION = 1.15
CORTICAL_WINNER_COOLDOWN_MS = 650.0
CORTICAL_MIN_WINNERS = 0  # 1.6.16: no hard minimum; top-k is a maximum, not a quota.
# Proyección cortical -> readout dispersa: cada fuente participa en 3 de 5
# pools, elegidos de forma determinista y balanceada, en vez de broadcast 5/5.
READOUT_POOLS_PER_SOURCE = 5
# 1.6.21: una sinapsis estable por fuente y pool; el score consume todas las
# conexiones visibles del par fuente/pool, evitando la lectura de sólo la primera.
READOUT_TARGETS_PER_POOL_PER_SOURCE = 1

# --- READOUT SNN V1.6.11: separación funcional de pools ---
# Inhibición lateral entre pools: cuando el pool ganador dispara, suprime
# parcialmente los demás. Esto fuerza diferenciación activa de representaciones.
READOUT_LATERAL_INHIBITION = 0.18    # competencia suave entre pools; solo actúa con dominancia clara
READOUT_LATERAL_DECAY = 0.90         # decaimiento por step del estado inhibitorio
# Mínimo de neuronas ÚNICAS por pool para considerar el trial informativo
READOUT_MIN_UNIQUE_POOL_NEURONS = 3
# Umbral de margen para actualizar el conteo de separación funcional
READOUT_SEPARATION_MARGIN_THRESHOLD = 0.15
# v1.6.13: presión one-vs-rest. El target se refuerza y los otros pools
# reciben una depresión pequeña y simétrica para evitar que queden todos en w0.
READOUT_ALL_RIVAL_DEPRESSION = 0.012
# 1.6.21: sólo la fracción más activa de fuentes recibe potenciación fuerte;
# evita reforzar casi toda la población cortical en cada ensayo.
READOUT_LEARNING_SOURCE_FRACTION = 1.0
# 1.6.22: separador cortical agnóstico. Usa geometría multiescala y una segunda
# proyección no lineal para reducir colisiones entre patrones similares sin
# introducir etiquetas de clase.
CORTICAL_SEPARATOR_BINS_X = 6
CORTICAL_SEPARATOR_BINS_Y = 6
CORTICAL_SEPARATOR_NONLINEAR_MIX = 0.40
CORTICAL_SEPARATOR_WHITEN_ALPHA = 0.035
CORTICAL_SEPARATOR_TEMPERATURE = 0.85
CORTICAL_SEPARATOR_SEED = 162200
# Reloj causal: durante LetterTraining el trial avanza la simulación a cadencia fija.
CAUSAL_TRIAL_FRAME_DT_MS = 1000.0 / 30.0
# 1.6.16: competencia física continua entre pools y homeostasis intrapool.
# La inhibición ya no espera a que un pool alcance >45% del tráfico.
READOUT_INTERPOOL_INHIBITION_STEP = 0.22
READOUT_INTRAPOOL_FATIGUE_STEP = 0.18
READOUT_INTRAPOOL_FATIGUE_DECAY_MS = 320.0
READOUT_INTERPOOL_INHIBITION_DECAY = 0.90

# --- IA 1.6.23: temporal core, sparse stable/fringe cortex y presupuesto causal ---
CAUSAL_TRIAL_FANOUT_CAP = 4
CAUSAL_READY_COALESCE_THRESHOLD = 7000
CAUSAL_READY_COALESCE_TARGET = 4500
CAUSAL_EVENT_ATTEMPT_BUDGET_PER_FRAME = 1400
CORTICAL_PROJECTION_NONZERO = 28
CORTICAL_SEPARATOR_NONLINEAR_MIX = 0.03
CORTICAL_SEPARATOR_WHITEN_ALPHA = 0.012
CORTICAL_SEPARATOR_TEMPERATURE = 0.95
CORTICAL_CORE_FRACTION = 0.18
CORTICAL_CORE_BONUS = 0.16
CORTICAL_USAGE_PENALTY = 0.20
CORTICAL_FRINGE_NOVELTY_GAIN = 0.08
CORTICAL_DIVERSITY_PENALTY = 0.04
CORTICAL_HOMEOSTASIS_TARGET = 0.070
CORTICAL_HOMEOSTASIS_RATE = 0.045
CORTICAL_HOMEOSTASIS_MAX_DELTA = 0.45
CORTICAL_PATTERN_MEMORY_SIZE = 12
CORTICAL_STABLE_SIMILARITY = 0.90

# --- IA 1.6.24: stable-core anchoring + activity floor ---
CORTICAL_CORE_SIZE = 6
CORTICAL_CORE_ANCHOR_BLEND = 0.72
CORTICAL_CORE_ANCHOR_BONUS = 0.14
CORTICAL_PERSISTENT_ANCHOR_BLEND = 0.18
CORTICAL_PERSISTENT_ANCHOR_SIMILARITY = 0.96
CORTICAL_MIN_EMIT = 6
CORTICAL_ACTIVITY_FLOOR = 0.18
CORTICAL_USAGE_SOFT_CAP = 0.35
CORTICAL_HOMEOSTASIS_MIN_RATE = 0.015
CORTICAL_HOMEOSTASIS_MAX_RATE = 0.16
CAUSAL_TRIAL_FANOUT_CAP = 4
CAUSAL_EVENT_ATTEMPT_BUDGET_PER_FRAME = 1400
CLEAN_EXPERIMENT_DEFAULT = True
CLEAN_EXPERIMENT_REMOVE_CALIBRATION = True

# --- IA 1.6.25: readout temporal + plasticity causal + identity prototypes ---
READOUT_TEMPORAL_BINS = 12
READOUT_TEMPORAL_DECAY = 0.88
READOUT_TEMPORAL_WINDOW_MS = 400.0
READOUT_ELIGIBILITY_TRACE_DECAY_MS = 220.0
READOUT_ELIGIBILITY_WINDOW_MS = 250.0
READOUT_ELIGIBILITY_POST_GAIN = 1.0
READOUT_SYNAPTIC_SCALING_INTERVAL = 8
READOUT_SYNAPTIC_SCALING_RATE = 0.025
READOUT_SYNAPTIC_NORM_TARGET = 2.347871

# Identity latente agnóstica: la estabilidad ya no depende del último trial
# ni de un core global compartido por todas las clases.
CORTICAL_IDENTITY_MAX_PROTOTYPES = 12
CORTICAL_IDENTITY_SIMILARITY = 0.93
CORTICAL_IDENTITY_UPDATE = 0.12
CORTICAL_IDENTITY_CORE_BONUS = 0.10
CORTICAL_IDENTITY_NOVELTY_GATE = 0.06

# Scheduler causal: presupuesto global por frame + utilidad física + compresión
# segura de eventos sin readout. El hard-cap histórico no gobierna el trial.
CAUSAL_EVENT_UTILITY_MIN = 0.12
CAUSAL_EVENT_PREFILTER_RATIO = 0.035
CAUSAL_EVENT_RECENCY_TAU_MS = 90.0
CAUSAL_EVENT_AGGREGATION_ENABLED = True
CAUSAL_EVENT_AGGREGATION_BUCKET_MS = 0.10
CAUSAL_EVENT_AGGREGATION_MAX_STRENGTH = 12.0
CAUSAL_TRIAL_FANOUT_CAP = 8
CAUSAL_EVENT_ATTEMPT_BUDGET_PER_FRAME = 1400



# --- IA 1.6.29: workspace latente v4 ---
# H0 se convierte en baseline operacional; el recurrente sólo recibe crédito
# cuando una etapa posterior demuestra una mejora real de loss sobre H0.
LATENT_WORKSPACE_VERSION = "1.6.29f-latent-workspace-v7"
LATENT_WORKSPACE_ENABLED = True
LATENT_WORKSPACE_AUTHORITATIVE = False
LATENT_WORKSPACE_STEPS = 4
LATENT_WORKSPACE_HIDDEN_DIM = 96
LATENT_WORKSPACE_TOP_K = 24
LATENT_WORKSPACE_RECURRENT_DENSITY = 0.08
LATENT_WORKSPACE_RECURRENT_GAIN = 0.62
LATENT_WORKSPACE_CONTEXT_GAIN = 0.35
LATENT_WORKSPACE_STATE_MIX = 0.50
LATENT_WORKSPACE_CONTEXT_UPDATE_RATE = 0.65
LATENT_WORKSPACE_SEED = 162900
LATENT_WORKSPACE_DECODER_LR = 0.08
LATENT_WORKSPACE_DECODER_MAX_UPDATE_NORM = 0.45
LATENT_WORKSPACE_RECURRENT_LR = 0.004
LATENT_WORKSPACE_RECURRENT_MAX_UPDATE_NORM = 0.08
LATENT_WORKSPACE_CONTEXT_LR = 0.006
LATENT_WORKSPACE_CONTEXT_MAX_UPDATE_NORM = 0.08
LATENT_WORKSPACE_MIN_STEP_MIX = 0.20
LATENT_WORKSPACE_HALT_DELTA = 0.015
LATENT_WORKSPACE_HALT_MARGIN_GAIN = 0.0015
LATENT_WORKSPACE_HALT_INNOVATION = 0.005
LATENT_WORKSPACE_HALT_PATIENCE = 1
LATENT_WORKSPACE_WEIGHT_CLIP = 0.40
LATENT_WORKSPACE_STAGE_MIN_LOSS_GAIN = 0.005
LATENT_WORKSPACE_EARLY_EXIT_ENABLED = True
LATENT_WORKSPACE_EARLY_EXIT_MIN_MARGIN_GAIN = 0.0015
LATENT_WORKSPACE_EARLY_EXIT_PATIENCE = 1
LATENT_WORKSPACE_DECODER_SEED_FROM_READOUT = False
LATENT_WORKSPACE_NEUTRAL_DECODER_SCALE = 0.75
LATENT_WORKSPACE_PROTOTYPE_PROBE_ENABLED = True
LATENT_WORKSPACE_PROTOTYPE_UPDATE_RATE = 0.08
LATENT_WORKSPACE_REPLAY_CAPACITY = 5
LATENT_WORKSPACE_REPLAY_PER_CLASS = 1
LATENT_WORKSPACE_REPLAY_WEIGHT = 0.15
LATENT_WORKSPACE_TEMPORAL_SHADOW_ENABLED = True
LATENT_WORKSPACE_TEMPORAL_SHADOW_LR = 0.02
LATENT_WORKSPACE_TEMPORAL_SHADOW_MAX_UPDATE_NORM = 0.25
LATENT_WORKSPACE_TEMPORAL_AUDIT_ENABLED = True
LATENT_WORKSPACE_TELEMETRY_HISTORY = 16
LATENT_WORKSPACE_TRAINING_ENABLED = True
LATENT_WORKSPACE_EVAL_EVERY = 5
LATENT_WORKSPACE_EVAL_BALANCE_STRICT = True
LATENT_WORKSPACE_EVAL_SYMBOLS = ("X", "O", "T", "A", "E")
LATENT_WORKSPACE_PAIRED_ABLATION = True
LATENT_WORKSPACE_QUEUE_AUDIT_ENABLED = True
LATENT_WORKSPACE_SPECIALIZATION_EXPERIMENT = False

# --- IA 1.6.29e: bridge de evidencia + consolidación tipo sueño + auditoría de flujo ---
# El sueño ocurre fuera del Event Queue y sólo después de trials completos.
# NREM-like es la fase activa por defecto; REM-like queda experimentalmente apagada.
LATENT_WORKSPACE_SLEEP_ENABLED = True
LATENT_WORKSPACE_SLEEP_INTERVAL = 25
LATENT_WORKSPACE_SLEEP_NREM_PASSES = 2
LATENT_WORKSPACE_SLEEP_REPLAY_PER_CLASS = 2
LATENT_WORKSPACE_SLEEP_REPLAY_WEIGHT = 0.08
LATENT_WORKSPACE_SLEEP_HOMEOSTASIS_RATE = 0.015
LATENT_WORKSPACE_SLEEP_HOMEOSTASIS_TARGET_L1 = 0.62
LATENT_WORKSPACE_SLEEP_REM_LIKE_ENABLED = False
LATENT_WORKSPACE_SLEEP_REM_LIKE_PASSES = 1
LATENT_WORKSPACE_SLEEP_REM_LIKE_RATE = 0.002
LATENT_WORKSPACE_INFORMATION_FLOW_AUDIT = True
LATENT_WORKSPACE_INFORMATION_LOSS_CLIP = 1.0

# Persistencia ligera de frontera de trial. Registra atómicamente el último
# trial comprometido sin intentar escribir la red completa mientras los hilos
# biológicos siguen activos. El guardado completo continúa siendo el cierre
# normal de SimulationEngine.save().


# --- IA 1.6.29f: readout stability + bridge credit control ---
# La decisión física sigue siendo neuronal. La evidencia usa sólo estado
# instantáneo/temporal del trial; drive_ema queda como telemetría histórica.
READOUT_EVIDENCE_SPIKE_WEIGHT = 0.50
READOUT_EVIDENCE_SURPRISE_WEIGHT = 0.30
READOUT_EVIDENCE_PRESENCE_WEIGHT = 0.20
READOUT_HISTORY_BIAS_RATE = 0.0
READOUT_HISTORY_BIAS_CLIP = 0.12
READOUT_BRIDGE_LATENT_GUIDE_RATE = 0.35
READOUT_BRIDGE_LATENT_GAIN_FLOOR = 0.005
READOUT_BRIDGE_MEMORY_CAPACITY = 40
# En 29f el replay físico se desactiva por defecto para aislar el experimento.
# El replay NREM del workspace latente permanece activo.
READOUT_BRIDGE_SLEEP_REPLAY_ENABLED = False
READOUT_BRIDGE_SLEEP_REPLAY_PER_CLASS = 2
READOUT_BRIDGE_SLEEP_REPLAY_WEIGHT = 0.04
READOUT_BRIDGE_SLEEP_REPLAY_PASSES = 1

# --- 1.6.29f: homeostasis unilateral contra hiperactividad ---
READOUT_29F_HOMEOSTASIS_ENABLED = True
READOUT_29F_HOMEOSTASIS_WARMUP_TRIALS = 8
READOUT_29F_HOMEOSTASIS_RATE_EMA_DECAY = 0.88
READOUT_29F_HOMEOSTASIS_REFERENCE_QUANTILE = 0.50
READOUT_29F_HOMEOSTASIS_OVERACTIVITY_RATIO = 1.35
READOUT_29F_HOMEOSTASIS_MIN_RATE = 0.25
READOUT_29F_HOMEOSTASIS_OFFSET_RATE = 0.0125
READOUT_29F_HOMEOSTASIS_MAX_OFFSET = 0.20
READOUT_29F_HOMEOSTASIS_MIN_THRESHOLD = READOUT_POOL_VTHRESH
READOUT_29F_HOMEOSTASIS_MAX_THRESHOLD = 1.79
READOUT_29F_HOMEOSTASIS_EPS = 1e-6
READOUT_29F_HOMEOSTASIS_STATE_VERSION = "1.6.29f-homeostasis-v2"

# Aprendizaje por margen: el target deja de recibir LTP por el simple hecho
# de no haber llegado todavía a score=1. Sólo se corrige un margen insuficiente.
READOUT_29F_DESIRED_TARGET_MARGIN = 0.08
READOUT_29F_TARGET_ERROR_FLOOR = 0.02
READOUT_29F_RIVAL_MARGIN_GUARD = 0.05
READOUT_29F_MAX_DELTA = 0.12

# La normalización simétrica 1.6.27 se conserva disponible para auditorías
# históricas, pero queda fuera del circuito de entrenamiento de 29f.
READOUT_29F_DISABLE_COLUMN_NORMALIZATION = True
READOUT_29F_DISABLE_SYNAPTIC_SCALING = True

TRIAL_COMMIT_JOURNAL_ENABLED = True
TRIAL_COMMIT_JOURNAL_PATH = "trial_commit_1.6.29f.json"

# Observabilidad del trial: la ventana temporal del source readout usa siempre
# expected_frames, nunca el número de frames observados hasta ese instante.
READOUT_TRIAL_EXPECTED_FRAME_DT_MS = 1000.0 / 30.0

# --- IA 1.6.29c: telemetría de visión/fóvea (observación; sin reward) ---
# La trayectoria visual se registra por frame durante LetterTraining para
# poder estudiar si la mirada aporta información útil. Estos parámetros NO
# cambian la dinámica de atención ni aplican recompensa por posición.
CAUSAL_VISION_TELEMETRY_ENABLED = True
CAUSAL_VISION_TRACE_STRIDE = 1
CAUSAL_VISION_TARGET_RADIUS = 0.10
CAUSAL_VISION_SACCADE_STEP = 0.04
CAUSAL_VISION_FIXATION_STEP = 0.01
CAUSAL_VISION_MIN_FIXATION_FRAMES = 3

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

# La entrada sensorial usa rutas estructurales fijas sobre sinapsis normales.
# No se usa radio de inyección por frame.

# --- FASE 4: DARWINISMO Y COMPETENCIA ---
ENERGY_DECAY_IDLE = 0.005
NEURON_DEATH_THRESHOLD = 0.20
REGION_INHIBITION_STRENGTH = 0.9
REGION_DOMINANCE_THRESHOLD = 0.65
IGNITION_THRESHOLD = 30
IGNITION_BOOST_FACTOR = 4.0
ENERGY_RECOVERY_SPIKE = 0.02
MAX_NEURON_ENERGY = 2.0





# --- ETAPA 2: VOCABULARIO Y CURRÍCULA ADAPTATIVA (v1.6.14) ---
# Vocabulario objetivo completo. X y O van primero porque ya están
# validadas por la Fase A (Secuencias); el resto se va desbloqueando solo
# a medida que la red domina las letras activas. Elegidas por ser
# visualmente distintas entre sí (menos ambigüedad en el trazo que letras
# similares como E/F o C/O).
# IMPORTANTE: cada letra debe ser 1 solo carácter, porque además de
# identificar el símbolo, ese carácter es la tecla que lo dispara a mano
# (evitar que coincida con 'q', 'r', 'c' o 'v', que ya tienen otro uso).
LETTER_VOCABULARY = ["X", "O", "T", "A", "E"]
CURRICULUM_STARTER_LETTERS = ["X", "O", "T", "A", "E"]
CURRICULUM_BALANCED_MODE = True
CURRICULUM_MAX_TRIAL_IMBALANCE = 1
CURRICULUM_WINDOW = 8                    # trials recientes que se promedian por letra
CURRICULUM_MASTERY_EVIDENCE = 0.40        # evidencia_real promedio para considerar dominada una letra
CURRICULUM_MASTERY_SUCCESS_RATE = 0.7    # % de trials correctamente clasificados para considerarla dominada
CURRICULUM_MASTERY_MARGIN = 0.10           # separación media mínima top-vs-second del learned readout
CURRICULUM_MIN_TRIALS_TO_UNLOCK = 6      # mínimo de trials por letra activa antes de evaluar el próximo desbloqueo
CURRICULUM_WEAK_LETTER_BOOST = 3.0       # cuánto más se practican las letras flojas frente a las dominadas
CURRICULUM_AUTOPLAY_DEFAULT = False      # arranca en manual (tecla por trial); alternar en vivo con 'c'
CURRICULUM_AUTOPLAY_COOLDOWN_S = 4.0     # segundos de pausa entre trials automáticos

# --- ETAPA 3: SECUENCIAS (letras en el tiempo -> ¿palabras?) ---
# Reutiliza el mismo motor de currícula que Etapa 2 (VocabularyCurriculum
# no sabe ni le importa si sus "símbolos" son letras o bigramas), pero
# aplicado a pares ORDENADOS de letras ya dominadas. La evidencia no sale
# de una neurona nueva: sale del log de flancos que check_edge_events ya
# viene generando (ver LetterIOModule._event_log / update_sequence_evidence).
# Empezamos por el par OX/XO a propósito: son las mismas dos letras en
# orden invertido, así que si la red los distingue es porque de verdad
# está usando el ORDEN y no solo "vi X y vi O en algún momento".
WORD_VOCABULARY = ["OX", "XO", "TA", "AT"]
WORD_STARTER_WORDS = ["OX", "XO"]
WORD_TRIAL_FRAMES_PER_LETTER = 40   # ~1.3s por letra a 30fps
WORD_TRIAL_GAP_FRAMES = 8           # ~0.27s de pantalla en blanco entre letra 1 y letra 2
WORD_MAX_GAP_MS = 1500.0            # si letra2 llega más tarde que esto tras letra1, no cuenta como secuencia
WORD_CURRICULUM_WINDOW = 6
WORD_MASTERY_EVIDENCE = 0.4
WORD_MASTERY_SUCCESS_RATE = 0.6
WORD_MIN_TRIALS_TO_UNLOCK = 6
WORD_WEAK_LETTER_BOOST = 3.0
WORD_CURRICULUM_AUTOPLAY_DEFAULT = False   # manual al inicio; alternar en vivo con 'w'
WORD_AUTOPLAY_COOLDOWN_S = 5.0

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

# --- IA 1.6.26: pool calibration + temporal competition + subthreshold bootstrap ---
# Normalización lenta por pool: corrige sesgos de excitabilidad sin cambiar
# la señal cortical ni codificar clases. La baseline se actualiza sólo al cierre
# de trials completos y se usa para normalizar la evidencia temporal real.
READOUT_POOL_BASELINE_DECAY = 0.90
READOUT_POOL_BASELINE_VAR_DECAY = 0.90
READOUT_POOL_BASELINE_FLOOR = 0.50
READOUT_POOL_Z_CLIP = 2.50
READOUT_POOL_SCORE_RAW_MIX = 0.18

# Competición física retardada: la inhibición sólo comienza cuando existe una
# dominancia medible, evitando que el primer spike sesgue toda la ventana.
READOUT_COMPETITION_MIN_SPIKES = 3
READOUT_COMPETITION_MIN_MARGIN = 0.12
READOUT_COMPETITION_MIN_INTERVAL_MS = 45.0

# Bootstrap causal subumbral: permite que una clase débil aprenda sin inyectar
# spikes artificiales. Se acumula proximidad a threshold + traza pre, y se
# convierte en aprendizaje sólo al cerrar el trial con su target real.
READOUT_SUBTHRESHOLD_GATE = 0.60
READOUT_SUBTHRESHOLD_GAIN = 0.20
READOUT_SUBTHRESHOLD_MAX = 3.00
READOUT_SUBTHRESHOLD_MIN_TARGET_SPIKES = 2


# --- IA 1.6.27: crédito sináptico local + homeostasis del readout + decoder v2 ---
# No se descarta el 65% de las fuentes corticales activas: toda fuente cortical
# que haya participado puede aportar crédito, pero sólo mediante una elegibilidad
# específica por sinapsis (pre->post real o aproximación subumbral real).
READOUT_1627_USE_ALL_ACTIVE_SOURCES = True
READOUT_1627_CREDIT_RATE = 0.014
READOUT_1627_CREDIT_RIVAL_RATE = 0.008
READOUT_1627_CREDIT_MAX_ELIGIBILITY = 10.0
READOUT_1627_CREDIT_MIN_ELIGIBILITY = 0.005
READOUT_1627_CREDIT_ERROR_FLOOR = 0.02
READOUT_1627_SUBTHRESHOLD_CREDIT_SCALE = 0.22
READOUT_1627_SOURCE_COUNT_SCALE_MIN = 0.65
READOUT_1627_SOURCE_COUNT_SCALE_MAX = 1.35
# Decoder v2: evidencia = exceso positivo sobre baseline + presencia observada.
# El estado neutro (arr==baseline) ya no se transforma en 0.5 artificial.
READOUT_1627_SURPRISE_WEIGHT = 0.82
READOUT_1627_PRESENCE_WEIGHT = 0.18
READOUT_1627_SURPRISE_TEMPERATURE = 1.0
READOUT_1627_SCORE_EPS = 1e-6
# Homeostasis del readout: muy lenta, agnóstica a la identidad de la clase.
# Target 0.50 spikes/frame/pool ≈ 30 spikes/pool por trial de 60 frames.
READOUT_1627_HOMEOSTASIS_TARGET_RATE = 0.50
READOUT_1627_HOMEOSTASIS_EMA_DECAY = 0.92
READOUT_1627_HOMEOSTASIS_THRESHOLD_RATE = 0.020
READOUT_1627_HOMEOSTASIS_MAX_OFFSET = 0.24
READOUT_1627_HOMEOSTASIS_MIN_THRESHOLD = 1.31
READOUT_1627_HOMEOSTASIS_MAX_THRESHOLD = 1.79
# Normalización de columna: iguala capacidad global sin borrar selectividad local.
READOUT_1627_COLUMN_NORM_INTERVAL = 8
READOUT_1627_COLUMN_NORM_RATE = 0.025
READOUT_1627_COLUMN_NORM_MIN_FACTOR = 0.975
READOUT_1627_COLUMN_NORM_MAX_FACTOR = 1.025
READOUT_1627_COLUMN_NORM_EPS = 1e-8
# Auditoría sombra: mide cuánto de la información queda en cortex y pools.
READOUT_1627_SHADOW_BINS = 3
READOUT_1627_SHADOW_MIN_SOURCE_RATE = 0.0
TEMPORAL_CORE_VERSION = "1.6.27"
