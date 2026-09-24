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