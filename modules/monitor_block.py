# modules/monitor_block.py
#
# =====================================================================
# BLOQUE MONITOR — Arquitectura de Bloques v0.1
# =====================================================================
#
# Qué es:
#   El primer bloque de la nueva arquitectura. Se inyecta en una zona
#   específica de la red (por defecto la banda z=37-53 donde vive el
#   lenguaje) y reemplaza el diagnóstico externo que hoy hace simulation.py.
#
# Qué hace:
#   - Mantiene un núcleo de neuronas INMORTALES que no pueden morir
#   - Monitoriza la salud de su zona (spike-rate, energía promedio)
#   - Si la actividad cae bajo el umbral, inyecta spikes locales
#     (electroshock quirúrgico, no global)
#   - Emite logs de diagnóstico propios cada N ciclos
#   - Expone métricas de salud para que otras IAs o el sistema puedan leerlas
#
# Cómo se usa:
#   monitor = MonitorBlock(zone_z=(37.0, 53.0), n_immortal=8)
#   monitor.inject(net)          # una sola vez al iniciar o restaurar
#   monitor.update(net, t)       # llamar cada ciclo de SlowBioThread
#
# Para IAs colaboradoras:
#   Este bloque es intencionalmente simple. Es la base del patrón
#   que se va a repetir para detectores de bordes, combinadores, etc.
#   Cada bloque futuro hereda de BaseBlock (al final de este archivo).
# =====================================================================

import numpy as np
import time
import logging
from typing import Optional, List, Tuple


# -----------------------------------------------------------------------
# BASE — patrón que heredan todos los bloques futuros
# -----------------------------------------------------------------------

class BaseBlock:
    """
    Clase base para todos los bloques de la arquitectura.
    
    Un bloque es una región funcional con:
    - Coordenadas fijas en el espacio 3D de la red
    - Núcleo de neuronas inmortales que no pueden morir
    - Función de pérdida local (cada bloque sabe qué tiene que hacer)
    - Capacidad de autogestionar su actividad
    - Log de diagnóstico propio
    """

    def __init__(
        self,
        block_id: str,
        zone_z: Tuple[float, float],
        zone_x: Tuple[float, float] = (0.0, 150.0),
        zone_y: Tuple[float, float] = (0.0, 150.0),
        n_immortal: int = 8,
        log_every: int = 50,        # cada cuántos update() se imprime el log
    ):
        self.block_id   = block_id
        self.zone_z     = zone_z    # (z_min, z_max)
        self.zone_x     = zone_x
        self.zone_y     = zone_y
        self.n_immortal = n_immortal
        self.log_every  = log_every

        # Neuronas del núcleo inmortal (índices en net)
        self.immortal_neurons: List[int] = []

        # Métricas de salud (actualizadas en cada update)
        self.health = {
            "spike_rate":    0.0,   # spikes/ciclo promedio en la zona
            "energy_mean":   0.0,   # energía promedio de las neuronas de la zona
            "energy_min":    0.0,   # energía mínima
            "n_fired":       0,     # cuántas neuronas dispararon en el último ciclo
            "n_zone":        0,     # total de neuronas en la zona
            "electroshocks": 0,     # cuántos electroshocks se inyectaron
            "last_update":   0.0,   # timestamp del último update
        }

        self._update_count = 0
        self._injected     = False

    # ------------------------------------------------------------------
    # INYECCIÓN — llamar UNA sola vez al iniciar o restaurar
    # ------------------------------------------------------------------

    def inject(self, net) -> bool:
        """
        Inyecta el bloque en la red:
        1. Identifica las neuronas activas dentro de la zona
        2. Marca las N más activas como inmortales
        3. Registra el bloque en net.modules

        Retorna True si la inyección fue exitosa.
        """
        with net.lock:
            n_actual = net.n
            pos      = net.positions[:n_actual].copy()
            active   = net.active[:n_actual].copy()
            spikes   = net.last_spike[:n_actual].copy()

        z_min, z_max = self.zone_z
        x_min, x_max = self.zone_x
        y_min, y_max = self.zone_y

        # Neuronas dentro de la zona
        en_zona = np.where(
            active &
            (pos[:, 2] >= z_min) & (pos[:, 2] <= z_max) &
            (pos[:, 0] >= x_min) & (pos[:, 0] <= x_max) &
            (pos[:, 1] >= y_min) & (pos[:, 1] <= y_max)
        )[0]

        if len(en_zona) == 0:
            print(f"⚠️ [{self.block_id}] Zona vacía al inyectar — reintentá más tarde.", flush=True)
            return False

        # Seleccionar las N más activas (por last_spike más reciente)
        # Si hay menos que n_immortal, tomamos todas
        n_sel = min(self.n_immortal, len(en_zona))
        orden = np.argsort(spikes[en_zona])[::-1]   # más recientes primero
        self.immortal_neurons = [int(en_zona[orden[i]]) for i in range(n_sel)]
        # v1.6.19: las inmortales son infraestructura de observación.
        if hasattr(net, "register_noncompetitive_neurons"):
            net.register_noncompetitive_neurons(self.immortal_neurons, role="monitor_immortal")

        # Dar energía alta a las inmortales para que arranquen bien
        with net.lock:
            for idx in self.immortal_neurons:
                net.energy[idx] = max(float(net.energy[idx]), 2.0)

        # Registrar en net.modules para que persista
        net.modules[self.block_id] = self
        self._injected = True

        print(
            f"🧱 [{self.block_id}] Inyectado | "
            f"zona z∈{self.zone_z} | "
            f"{len(en_zona)} neuronas en zona | "
            f"{n_sel} inmortales: {self.immortal_neurons[:4]}...",
            flush=True
        )
        return True

    # ------------------------------------------------------------------
    # INMORTALIDAD — llamar en cada ciclo de homeostasis
    # ------------------------------------------------------------------

    def _enforce_immortality(self, net):
        """
        Las neuronas inmortales no pueden morir ni quedarse sin energía.
        Si alguna cayó por debajo del piso, la reanimamos.
        """
        if not self.immortal_neurons:
            return

        with net.lock:
            for idx in self.immortal_neurons:
                if idx >= net.n:
                    continue
                # Reactivar si se desactivó
                if not net.active[idx]:
                    net.active[idx] = True
                # Piso de energía más alto que el resto de la red
                if net.energy[idx] < 1.5:
                    net.energy[idx] = 1.5

    # ------------------------------------------------------------------
    # MÉTRICAS — leer el estado de la zona
    # ------------------------------------------------------------------

    def _read_zone_health(self, net, t: float):
        """Lee las métricas de salud de la zona sin mantener el lock."""
        with net.lock:
            n_actual  = net.n
            pos       = net.positions[:n_actual].copy()
            active    = net.active[:n_actual].copy()
            energy    = net.energy[:n_actual].copy()
            last_sp   = net.last_spike[:n_actual].copy()

        z_min, z_max = self.zone_z

        en_zona = np.where(
            active &
            (pos[:, 2] >= z_min) & (pos[:, 2] <= z_max)
        )[0]

        if len(en_zona) == 0:
            return

        # Ventana reciente: disparos en los últimos 500ms de tiempo de red
        ventana = 500.0
        n_fired  = int(np.sum(last_sp[en_zona] > (t - ventana)))
        e_mean   = float(np.mean(energy[en_zona]))
        e_min    = float(np.min(energy[en_zona]))

        self.health["n_zone"]      = len(en_zona)
        self.health["n_fired"]     = n_fired
        self.health["spike_rate"]  = n_fired / max(1, len(en_zona))
        self.health["energy_mean"] = e_mean
        self.health["energy_min"]  = e_min
        self.health["last_update"] = t

    # ------------------------------------------------------------------
    # ELECTROSHOCK LOCAL — solo si la salud es crítica
    # ------------------------------------------------------------------

    def _local_electroshock(self, net, t: float, threshold_rate: float = 0.02):
        """
        Si el spike_rate cae bajo el umbral, fuerza disparo directo en las
        neuronas inmortales. Bypassa el umbral de membrana para garantizar
        que last_spike se actualice y spike_rate deje de ser 0.
        NO toca el resto de la red.
        """
        # El Monitor no puede modificar la dinámica durante un trial causal.
        if bool(getattr(net, "_causal_trial_active", False)):
            return
        if self.health["spike_rate"] > threshold_rate:
            return
        if not self.immortal_neurons:
            return

        with net.lock:
            for idx in self.immortal_neurons:
                if idx >= net.n or not net.active[idx]:
                    continue
                # Saltar si está en período refractario
                if t < net.refractory_until[idx]:
                    continue
                # Forzar disparo directo (igual que _execute_spike fase 1)
                net.fired[idx] = 1
                net.last_spike[idx] = float(t)
                net.membrane_potential[idx] = 0.0   # V_RESET
                net.refractory_until[idx] = t + net.refr_period[idx]

        self.health["electroshocks"] += 1

    # ------------------------------------------------------------------
    # UPDATE — llamar desde SlowBioThread cada ciclo
    # ------------------------------------------------------------------

    def update(self, net, t: float):
        """
        Ciclo principal del bloque. Llamar desde SlowBioThread.
        Hace tres cosas:
        1. Mantiene la inmortalidad del núcleo
        2. Lee las métricas de salud
        3. Inyecta electroshock local si es necesario
        4. Emite log de diagnóstico cada log_every ciclos
        """
        if not self._injected:
            return

        self._update_count += 1

        self._enforce_immortality(net)
        self._read_zone_health(net, t)
        self._local_electroshock(net, t)

        if self._update_count % self.log_every == 0:
            self._emit_log()

    # ------------------------------------------------------------------
    # LOG — diagnóstico propio del bloque
    # ------------------------------------------------------------------

    def _precharge_evidence_neurons(self, net, lang_module=None):
        """
        Precarga de energía para las neuronas de evidencia antes de un trial.

        Garantiza que todas las neuronas conectadas a los puertos de evidencia
        tengan energía suficiente para disparar cuando llegue la señal del
        LetraDetector. Sin esto, la red en estado de baja energía ignora
        prácticamente toda la señal de entrenamiento.

        Se llama desde trigger_letter_training() en simulation.py, antes
        de lanzar el estímulo visual.
        """
        PRECHARGE_LEVEL = 1.0   # energía mínima garantizada pre-trial

        # 1. Inmortales propios — siempre
        with net.lock:
            for idx in self.immortal_neurons:
                if idx < net.n and net.active[idx]:
                    if net.energy[idx] < PRECHARGE_LEVEL:
                        net.energy[idx] = PRECHARGE_LEVEL

        # 2. Neuronas de evidencia del módulo de lenguaje
        if lang_module is None:
            lang_module = net.modules.get("lenguaje")
        if lang_module is None:
            return

        evidencia_idxs = set()
        for name, port in lang_module.input_ports.items():
            if name.startswith("evidencia_"):
                for idx in port.connected_neurons:
                    if idx < net.n:
                        evidencia_idxs.add(int(idx))

        with net.lock:
            for idx in evidencia_idxs:
                if net.active[idx] and net.energy[idx] < PRECHARGE_LEVEL:
                    net.energy[idx] = PRECHARGE_LEVEL

        print(f"⚡ [Monitor] Precharge: {len(evidencia_idxs)} neuronas de evidencia → E≥{PRECHARGE_LEVEL}", flush=True)

    def _emit_log(self):
        h = self.health
        print(
            f"🧱 [{self.block_id}] "
            f"zona={self.zone_z} | "
            f"neuronas={h['n_zone']} | "
            f"spike_rate={h['spike_rate']:.3f} | "
            f"energy_mean={h['energy_mean']:.3f} | "
            f"energy_min={h['energy_min']:.3f} | "
            f"shocks={h['electroshocks']}",
            flush=True
        )

    # ------------------------------------------------------------------
    # SERIALIZACIÓN — para que persista con el sistema híbrido
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "block_id":        self.block_id,
            "zone_z":          self.zone_z,
            "zone_x":          self.zone_x,
            "zone_y":          self.zone_y,
            "n_immortal":      self.n_immortal,
            "immortal_neurons": self.immortal_neurons,
            "health":          self.health,
            "injected":        self._injected,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BaseBlock":
        b = cls(
            block_id  = d["block_id"],
            zone_z    = tuple(d["zone_z"]),
            zone_x    = tuple(d.get("zone_x", (0.0, 150.0))),
            zone_y    = tuple(d.get("zone_y", (0.0, 150.0))),
            n_immortal= d["n_immortal"],
        )
        b.immortal_neurons = d.get("immortal_neurons", [])
        b.health           = d.get("health", b.health)
        b._injected        = d.get("injected", False)
        return b


# -----------------------------------------------------------------------
# MONITOR BLOCK — bloque concreto para la zona de lenguaje
# -----------------------------------------------------------------------

class MonitorBlock(BaseBlock):
    """
    Bloque Monitor especializado en la banda de lenguaje (z=37-53).
    
    Es el primer bloque concreto de la arquitectura.
    Además del monitoreo base, trackea específicamente las neuronas
    de evidencia de la currícula y reporta su estado.
    """

    def __init__(self, zone_z=(37.0, 53.0), n_immortal=8, log_every=50):
        super().__init__(
            block_id   = "monitor_lenguaje",
            zone_z     = zone_z,
            n_immortal = n_immortal,
            log_every  = log_every,
        )
        # Métricas extra específicas del monitor de lenguaje
        self.health["evidencia_overlap"] = {}   # {letra: n_neuronas_en_zona}

    def update(self, net, t: float):
        """Extiende el update base con chequeo de puertos de evidencia."""
        if not self._injected:
            return

        self._update_count += 1
        self._enforce_immortality(net)
        self._read_zone_health(net, t)
        self._check_evidence_ports(net)
        self._local_electroshock(net, t)

        if self._update_count % self.log_every == 0:
            self._emit_log()

    def _check_evidence_ports(self, net):
        """
        Lee los puertos de evidencia del módulo de lenguaje y verifica
        cuántas de sus neuronas están dentro de la zona activa.
        Sustituye el [Diag-Overlap] que hoy calcula simulation.py externamente.
        """
        lang = net.modules.get("lenguaje")
        if not lang:
            return

        overlap = {}
        z_min, z_max = self.zone_z

        with net.lock:
            n_actual = net.n
            pos      = net.positions[:n_actual, 2].copy()
            active   = net.active[:n_actual].copy()

        for name, port in lang.input_ports.items():
            if not name.startswith("evidencia_"):
                continue
            letra = name.split("evidencia_")[1]
            en_banda = sum(
                1 for idx in port.connected_neurons
                if idx < n_actual and active[idx] and z_min <= pos[idx] <= z_max
            )
            overlap[letra] = en_banda

        self.health["evidencia_overlap"] = overlap

    def _emit_log(self):
        """Log extendido con información de puertos de evidencia."""
        h = self.health
        overlap_str = " | ".join(
            f"{k}:{v}/30" for k, v in h.get("evidencia_overlap", {}).items()
        )
        print(
            f"🧱 [MonitorLenguaje] "
            f"neuronas={h['n_zone']} | "
            f"spike_rate={h['spike_rate']:.3f} | "
            f"energy={h['energy_mean']:.3f}(min:{h['energy_min']:.3f}) | "
            f"shocks={h['electroshocks']} | "
            f"overlap=[{overlap_str}]",
            flush=True
        )

    @classmethod
    def from_dict(cls, d: dict) -> "MonitorBlock":
        """
        from_dict propio para evitar que BaseBlock.from_dict pase 'block_id'
        al constructor, que no lo acepta (MonitorBlock lo hardcodea internamente).
        """
        b = cls(
            zone_z    = tuple(d.get("zone_z",    (37.0, 53.0))),
            n_immortal= d.get("n_immortal", 8),
            log_every = d.get("log_every",  50),
        )
        b.immortal_neurons = d.get("immortal_neurons", [])
        b.health           = d.get("health", b.health)
        b._injected        = d.get("injected", False)
        return b


# -----------------------------------------------------------------------
# FÁBRICA — para crear bloques desde configuración
# -----------------------------------------------------------------------

from modules.transductor_block import TransductorBlock
from modules.detector_bordes_v import VerticalEdgeDetectorBlock
from modules.detector_bordes_h import HorizontalEdgeDetectorBlock
from modules.combinador_esquina import EsquinaBlock
from modules.detector_diagonales import DiagonalDetectorBlock

def _get_full_registry():
    """
    Construye el registry completo con imports lazy para evitar
    circular imports (los nuevos módulos también importan BaseBlock
    desde este mismo archivo).
    """
    from modules.detector_curvas   import CurvaDetectorBlock
    from modules.detector_simetria import SimetriaDetectorBlock
    from modules.combinador_junction import JunctionBlock
    return {
        "monitor_lenguaje":    MonitorBlock,
        "transductor_vision":  TransductorBlock,
        "detector_bordes_v":   VerticalEdgeDetectorBlock,
        "detector_bordes_h":   HorizontalEdgeDetectorBlock,
        "detector_diagonales": DiagonalDetectorBlock,
        "combinador_esquina":  EsquinaBlock,
        "detector_curvas":     CurvaDetectorBlock,
        "detector_simetria":   SimetriaDetectorBlock,
        "combinador_junction": JunctionBlock,
    }

BLOCK_REGISTRY = {
    "monitor_lenguaje":    MonitorBlock,
    "transductor_vision":  TransductorBlock,
    "detector_bordes_v":   VerticalEdgeDetectorBlock,
    "detector_bordes_h":   HorizontalEdgeDetectorBlock,
    "detector_diagonales": DiagonalDetectorBlock,
    "combinador_esquina":  EsquinaBlock,
}

def create_block(block_type: str, **kwargs) -> Optional[BaseBlock]:
    """
    Crea un bloque por nombre de tipo.
    Ejemplo: create_block("monitor_lenguaje", zone_z=(37.0, 53.0))
    """
    registry = _get_full_registry()
    cls = registry.get(block_type)
    if cls is None:
        print(f"⚠️ [Bloques] Tipo desconocido: '{block_type}'. Disponibles: {list(registry)}")
        return None
    return cls(**kwargs)


def restore_blocks_from_state(state: dict, net) -> dict:
    """
    Restaura bloques desde el estado guardado en persistencia.
    Llamar desde load_state() en simulation.py.
    
    state: el dict que se guardó en net_state["blocks"]
    Retorna: {block_id: instancia_restaurada}
    """
    registry = _get_full_registry()
    restored = {}
    for block_id, block_data in state.items():
        block_type = block_data.get("block_id", block_id)
        cls = registry.get(block_type)
        if cls is None:
            # Fallback: restaurar como BaseBlock genérico
            cls = BaseBlock
        b = cls.from_dict(block_data)
        b._injected = True   # ya estaba inyectado antes de guardar
        if hasattr(net, "register_noncompetitive_neurons") and getattr(b, "immortal_neurons", None):
            net.register_noncompetitive_neurons(b.immortal_neurons, role="monitor_immortal")
        net.modules[block_id] = b
        restored[block_id] = b
        print(f"🧱 [{block_id}] Restaurado desde persistencia | inmortales: {b.immortal_neurons[:4]}...")

    return restored