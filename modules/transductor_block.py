# modules/transductor_block.py
#
# =====================================================================
# BLOQUE TRANSDUCTOR — Arquitectura de Bloques v0.2
# =====================================================================
#
# Qué es:
#   El segundo bloque de la nueva arquitectura (Paso 2 de la hoja de ruta).
#   Intercepta la señal visual ANTES de que llegue a inject_sensory_activity()
#   y la transforma en una señal que la red puede usar eficientemente.
#
# Por qué existe:
#   screen_interface.py produce intensity en [0.04, 0.60].
#   V_THRESHOLD de las neuronas es 20.0.
#   Con esos valores, una neurona necesita 60-360 spikes simultáneos
#   para disparar — prácticamente imposible por diseño.
#   El TransductorBlock rescala esa señal a [3.0, 8.0], donde 3-5
#   spikes bastan para activar una neurona RELAY (thresh=18.0).
#
# Qué hace (en orden de ejecución por cada frame):
#   1. _suppress_noise()      — descarta puntos de intensidad muy baja
#   2. _normalize_amplitude() — rescala el rango al objetivo
#   3. _balance_channels()    — ajusta peso relativo magno vs parvo
#
# Cómo se conecta (sin tocar network.py):
#   El Transductor opera sobre la lista Python current_activations de
#   VisionModule, entre update_from_external() y transport_to_net().
#   vision.py llama self._transductor.process(safe_activations) y usa
#   el resultado. Si el transductor no existe, pasa sin cambios.
#
# Cómo se usa (en simulation.py):
#   from modules.transductor_block import TransductorBlock
#   self.transductor = TransductorBlock()
#   self.transductor.register(self.net)   # una sola vez al arrancar
#   self.vision._transductor = self.transductor
#
# Para IAs colaboradoras:
#   - NO llames .inject(net) — este bloque NO tiene neuronas inmortales.
#     Opera solo sobre listas Python, no sobre la red directamente.
#   - El método principal es .process(activations) → lista normalizada.
#   - Las métricas de .health se actualizan en cada .process() y son
#     legibles por el MonitorBlock o por simulation.py para diagnóstico.
#   - El import de BaseBlock usa ruta relativa al proyecto:
#     from modules.monitor_block import BaseBlock
# =====================================================================

import numpy as np
import logging
from typing import List, Tuple, Optional


# -----------------------------------------------------------------------
# Import de BaseBlock — viene de monitor_block.py (Paso 1 ya implementado)
# -----------------------------------------------------------------------
try:
    from modules.monitor_block import BaseBlock
except ImportError:
    # Fallback por si se corre el archivo directamente en tests
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from modules.monitor_block import BaseBlock


class TransductorBlock(BaseBlock):
    """
    Normaliza la señal visual antes de que llegue a la red.

    Hereda de BaseBlock para reutilizar la serialización y el registro
    en BLOCK_REGISTRY, pero NO usa neuronas inmortales ni .inject().
    Opera exclusivamente sobre listas Python de activaciones.

    Parámetros configurables:
        noise_floor_magno  — umbral mínimo de intensity para canal de movimiento
        noise_floor_parvo  — umbral mínimo de intensity para canal estático (DoG)
        target_min         — mínimo del rango de salida (strength a receive_spike)
        target_max         — máximo del rango de salida
        magno_weight       — multiplicador para el canal de movimiento (p_type=0, z=30)
        parvo_weight       — multiplicador para el canal estático (p_type=1, z=45)
        log_every          — cada cuántos .process() se imprime el log de diagnóstico
    """

    def __init__(
        self,
        zone_z: Tuple[float, float] = (10.0, 25.0),   # separado de banda de lenguaje (37-53)
        noise_floor_magno: float = 0.10,
        noise_floor_parvo: float = 0.08,
        target_min: float = 3.0,
        target_max: float = 8.0,
        magno_weight: float = 0.8,
        parvo_weight: float = 1.2,
        log_every: int = 100,
    ):
        super().__init__(
            block_id   = "transductor_vision",
            zone_z     = zone_z,
            n_immortal = 0,     # no tiene neuronas inmortales
            log_every  = log_every,
        )

        # --- Parámetros de normalización ---
        self.noise_floor_magno = noise_floor_magno
        self.noise_floor_parvo = noise_floor_parvo
        self.target_min        = target_min
        self.target_max        = target_max
        self.magno_weight      = magno_weight
        self.parvo_weight      = parvo_weight

        # --- Métricas propias (ampliamos el dict de BaseBlock) ---
        self.health.update({
            "frames_procesados": 0,
            "puntos_entrada":    0,   # promedio de puntos antes del filtro
            "puntos_salida":     0,   # promedio de puntos después del filtro
            "puntos_descartados_ruido": 0,
            "intensity_min_in":  0.0, # mínimo de intensity cruda (último frame)
            "intensity_max_in":  0.0, # máximo de intensity cruda (último frame)
            "intensity_min_out": 0.0, # mínimo de strength normalizada (último frame)
            "intensity_max_out": 0.0, # máximo de strength normalizada (último frame)
            "magno_count":       0,   # puntos del canal de movimiento (último frame)
            "parvo_count":       0,   # puntos del canal estático (último frame)
        })

        # Acumuladores internos para el log (se resetean cada log_every)
        self._acc_entrada    = 0
        self._acc_salida     = 0
        self._acc_descartado = 0

        # Marcar como "activo" sin necesidad de .inject()
        self._injected = True

    # ------------------------------------------------------------------
    # MÉTODO PRINCIPAL — llamar desde vision.py en transport_to_net()
    # ------------------------------------------------------------------

    def process(self, activations: list) -> list:
        """
        Transforma la lista de activaciones visuales antes de que lleguen
        a inject_sensory_activity().

        Entrada:
            activations — lista de tuplas (x, y, z, intensity)
                          x,y en [0,1] (se escalan a 0-150 en network.py)
                          z en unidades 3D (30=magno, 45=parvo)
                          intensity en [0.04, 0.60] (producida por screen_interface)

        Salida:
            lista de tuplas (x, y, z, strength) con strength en [target_min, target_max]
            Lista vacía si no hay activaciones que superen el piso de ruido.
        """
        if not activations:
            return []

        self._update_count += 1
        self.health["frames_procesados"] += 1

        n_entrada = len(activations)
        self._acc_entrada += n_entrada
        self.health["puntos_entrada"] = n_entrada  # valor del último frame

        # 1. Separar por canal y suprimir ruido
        magno_clean, parvo_clean = self._suppress_noise(activations)

        n_descartados = n_entrada - len(magno_clean) - len(parvo_clean)
        self._acc_descartado += n_descartados
        self.health["puntos_descartados_ruido"] = n_descartados
        self.health["magno_count"] = len(magno_clean)
        self.health["parvo_count"] = len(parvo_clean)

        if not magno_clean and not parvo_clean:
            if self._update_count % self.log_every == 0:
                self._emit_log()
            return []

        # 2. Normalizar amplitud (cada canal por separado para no mezclar rangos)
        magno_norm = self._normalize_amplitude(magno_clean)
        parvo_norm = self._normalize_amplitude(parvo_clean)

        # 3. Balancear canales (multiplicar por su peso relativo)
        result = self._balance_channels(magno_norm, parvo_norm)

        n_salida = len(result)
        self._acc_salida += n_salida
        self.health["puntos_salida"] = n_salida

        # Registrar rango real de este frame para diagnóstico
        if result:
            intensities_out = [r[3] for r in result]
            self.health["intensity_min_out"] = float(min(intensities_out))
            self.health["intensity_max_out"] = float(max(intensities_out))

        if self._update_count % self.log_every == 0:
            self._emit_log()
            # Reset acumuladores
            self._acc_entrada    = 0
            self._acc_salida     = 0
            self._acc_descartado = 0

        return result

    # ------------------------------------------------------------------
    # SUB-FUNCIÓN 1 — Supresión de ruido
    # ------------------------------------------------------------------

    def _suppress_noise(
        self,
        activations: list,
    ) -> Tuple[list, list]:
        """
        Filtra puntos de intensidad muy baja (ruido de umbral de screen_interface)
        y separa por canal (magno z=30 / parvo z=45).

        El canal magno usa noise_floor_magno (0.10 por defecto).
        El canal parvo usa noise_floor_parvo (0.08 por defecto).
        Puntos en otras coordenadas z pasan con el piso magno.

        Retorna (magno_limpio, parvo_limpio) — dos listas separadas.
        """
        magno = []
        parvo = []

        intensities_in = []

        for act in activations:
            try:
                x, y, z, intensity = float(act[0]), float(act[1]), float(act[2]), float(act[3])
            except (IndexError, TypeError, ValueError):
                continue

            intensities_in.append(intensity)

            # z=45 → canal parvo (formas estáticas)
            if abs(z - 45.0) < 5.0:
                if intensity >= self.noise_floor_parvo:
                    parvo.append((x, y, z, intensity))
            else:
                # z=30 → canal magno (movimiento), o cualquier otro z
                if intensity >= self.noise_floor_magno:
                    magno.append((x, y, z, intensity))

        if intensities_in:
            self.health["intensity_min_in"] = float(min(intensities_in))
            self.health["intensity_max_in"] = float(max(intensities_in))

        return magno, parvo

    # ------------------------------------------------------------------
    # SUB-FUNCIÓN 2 — Normalización de amplitud
    # ------------------------------------------------------------------

    def _normalize_amplitude(self, activations: list) -> list:
        """
        Rescala la intensity cruda al rango [target_min, target_max].

        Usa min-max por frame para preservar las diferencias relativas
        dentro del frame (un punto más brillante sigue siendo más brillante).

        Si todos los puntos tienen la misma intensity (caso borde),
        asigna target_min + (target_max - target_min) / 2.

        Retorna la lista con el cuarto elemento (intensity) reemplazado
        por la strength normalizada.
        """
        if not activations:
            return []

        intensities = np.array([a[3] for a in activations], dtype=np.float32)

        i_min = intensities.min()
        i_max = intensities.max()

        span = i_max - i_min

        if span < 1e-6:
            # Todos iguales → valor medio del rango objetivo
            normalized = np.full_like(
                intensities,
                self.target_min + (self.target_max - self.target_min) * 0.5
            )
        else:
            # Min-max al rango objetivo
            normalized = self.target_min + (
                (intensities - i_min) / span
            ) * (self.target_max - self.target_min)

        # Reconstruir lista con la nueva strength
        return [
            (activations[i][0], activations[i][1], activations[i][2], float(normalized[i]))
            for i in range(len(activations))
        ]

    # ------------------------------------------------------------------
    # SUB-FUNCIÓN 3 — Balanceo de canales
    # ------------------------------------------------------------------

    def _balance_channels(self, magno: list, parvo: list) -> list:
        """
        Aplica pesos relativos a cada canal y los une en una sola lista.

        magno_weight=0.8 y parvo_weight=1.2 refuerza levemente las formas
        estáticas (más útiles para reconocimiento de letras) sobre el
        movimiento (más ruidoso en escenas con letras quietas).

        El clip final garantiza que ningún punto salga del rango objetivo
        incluso si los pesos están mal configurados.
        """
        result = []

        for x, y, z, strength in magno:
            s = float(np.clip(
                strength * self.magno_weight,
                self.target_min,
                self.target_max
            ))
            result.append((x, y, z, s))

        for x, y, z, strength in parvo:
            s = float(np.clip(
                strength * self.parvo_weight,
                self.target_min,
                self.target_max
            ))
            result.append((x, y, z, s))

        return result

    # ------------------------------------------------------------------
    # LOG DE DIAGNÓSTICO
    # ------------------------------------------------------------------

    def _emit_log(self):
        h = self.health
        n_frames = self.log_every  # ventana del log

        # Promedios de la ventana
        avg_entrada    = self._acc_entrada    / max(1, n_frames)
        avg_salida     = self._acc_salida     / max(1, n_frames)
        avg_descartado = self._acc_descartado / max(1, n_frames)

        print(
            f"🔌 [Transductor] "
            f"frames={h['frames_procesados']} | "
            f"entrada={avg_entrada:.0f}pts | "
            f"salida={avg_salida:.0f}pts | "
            f"ruido_descartado={avg_descartado:.0f}pts | "
            f"in=[{h['intensity_min_in']:.3f},{h['intensity_max_in']:.3f}] "
            f"out=[{h['intensity_min_out']:.3f},{h['intensity_max_out']:.3f}] | "
            f"magno={h['magno_count']} parvo={h['parvo_count']}",
            flush=True
        )

    # ------------------------------------------------------------------
    # REGISTRO EN LA RED (sin neuronas, solo para persistencia)
    # ------------------------------------------------------------------

    def register(self, net):
        """
        Registra el bloque en net.modules para que persista con el sistema
        híbrido (.npz + .pkl). NO llama inject() porque no tiene neuronas.
        Llamar una sola vez desde simulation.py al iniciar.
        """
        net.modules[self.block_id] = self
        print(
            f"🔌 [Transductor] Registrado | "
            f"noise_floor magno={self.noise_floor_magno} parvo={self.noise_floor_parvo} | "
            f"target=[{self.target_min}, {self.target_max}] | "
            f"pesos magno={self.magno_weight} parvo={self.parvo_weight}",
            flush=True
        )

    # ------------------------------------------------------------------
    # SERIALIZACIÓN — compatible con el sistema híbrido existente
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "noise_floor_magno": self.noise_floor_magno,
            "noise_floor_parvo": self.noise_floor_parvo,
            "target_min":        self.target_min,
            "target_max":        self.target_max,
            "magno_weight":      self.magno_weight,
            "parvo_weight":      self.parvo_weight,
        })
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TransductorBlock":
        b = cls(
            zone_z             = tuple(d.get("zone_z", (10.0, 25.0))),   # default nuevo sin solapamiento
            noise_floor_magno  = d.get("noise_floor_magno", 0.10),
            noise_floor_parvo  = d.get("noise_floor_parvo", 0.08),
            target_min         = d.get("target_min",        3.0),
            target_max         = d.get("target_max",        8.0),
            magno_weight       = d.get("magno_weight",      0.8),
            parvo_weight       = d.get("parvo_weight",      1.2),
        )
        b.health    = d.get("health", b.health)
        b._injected = True   # siempre activo tras restaurar
        return b
