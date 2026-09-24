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