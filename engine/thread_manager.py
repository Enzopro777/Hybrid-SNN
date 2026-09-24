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
            "VisionThread":      self.engine.worker_vision,
            "SpikingThread":     self.engine.worker_spiking,
            "SlowBioThread":     self.engine.worker_slow,
            "CurriculumThread":  self.engine.worker_curriculum,
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
                    # Son hilos de servicio controlados por stop(); no deben
                    # sobrevivir al cierre del intérprete.
                    daemon=False
                )
                self.threads[name] = t
                threads_to_start.append(t)

        # Arrancamos los hilos fuera del lock para evitar condiciones de carrera (Deadlocks)
        for t in threads_to_start:
            t.start()

        logging.info(f"✅ Arquitectura v3.1 iniciada: {len(threads_to_start)} hilos protegidos.")

    def stop(self, timeout_s=10.0):
        """Detiene la simulación y no oculta hilos que sigan vivos."""
        logging.info("🛑 Iniciando protocolo de apagado de hilos...")
        self.engine.is_running = False
        self._stop_event.set()
        with self.lock:
            threads_copy = list(self.threads.items())
        deadline = time.time() + float(timeout_s)
        alive = []
        for name, t in threads_copy:
            if not t.is_alive():
                continue
            t.join(timeout=max(0.05, deadline - time.time()))
            if t.is_alive():
                alive.append(name)
        if alive:
            logging.error("❌ Hilos aún vivos tras el apagado: %s", alive)
            return False
        with self.lock:
            self.threads.clear()
        logging.info("✅ Simulación detenida. CPU y memoria liberadas.")
        return True

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