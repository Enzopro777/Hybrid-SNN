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