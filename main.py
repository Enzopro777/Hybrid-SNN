#main.py


import time
import logging
import os
import pickle  # <-- NUEVO: Para inspeccionar el tamaño real de la red guardada

# --- Importaciones del proyecto ---
from config import *
from core.network import NeuralNetwork
from engine.simulation import SimulationEngine
from engine.thread_manager import ThreadManager

try:
    from screen_interface import ScreenProcessor
    HAS_SCREEN = True
except ImportError:
    logging.warning("screen_interface.py no encontrado. Visión desactivada.")
    HAS_SCREEN = False

def main():
    print("-" * 60)
    print("🚀 INICIANDO SIMULACIÓN  SNN  -  (Multi-Threaded)")
    print("🎯 Reward System: ACTIVO (Dopamina por Visión)")
    print("🧪 IA-1.6.29f: readout estable, homeostasis anti-hiperactividad, arranque experimental limpio")
    print("-" * 60)

    # Nombres base coherentes con simulation.py
    filename_base = "simulation_state"
    file_meta = f"{filename_base}_meta.pkl"
    file_np = f"{filename_base}_matrices.npz"
    try:
        from systems.trial_commit_journal import TrialCommitJournal
        journal = TrialCommitJournal(str(globals().get("TRIAL_COMMIT_JOURNAL_PATH", "trial_commit_1.6.29f.json")))
        last_commit = journal.load()
        if last_commit:
            print(
                f"🧾 [TrialCommit] Última frontera durable: trial={last_commit.get('sequence')} "
                f"label={last_commit.get('label')} frames={last_commit.get('frames')} "
                f"modo={last_commit.get('trial_mode')}"
            )
    except Exception:
        pass

    # ---------------------------------------------------------------------
    # 1. INICIALIZACIÓN DE LA RED
    # ---------------------------------------------------------------------
    # IA 1.6.24: las pruebas experimentales arrancan limpias por defecto.
    # Para reanudar aprendizaje histórico, cambiar CLEAN_EXPERIMENT_DEFAULT a False.
    clean_start = bool(globals().get("CLEAN_EXPERIMENT_DEFAULT", True))
    if clean_start:
        print("🧪 [Experiment] Arranque limpio: sin snapshots persistidos.")
        net = NeuralNetwork(n=NUM_NEURONS, max_neurons=MAX_NEURONS)
        sim = SimulationEngine(net)
    elif os.path.exists(file_meta) and os.path.exists(file_np):
        print(f"💾 [Persistencia] ¡Guardado previo detectado! Reconstruyendo cerebro...")
        try:
            with open(file_meta, "rb") as f:
                data = pickle.load(f)
            n_guardado = data['network_meta'].get('n', NUM_NEURONS)
            
            print(f"🧠 Dimensionando NeuralNetwork a {n_guardado} neuronas...")
            # skip_wiring=True: vamos a llamar a sim.load_state() dos líneas
            # más abajo, que sobrescribe todo el cableado igual — no tiene
            # sentido pagar el costo O(n²) de _init_connections_sparse()
            # para tirarlo a la basura al toque.
            net = NeuralNetwork(n=n_guardado, skip_wiring=True, max_neurons=MAX_NEURONS)
            sim = SimulationEngine(net)
            
            if sim.load_state(filename_base):
                # --- IMPORTANTE: load_state() restaura net.max_neurons desde
                # el guardado (viaja en network_meta junto con el resto de
                # los escalares). Si el guardado es viejo y tenía un techo
                # menor (p.ej. 8000), te lo pisa acá mismo. Lo reafirmamos
                # al valor de config.py para que MAX_NEURONS sea siempre la
                # última palabra, sin importar con qué techo se guardó antes.
                if net.max_neurons != MAX_NEURONS:
                    print(f"🧬 [Config] max_neurons del guardado ({net.max_neurons}) "
                          f"difiere de MAX_NEURONS en config.py ({MAX_NEURONS}) — "
                          f"actualizando.")
                    net.max_neurons = MAX_NEURONS
                print(f"✅ Cerebro restaurado y enlazado. Listo para reanudar.")
            else:
                raise Exception("Fallo en load_state interno.")
                
        except Exception as e:
            logging.error(f"Error en reconstrucción: {e}. Iniciando red limpia.")
            net = NeuralNetwork(n=NUM_NEURONS, max_neurons=MAX_NEURONS)
            sim = SimulationEngine(net)
    else:
        print("🌱 [Persistencia] No hay guardado válido: creando red base con NUM_NEURONS neuronas configuradas.")
        net = NeuralNetwork(n=NUM_NEURONS, max_neurons=MAX_NEURONS)
        sim = SimulationEngine(net)
    
    # --- NUEVO: backfill de sinapsis (una vez por arranque; idempotente) ---
    if hasattr(net, 'backfill_orphan_connections'):
        net.backfill_orphan_connections(min_connections=4)

    # 2. Procesador de pantalla para la entrada visual (si existe)
    scr_processor = ScreenProcessor() if HAS_SCREEN else None

    # 3. Ejecución
    try:
        # sim.run ya inicializa e inicia internamente el ThreadManager
        sim.run(screen_processor=scr_processor)
    except Exception as e:
        logging.error(f"Error crítico en la ejecución: {e}")
    finally:
        print("\n✅ Simulación finalizada correctamente.")
        print(f"📊 Estado final: {net.n} neuronas | Memoria liberada.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(threadName)s] %(message)s')
    main()