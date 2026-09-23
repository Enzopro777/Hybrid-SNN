# menu_arranque.py
#
# =====================================================================
# MENÚ DE ARRANQUE CON AUTO-CALIBRACIÓN — v0.1
# =====================================================================
#
# Reemplaza el arranque directo por python main.py con una interfaz
# interactiva que:
#
#   1. Muestra el estado actual de todos los detectores y combinadores
#   2. Ofrece opciones:
#      A) Iniciar normalmente (carga calibración guardada)
#      B) Recalibrar y arrancar (warmup + ajuste automático de umbrales)
#      C) Ajuste manual rápido (cambiar umbrales individualmente)
#      D) Solo calibrar sin arrancar (para diagnóstico)
#      E) Ver logs del último warmup
#      F) Salir
#
# Uso:
#   python menu_arranque.py
#
# Lo que hace internamente:
#   - Inicializa la red y los módulos igual que main.py
#   - Antes de sim.run() le da control al menú
#   - Dependiendo de la opción, invoca AutoCalibrador o arranca directo
#
# =====================================================================

import os
import sys
import time
import json
import logging

# Importaciones del proyecto
from config import *
from core.network import NeuralNetwork
from engine.simulation import SimulationEngine

try:
    from screen_interface import ScreenProcessor
    HAS_SCREEN = True
except ImportError:
    logging.warning("screen_interface.py no encontrado. Visión desactivada.")
    HAS_SCREEN = False

try:
    from modules.auto_calibrador import AutoCalibrador, CALIB_FILE
    HAS_CALIB = True
except ImportError:
    HAS_CALIB = False

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║          🧠  SNN — MENÚ DE ARRANQUE v0.1                     ║
║          Red Neuronal de Spikes — Sistema Visual             ║
╚══════════════════════════════════════════════════════════════╝
"""

MENU_OPTIONS = """
  [A] Iniciar normalmente        (usa calibración guardada si existe)
  [B] Recalibrar y arrancar      (warmup automático + ajuste umbrales)
  [C] Ajuste manual de umbrales  (cambiar parámetros individualmente)
  [D] Solo calibrar              (diagnóstico sin arrancar la red)
  [E] Ver estado de detectores   (umbrales actuales + última calibración)
  [F] Salir

"""


def _ask(prompt: str, valid: list) -> str:
    while True:
        try:
            ans = input(prompt).strip().upper()
        except (EOFError, KeyboardInterrupt):
            return 'F'
        if ans in valid:
            return ans
        print(f"  Opción no válida. Elegí entre: {', '.join(valid)}")


def _init_sim() -> tuple:
    """Inicializa NeuralNetwork + SimulationEngine igual que main.py."""
    filename_base = "simulation_state"
    file_meta     = f"{filename_base}_meta.pkl"
    file_np       = f"{filename_base}_matrices.npz"

    import pickle

    clean_start = bool(globals().get("CLEAN_EXPERIMENT_DEFAULT", True))
    if clean_start:
        print("🧪 [Experiment] Arranque limpio: ignorando snapshots persistidos.")
        net = NeuralNetwork(n=NUM_NEURONS, max_neurons=MAX_NEURONS)
        sim = SimulationEngine(net)
    elif os.path.exists(file_meta) and os.path.exists(file_np):
        print("💾 Guardado previo detectado. Reconstruyendo...")
        try:
            with open(file_meta, "rb") as f:
                data = pickle.load(f)
            n_guardado = data['network_meta'].get('n', NUM_NEURONS)
            net = NeuralNetwork(n=n_guardado, skip_wiring=True, max_neurons=MAX_NEURONS)
            sim = SimulationEngine(net)
            if sim.load_state(filename_base):
                if net.max_neurons != MAX_NEURONS:
                    net.max_neurons = MAX_NEURONS
                print("✅ Red restaurada.")
            else:
                raise Exception("Fallo en load_state.")
        except Exception as e:
            logging.error(f"Error restaurando: {e}. Red limpia.")
            net = NeuralNetwork(n=NUM_NEURONS, max_neurons=MAX_NEURONS)
            sim = SimulationEngine(net)
    else:
        print("🌱 Red nueva (10000 neuronas configuradas).")
        net = NeuralNetwork(n=NUM_NEURONS, max_neurons=MAX_NEURONS)
        sim = SimulationEngine(net)

    if hasattr(net, 'backfill_orphan_connections'):
        net.backfill_orphan_connections(min_connections=4)

    return net, sim


def _show_detector_status(sim):
    """Muestra el estado de todos los detectores y combinadores."""
    print("\n" + "─"*60)
    print("📡 ESTADO DE DETECTORES Y COMBINADORES")
    print("─"*60)

    detector_map = {
        "Transductor":         ("transductor",         ["BLOCK_TYPE"]),
        "Bordes Verticales":   ("edge_detector_v",     ["edge_threshold", "output_z"]),
        "Bordes Horizontales": ("edge_detector_h",     ["edge_threshold", "output_z"]),
        "Diagonales":          ("diag_detector",       ["edge_threshold", "barra_z", "slash_z"]),
        "Curvas":              ("curva_detector",      ["curv_threshold", "output_z"]),
        "Simetría":            ("simetria_detector",   ["sym_threshold", "sym_v_z", "sym_h_z"]),
        "Combinador Esquinas": ("combinador_esquina",  ["corner_threshold", "output_z",
                                                        "min_bordes_v", "min_bordes_h"]),
        "Combinador Junctions":("junction_detector",   ["junction_threshold", "junction_z",
                                                         "min_bordes"]),
    }

    for nombre, (attr, params) in detector_map.items():
        obj = getattr(sim, attr, None)
        if obj is None:
            print(f"  ❌ {nombre:<22} — no inicializado")
            continue
        param_str = "  ".join(
            f"{p}={getattr(obj, p, '?')}"
            for p in params
            if p != "BLOCK_TYPE"
        )
        block_type = getattr(obj, 'BLOCK_TYPE', type(obj).__name__)
        print(f"  ✅ {nombre:<22} [{block_type}]  {param_str}")

    # Mostrar info de calibración guardada
    if os.path.exists(CALIB_FILE):
        try:
            with open(CALIB_FILE) as f:
                cd = json.load(f)
            age_h = (time.time() - cd.get("timestamp", 0)) / 3600
            stats = cd.get("warmup_stats", {})
            t_mean = stats.get("transductor", {}).get("mean", 0)
            print(f"\n  📋 Calibración guardada: hace {age_h:.1f}h | "
                  f"densidad transductor: {t_mean:.0f} pts/frame")
        except Exception:
            pass
    else:
        print("\n  📋 Sin calibración guardada — primera vez o fue borrada.")
    print("─"*60)


def _menu_manual_adjust(sim):
    """Menú para ajuste manual de umbrales individuales."""
    print("\n🔧 AJUSTE MANUAL DE UMBRALES")
    print("  Ingresá: <detector>.<param>=<valor>")
    print("  Ejemplos:")
    print("    edge_detector_v.edge_threshold=0.03")
    print("    combinador_esquina.corner_threshold=0.40")
    print("    curva_detector.curv_threshold=0.06")
    print("  [Enter vacío para terminar]\n")

    while True:
        try:
            line = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            break
        try:
            if '=' not in line or '.' not in line:
                print("  Formato: <detector>.<param>=<valor>")
                continue
            left, val_str = line.split('=', 1)
            det_name, param = left.strip().rsplit('.', 1)
            det_name = det_name.strip()
            param    = param.strip()
            val      = float(val_str.strip())

            obj = getattr(sim, det_name, None)
            if obj is None:
                print(f"  ❌ '{det_name}' no encontrado.")
                continue
            if not hasattr(obj, param):
                print(f"  ❌ '{param}' no existe en {det_name}.")
                continue

            old_val = getattr(obj, param)
            setattr(obj, param, val)
            print(f"  ✅ {det_name}.{param}: {old_val} → {val}")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    print("  Ajuste manual finalizado.")


def _show_warmup_log():
    """Muestra estadísticas del último warmup guardado."""
    if not os.path.exists(CALIB_FILE):
        print("\n  Sin datos de warmup guardados.")
        return
    try:
        with open(CALIB_FILE) as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Error: {e}")
        return

    stats = data.get("warmup_stats", {})
    thresholds = data.get("thresholds", {})

    print("\n" + "─"*60)
    print("📊 ÚLTIMO WARMUP")
    print("─"*60)
    age_h = (time.time() - data.get("timestamp", 0)) / 3600
    print(f"  Ejecutado hace: {age_h:.1f}h  |  frames: {stats.get('n_frames', '?')}\n")

    for key in ["transductor", "bordes_v", "bordes_h", "barra", "slash", "curvas"]:
        s = stats.get(key, {})
        mean = s.get("mean", 0.0)
        std  = s.get("std",  0.0)
        if mean > 0:
            ratio = mean / max(1.0, stats.get("transductor", {}).get("mean", 1.0))
            print(f"  {key:<16}  media={mean:6.1f}  std={std:5.1f}  "
                  f"ratio={ratio:.2f}")

    print("\n  Umbrales calibrados:")
    for det, params in thresholds.items():
        for p, v in params.items():
            print(f"    {det}.{p} = {v:.4f}")
    print("─"*60)


def main():
    """Punto de entrada del menú de arranque."""
    print(BANNER)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(threadName)s] %(message)s'
    )

    # ── Inicializar red y simulación ──────────────────────────────────
    print("🔄 Inicializando red y módulos...\n")
    try:
        net, sim = _init_sim()
    except Exception as e:
        print(f"❌ Error crítico inicializando: {e}")
        sys.exit(1)

    scr_processor = ScreenProcessor() if HAS_SCREEN else None

    # ── Mostrar estado rápido ─────────────────────────────────────────
    _show_detector_status(sim)

    # ── Bucle del menú ────────────────────────────────────────────────
    while True:
        print(MENU_OPTIONS)
        opt = _ask("  ¿Qué querés hacer? [A/B/C/D/E/F]: ", list("ABCDEF"))

        # ── A: Iniciar normalmente ────────────────────────────────────
        if opt == 'A':
            if HAS_CALIB and not bool(globals().get("CLEAN_EXPERIMENT_REMOVE_CALIBRATION", True)):
                calib = AutoCalibrador(sim, verbose=True)
                loaded = calib.run_or_load()
                if not loaded:
                    print("ℹ️  Sin calibración previa — arrancando con umbrales por defecto.")
            elif HAS_CALIB:
                print("🧪 [Experiment] Calibración persistida ignorada: usando parámetros de arranque limpios.")
            print("\n🚀 Iniciando simulación...")
            try:
                sim.run(screen_processor=scr_processor)
            except KeyboardInterrupt:
                print("\n⏹ Simulación interrumpida por el usuario.")
            break

        # ── B: Recalibrar y arrancar ──────────────────────────────────
        elif opt == 'B':
            if not HAS_CALIB:
                print("❌ Módulo auto_calibrador no disponible.")
                continue
            calib = AutoCalibrador(sim, verbose=True)
            calib.force_recalibrate()
            print("\n🚀 Iniciando simulación con parámetros calibrados...")
            try:
                sim.run(screen_processor=scr_processor)
            except KeyboardInterrupt:
                print("\n⏹ Simulación interrumpida por el usuario.")
            break

        # ── C: Ajuste manual ──────────────────────────────────────────
        elif opt == 'C':
            _menu_manual_adjust(sim)
            # Después del ajuste, preguntar si arrancar
            arr = _ask("\n  ¿Arrancar ahora? [S/N]: ", ['S', 'N'])
            if arr == 'S':
                print("\n🚀 Iniciando simulación...")
                try:
                    sim.run(screen_processor=scr_processor)
                except KeyboardInterrupt:
                    print("\n⏹ Simulación interrumpida.")
                break

        # ── D: Solo calibrar ─────────────────────────────────────────
        elif opt == 'D':
            if not HAS_CALIB:
                print("❌ Módulo auto_calibrador no disponible.")
                continue
            calib = AutoCalibrador(sim, verbose=True)
            calib.force_recalibrate()
            print("\n✅ Calibración completada. Volviendo al menú.")

        # ── E: Ver estado ────────────────────────────────────────────
        elif opt == 'E':
            _show_detector_status(sim)
            _show_warmup_log()

        # ── F: Salir ─────────────────────────────────────────────────
        elif opt == 'F':
            print("\n👋 Saliendo.")
            break

    print(f"\n✅ Finalizado. {net.n} neuronas en red.")


if __name__ == "__main__":
    main()
