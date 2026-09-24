"""
test_v1611_pool_separation.py
─────────────────────────────
Tests para IA 1.6.11: separación funcional de pools y mecanismos WTA.

El diagnóstico de 1.6.10 reveló que los pools producen actividad casi idéntica
(~122 spikes cada uno) aunque los pesos divergan. Estas pruebas verifican que:

1. La inhibición lateral entre pools se inicializa y resetea correctamente.
2. apply_lateral_pool_inhibition() suprime activamente los pools rivales.
3. get_pool_separation_metrics() reporta métricas reales de separación.
4. Los parámetros 1.6.11 (temperatura, competitor_rate, lateral_inhibition)
   están en el rango correcto.
5. Con pesos desiguales, la separación de scores aumenta respecto a 1.6.10.
6. El scale adaptativo aumenta cuando el target fue inhibido lateralmente.
"""

import threading
import numpy as np
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# ─── Fixture: red mínima con readout configurado ───────────────────────────

class FakeNet:
    def __init__(self, n=64):
        self.n = n
        self.lock = threading.RLock()
        self.current_time = 100.0
        self.last_spike = np.zeros(n, dtype=float)
        self.last_update_time = np.zeros(n, dtype=float)
        self.refractory_until = np.zeros(n, dtype=float)
        self.membrane_potential = np.zeros(n, dtype=float)
        self.active = np.ones(n, dtype=bool)
        self.connections = [[] for _ in range(n)]
        self.weights = {}
        self.readout_target_to_symbol = {}
        self.tau_m = np.full(n, 22.0, dtype=float)
        self.v_thresh = np.full(n, 2.6, dtype=float)
        self.refr_period = np.full(n, 2.0, dtype=float)
        self.energy = np.ones(n, dtype=float)


def make_readout_with_5_symbols():
    """Readout con 5 símbolos, 2 fuentes, 1 pool neuron por símbolo."""
    net = FakeNet()
    from modules.language_io import LetterIOModule
    lang = LetterIOModule(net=net, symbols=("X", "O", "T", "A", "E"))
    lang._readout_ready = True
    lang._readout_sources = [1, 2]
    # Pool: 1 neurona por símbolo
    lang._readout_pools = {s: [10 + i] for i, s in enumerate(lang.symbols)}
    for src in lang._readout_sources:
        net.connections[src] = [10, 11, 12, 13, 14]
        for tgt in net.connections[src]:
            net.weights[(src, tgt)] = 1.05
    lang.begin_clean_trial(0.0, net=net)
    return net, lang


# ─── Tests de parámetros config ────────────────────────────────────────────

def test_temperature_increased_for_better_weight_sensitivity():
    """v1.6.11: temperatura softmax debe ser >= 0.60 para separar clases con pesos distintos."""
    cfg = (ROOT / "config.py").read_text(encoding="utf-8")
    # Extraer valor numérico
    import re
    m = re.search(r"READOUT_SCORE_TEMPERATURE\s*=\s*([0-9.]+)", cfg)
    assert m, "READOUT_SCORE_TEMPERATURE no encontrado en config.py"
    val = float(m.group(1))
    assert val >= 0.60, (
        f"READOUT_SCORE_TEMPERATURE={val} demasiado bajo. "
        "La baja temperatura aplana diferencias de peso, colapsa scores al mismo valor."
    )


def test_competitor_rate_effective_enough():
    """v1.6.11: competitor_rate debe ser >= 0.015 para depresión real del competidor."""
    cfg = (ROOT / "config.py").read_text(encoding="utf-8")
    import re
    m = re.search(r"READOUT_COMPETITOR_RATE\s*=\s*([0-9.]+)", cfg)
    assert m, "READOUT_COMPETITOR_RATE no encontrado"
    val = float(m.group(1))
    assert val >= 0.015, (
        f"READOUT_COMPETITOR_RATE={val} demasiado bajo. "
        "El pool rival no recibe suficiente depresión para forzar diferenciación."
    )


def test_lateral_inhibition_params_present():
    """v1.6.11: parámetros de inhibición lateral deben estar en config.py."""
    cfg = (ROOT / "config.py").read_text(encoding="utf-8")
    assert "READOUT_LATERAL_INHIBITION" in cfg, "Falta READOUT_LATERAL_INHIBITION"
    assert "READOUT_LATERAL_DECAY" in cfg, "Falta READOUT_LATERAL_DECAY"
    assert "READOUT_MIN_UNIQUE_POOL_NEURONS" in cfg, "Falta READOUT_MIN_UNIQUE_POOL_NEURONS"

    import re
    m_inh = re.search(r"READOUT_LATERAL_INHIBITION\s*=\s*([0-9.]+)", cfg)
    m_dec = re.search(r"READOUT_LATERAL_DECAY\s*=\s*([0-9.]+)", cfg)
    assert m_inh and 0.05 <= float(m_inh.group(1)) <= 0.50, "READOUT_LATERAL_INHIBITION fuera de rango razonable"
    assert m_dec and 0.80 <= float(m_dec.group(1)) <= 0.99, "READOUT_LATERAL_DECAY fuera de rango"


# ─── Tests de estado interno de LetterIOModule ─────────────────────────────

def test_pool_inhibition_state_initialized():
    """v1.6.11: _pool_inhibition_state debe existir y estar en cero al inicio."""
    net, lang = make_readout_with_5_symbols()
    assert hasattr(lang, "_pool_inhibition_state"), "_pool_inhibition_state no existe"
    assert all(v == 0.0 for v in lang._pool_inhibition_state.values()), \
        "inhibition state debe arrancar en cero"
    assert set(lang._pool_inhibition_state.keys()) == set(lang.symbols)


def test_pool_separation_history_initialized():
    """v1.6.11: _pool_separation_history debe existir y estar vacío al inicio."""
    net, lang = make_readout_with_5_symbols()
    assert hasattr(lang, "_pool_separation_history"), "_pool_separation_history no existe"
    assert lang._pool_separation_history == []


def test_pool_inhibition_state_reset_on_begin_clean_trial():
    """v1.6.11: begin_clean_trial debe resetear _pool_inhibition_state."""
    net, lang = make_readout_with_5_symbols()
    # Simular estado sucio
    lang._pool_inhibition_state["O"] = 0.5
    lang._pool_inhibition_state["T"] = 0.3
    # Reset
    lang.begin_clean_trial(10.0, net=net)
    assert all(v == 0.0 for v in lang._pool_inhibition_state.values()), \
        "begin_clean_trial debe resetear inhibition state a cero"


def test_pool_unique_spikes_reset_on_begin_clean_trial():
    """v1.6.11: begin_clean_trial debe resetear contadores de neuronas únicas."""
    net, lang = make_readout_with_5_symbols()
    lang._pool_unique_spikes_per_trial["X"] = 5
    lang.begin_clean_trial(10.0, net=net)
    assert all(v == 0 for v in lang._pool_unique_spikes_per_trial.values()), \
        "begin_clean_trial debe resetear _pool_unique_spikes_per_trial"


# ─── Tests del método get_pool_separation_metrics ──────────────────────────

def test_get_pool_separation_metrics_exists():
    """v1.6.11: get_pool_separation_metrics() debe existir."""
    net, lang = make_readout_with_5_symbols()
    assert hasattr(lang, "get_pool_separation_metrics"), \
        "Falta método get_pool_separation_metrics"
    metrics = lang.get_pool_separation_metrics()
    assert isinstance(metrics, dict)
    assert "separation_margin_mean_10" in metrics
    assert "pool_inhibition_state" in metrics
    assert "n_trials_recorded" in metrics


def test_separation_metrics_increase_after_asymmetric_trial():
    """v1.6.11: el margen de separación debe registrarse tras un trial asimétrico."""
    net, lang = make_readout_with_5_symbols()
    lang._trial_source_spike_count = 2
    lang._trial_source_spike_counts = {1: 1, 2: 1}
    # X tiene muchos spikes, el resto casi nada
    lang._trial_pool_spike_counts = {"X": 50, "O": 5, "T": 3, "A": 2, "E": 1}
    lang._trial_pool_indices["X"] = {10}
    lang.apply_readout_learning(net, [1, 2], "X", prediction="X")
    metrics = lang.get_pool_separation_metrics()
    # Debe haber al menos 1 trial registrado
    assert metrics["n_trials_recorded"] >= 1
    # El margen debe ser > 0 cuando hay diferencia real
    assert metrics["separation_margin_mean_10"] > 0.0, \
        "Separación no registrada con actividad asimétrica"


def test_separation_margin_low_when_symmetric():
    """v1.6.11: con actividad simétrica entre pools, margen debe ser < 0.10."""
    net, lang = make_readout_with_5_symbols()
    lang._trial_source_spike_count = 2
    lang._trial_source_spike_counts = {1: 1, 2: 1}
    # Todos los pools disparan igual — el problema central de 1.6.10
    lang._trial_pool_spike_counts = {"X": 122, "O": 122, "T": 122, "A": 122, "E": 122}
    lang._trial_pool_indices["X"] = {10}
    lang.apply_readout_learning(net, [1, 2], "X", prediction="O")
    metrics = lang.get_pool_separation_metrics()
    assert metrics["separation_margin_mean_10"] < 0.10, \
        "Con actividad simétrica el margen debe ser prácticamente 0"


# ─── Tests del método apply_lateral_pool_inhibition ────────────────────────

def test_lateral_inhibition_method_exists():
    """v1.6.11: apply_lateral_pool_inhibition debe existir."""
    net, lang = make_readout_with_5_symbols()
    assert hasattr(lang, "apply_lateral_pool_inhibition"), \
        "Falta método apply_lateral_pool_inhibition"


def test_lateral_inhibition_suppresses_rival_pools():
    """v1.6.11: cuando X domina, los demás pools deben ser suprimidos."""
    net, lang = make_readout_with_5_symbols()
    # Simular que X disparó mucho más que los demás
    lang._trial_pool_spike_counts = {"X": 80, "O": 10, "T": 5, "A": 3, "E": 2}
    # Potencial inicial de todos los pools en 1.0
    for i in range(10, 15):
        net.membrane_potential[i] = 1.0
    pot_before_O = float(net.membrane_potential[11])
    lang.apply_lateral_pool_inhibition(net, t=100.0)
    pot_after_O = float(net.membrane_potential[11])
    assert pot_after_O < pot_before_O, \
        "La inhibición lateral debe reducir el potencial del pool rival (O)"


def test_lateral_inhibition_not_applied_without_active_trial():
    """v1.6.11: inhibición lateral no debe aplicarse si no hay trial activo."""
    net, lang = make_readout_with_5_symbols()
    lang._clean_trial_baseline_t = None  # Sin trial activo
    for i in range(10, 15):
        net.membrane_potential[i] = 1.0
    lang.apply_lateral_pool_inhibition(net, t=100.0)
    # Potenciales no deben cambiar
    for i in range(10, 15):
        assert float(net.membrane_potential[i]) == 1.0, \
            "Sin trial activo, inhibición lateral no debe modificar potenciales"


# ─── Tests de score con temperatura mejorada ───────────────────────────────

def test_scores_more_separated_with_higher_temperature():
    """v1.6.11: con temperatura 0.65 y diferencia de pesos, los scores deben separarse más."""
    net, lang = make_readout_with_5_symbols()
    lang.begin_clean_trial(0.0, net=net)
    lang._trial_source_spike_count = 3
    lang._trial_source_spike_counts = {1: 2, 2: 1}

    # Dar a X un peso significativamente mayor
    for src in [1, 2]:
        net.weights[(src, 10)] = 1.80  # X pool
        net.weights[(src, 11)] = 0.75  # O pool
        net.weights[(src, 12)] = 0.75  # T pool
        net.weights[(src, 13)] = 0.75  # A pool
        net.weights[(src, 14)] = 0.75  # E pool

    # 1.6.25: la autoridad es la actividad postsináptica del pool.
    lang.note_readout_spike(10, 20.0)
    lang.note_readout_spike(10, 30.0)
    scores = lang.readout_scores(net, t=100.0)
    # Con temperatura 0.65, el pool X con spikes reales debe liderar claramente
    assert scores["X"] > scores["O"], \
        f"Con pesos distintos, X debe superar O. Scores: {scores}"
    margin = scores["X"] - scores["O"]
    assert margin > 0.05, \
        f"Margen esperado > 0.05 con temperatura 0.65, obtenido: {margin:.4f}"


def test_telemetry_includes_separation_fields():
    """v1.6.11: get_readout_telemetry debe incluir fields de separación."""
    net, lang = make_readout_with_5_symbols()
    lang._trial_source_spike_count = 1
    lang._trial_source_spike_counts = {1: 1}
    lang._trial_pool_spike_counts = {"X": 10, "O": 2, "T": 1, "A": 1, "E": 1}
    lang._trial_pool_indices["X"] = {10}
    lang.apply_readout_learning(net, [1, 2], "X", prediction="X")
    audit = lang._diag_last_learning_audit
    assert "pool_unique_spikes" in audit, "Falta pool_unique_spikes en audit"
    assert "pool_inhibition_state" in audit, "Falta pool_inhibition_state en audit"
    assert "pool_separation_margin_mean" in audit, "Falta pool_separation_margin_mean en audit"


# ─── Test de versión ────────────────────────────────────────────────────────

def test_version_file_exists_and_mentions_1611():
    """v1.6.11: debe haber un archivo de versión que mencione 1.6.11."""
    version_file = ROOT / "VERSION_V1.11.md"
    assert version_file.exists(), f"Falta {version_file}"
    content = version_file.read_text(encoding="utf-8")
    assert "1.6.11" in content, "VERSION_V1.11.md debe mencionar 1.6.11"
    assert "lateral" in content.lower() or "separaci" in content.lower(), \
        "VERSION debe mencionar la inhibición lateral o separación"
