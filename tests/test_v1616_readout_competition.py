import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config.py'
LANG = ROOT / 'modules' / 'language_io.py'
NET = ROOT / 'core' / 'network.py'


def text(p):
    return p.read_text(encoding='utf-8')


def test_1616_disables_hard_cortical_minimum():
    s = text(CONFIG)
    assert 'CORTICAL_MIN_WINNERS = 0' in s
    src = text(LANG)
    assert 'top-k' in src or 'WTA' in src
    assert 'min_k = min(k, max(1, int(CORTICAL_MIN_WINNERS)))' not in src


def test_1616_readout_has_hard_source_isolation():
    src = text(LANG)
    net = text(NET)
    assert 'readout_allowed_sources' in src
    assert 'Solo las fuentes corticales' in src
    assert 'target_is_readout and int(idx) not in getattr(self, "readout_allowed_sources"' in net


def test_1616_continuous_competition_and_intrapool_fatigue():
    s = text(CONFIG)
    src = text(LANG)
    assert 'READOUT_INTERPOOL_INHIBITION_STEP = 0.22' in s
    assert 'READOUT_INTRAPOOL_FATIGUE_STEP = 0.18' in s
    assert '_apply_readout_competition' in src
    assert 'get_readout_effective_threshold' in src
    assert 'winner_share < 0.45' not in src


def test_1616_competition_is_physical_not_only_telemetry():
    src = text(LANG)
    assert 'membrane_potential[int(idx)] = max(V_REST' in src
    assert 'readout_pool_neuron_fatigue' in src


def test_1616_runtime_telemetry_exposes_isolation_and_competition():
    net = text(NET)
    assert 'readout_foreign_source_blocked' in net
    assert 'readout_competition_events' in net
    assert 'readout_intrapool_fatigue_events' in net
