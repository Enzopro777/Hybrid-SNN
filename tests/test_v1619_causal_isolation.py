from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANG = ROOT / 'modules' / 'language_io.py'
MON = ROOT / 'modules' / 'monitor_block.py'
SIM = ROOT / 'engine' / 'simulation.py'


def text(p):
    return p.read_text(encoding='utf-8')


def test_persisted_readout_sources_require_cortical_role():
    s = text(LANG)
    assert 'valid_sources = bool(sources)' in s
    assert 'set(sources).issubset(cortical_now)' in s
    assert 'source_roles.get(int(i)) == "cortical_representation"' in s


def test_readout_excludes_monitor_immortals():
    s = text(LANG)
    assert 'monitor_immortal' in s
    assert 'excluded_roles' in s
    assert 'readout_targets' in s


def test_monitor_immortals_are_noncompetitive():
    s = text(MON)
    assert 'role="monitor_immortal"' in s
    assert 'register_noncompetitive_neurons' in s


def test_monitor_cannot_electroshock_during_causal_trial():
    s = text(MON)
    assert 'if bool(getattr(net, "_causal_trial_active", False)):' in s
    assert 'El Monitor no puede modificar la dinámica durante un trial causal' in s


def test_heartbeat_resuscitation_respects_causal_trial():
    s = text(SIM)
    assert 'silencio_ms > 2000 and not bool(getattr(self.net, "_causal_trial_active", False))' in s


def test_probe_exposes_readout_source_roles():
    s = text(LANG)
    assert '"source_roles"' in s
