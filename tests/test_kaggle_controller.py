"""Offline checks for Kaggle controller; no credentials or GPU required."""
import importlib.util
import json
from pathlib import Path
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'kaggle_controller.py'
spec = importlib.util.spec_from_file_location('kaggle_controller', SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_accepted_default_and_reject_unbounded():
    valid = mod.validate_config({})
    assert valid['methods'][:2] == ['none', 'oracle']
    for bad in [
        {'worlds': 500},
        {'seeds': [1, 1]},
        {'methods': ['none', 'bm25']},
        {'model': 'attacker/arbitrary'},
        {'worlds': 24, 'seeds': [1, 2, 3], 'methods': list(mod.METHOD_ALLOWLIST)},
    ]:
        with pytest.raises(ValueError):
            mod.validate_config(bad)


def test_render_script_and_metadata(tmp_path):
    out = tmp_path / 'build'
    sha = '1' * 40
    mod.render_kernel(mod.validate_config({}), sha, 'user/evomem-hidden-rules', out)
    kernel = (out / 'run_hidden_rules.py').read_text()
    assert sha in kernel
    assert '__KAGGLE_CONFIG_JSON__' not in kernel
    metadata = json.loads((out / 'kernel-metadata.json').read_text())
    assert metadata['is_private'] and metadata['enable_gpu']
    assert metadata['id'] == 'user/evomem-hidden-rules'
    assert metadata['kernel_type'] == 'script'


def test_reject_bad_sha_and_kernel(tmp_path):
    with pytest.raises(ValueError):
        mod.render_kernel({}, 'main', 'user/name', tmp_path)
    with pytest.raises(ValueError):
        mod.render_kernel({}, '1' * 40, 'not-a-slug', tmp_path)
