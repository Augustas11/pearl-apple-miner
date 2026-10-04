# SPDX-License-Identifier: Apache-2.0
# Derives from Pearl fp8-scheme (ISC), pinned at f696760b259500ecb608469ea3953aeabbe78948.
"""Admission configuration must agree across Python and the native library."""
import os

import pytest

from pmk_miner.v4_admission import (
    V4AdmissionError,
    V4_G3_ADMISSION_ENV,
    configured_v4_admission,
    validate_v4_g3_admission_file,
)


def test_admission_config_exports_relative_path_and_restores_on_error(tmp_path, monkeypatch):
    monkeypatch.delenv(V4_G3_ADMISSION_ENV, raising=False)
    config = {'v4': {'admission_file': 'admission.json'}}
    with pytest.raises(RuntimeError, match='downstream failure'):
        with configured_v4_admission(config, tmp_path / 'miner.toml'):
            assert os.environ[V4_G3_ADMISSION_ENV] == str(tmp_path / 'admission.json')
            raise RuntimeError('downstream failure')
    assert V4_G3_ADMISSION_ENV not in os.environ


@pytest.mark.parametrize('override', ['', '/missing/operator-override.json'])
def test_explicit_environment_never_falls_back_to_config(tmp_path, monkeypatch, override):
    monkeypatch.setenv(V4_G3_ADMISSION_ENV, override)
    with configured_v4_admission({'v4': {'admission_file': 'other.json'}}, tmp_path / 'miner.toml'):
        assert os.environ[V4_G3_ADMISSION_ENV] == override
        with pytest.raises(V4AdmissionError):
            validate_v4_g3_admission_file()
    assert os.environ[V4_G3_ADMISSION_ENV] == override


@pytest.mark.parametrize('section', ['invalid', {'admission_file': ''}, {'admission_file': 4}])
def test_invalid_admission_config_fails_closed(tmp_path, monkeypatch, section):
    monkeypatch.delenv(V4_G3_ADMISSION_ENV, raising=False)
    with pytest.raises(V4AdmissionError):
        with configured_v4_admission({'v4': section}, tmp_path / 'miner.toml'):
            pytest.fail('invalid admission config accepted')
    assert V4_G3_ADMISSION_ENV not in os.environ


def test_missing_admission_does_not_affect_v3_configuration(tmp_path, monkeypatch):
    monkeypatch.delenv(V4_G3_ADMISSION_ENV, raising=False)
    with configured_v4_admission({}, tmp_path / 'miner.toml'):
        assert V4_G3_ADMISSION_ENV not in os.environ
        # Rejection happens when a v4 job requires admission, not at v3 startup.
        with pytest.raises(V4AdmissionError, match='missing v4 G3 admission file'):
            validate_v4_g3_admission_file()
