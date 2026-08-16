import pytest

from exo.worker.engines.mlx.constants import (
    CPU_PREFILL_STEP_SIZE,
    get_prefill_step_size,
)


def test_prefill_step_size_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EXO_PREFILL_STEP_SIZE", "16384")

    assert get_prefill_step_size() == 16384


def test_invalid_prefill_step_size_uses_safe_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EXO_PREFILL_STEP_SIZE", "not-a-number")

    assert get_prefill_step_size() == CPU_PREFILL_STEP_SIZE
