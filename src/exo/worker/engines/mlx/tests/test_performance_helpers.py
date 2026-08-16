import pytest

from exo.worker.engines.mlx.constants import (
    CPU_PREFILL_STEP_SIZE,
    CUDA_DEFAULT_PREFILL_STEP_SIZE,
    CUDA_HIGH_VRAM_PREFILL_STEP_SIZE,
    CUDA_LOW_VRAM_PREFILL_STEP_SIZE,
    get_prefill_step_size,
    get_prefill_step_size_for_memory,
)


@pytest.mark.parametrize(
    ("memory_size", "expected"),
    [
        (4 * 1024**3, CUDA_LOW_VRAM_PREFILL_STEP_SIZE),
        (8 * 1024**3, CUDA_DEFAULT_PREFILL_STEP_SIZE),
        (16 * 1024**3, CUDA_HIGH_VRAM_PREFILL_STEP_SIZE),
        (None, CUDA_DEFAULT_PREFILL_STEP_SIZE),
    ],
)
def test_prefill_step_size_uses_memory_profile(memory_size: int | None, expected: int):
    assert get_prefill_step_size_for_memory(memory_size) == expected


def test_prefill_step_size_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EXO_PREFILL_STEP_SIZE", "16384")

    assert get_prefill_step_size() == 16384


def test_invalid_prefill_step_size_uses_safe_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EXO_PREFILL_STEP_SIZE", "not-a-number")

    assert get_prefill_step_size() == CPU_PREFILL_STEP_SIZE
