"""Runtime and model-generation constants for the MLX engine."""

import os

CPU_PREFILL_STEP_SIZE = 4096
CUDA_LOW_VRAM_PREFILL_STEP_SIZE = 2048
CUDA_DEFAULT_PREFILL_STEP_SIZE = 4096
CUDA_HIGH_VRAM_PREFILL_STEP_SIZE = 8192
CUDA_LOW_VRAM_LIMIT_BYTES = 6 * 1024**3
CUDA_HIGH_VRAM_LIMIT_BYTES = 12 * 1024**3


def get_prefill_step_size_for_memory(memory_size_bytes: int | None) -> int:
    """Select a conservative prompt chunk for the detected CUDA memory size."""
    if memory_size_bytes is None:
        # Unknown CUDA devices use the middle profile rather than risking the
        # larger allocation on an entry-level card.
        return CUDA_DEFAULT_PREFILL_STEP_SIZE
    if memory_size_bytes <= CUDA_LOW_VRAM_LIMIT_BYTES:
        return CUDA_LOW_VRAM_PREFILL_STEP_SIZE
    if memory_size_bytes <= CUDA_HIGH_VRAM_LIMIT_BYTES:
        return CUDA_DEFAULT_PREFILL_STEP_SIZE
    return CUDA_HIGH_VRAM_PREFILL_STEP_SIZE


def _cuda_memory_size() -> int | None:
    """Return the active CUDA device memory size when MLX exposes it."""
    try:
        import mlx.core as mx

        if not mx.cuda.is_available():
            return None
        raw_memory_size = mx.device_info().get("memory_size")
        if isinstance(raw_memory_size, int):
            return raw_memory_size
        if isinstance(raw_memory_size, str):
            return int(raw_memory_size)
    except (AttributeError, ImportError, TypeError, ValueError, RuntimeError):
        pass
    return None


def get_prefill_step_size() -> int:
    """Return the prompt chunk size used by local and batched prefill.

    CUDA uses a memory-aware chunk: 2048 tokens for devices up to 6 GiB,
    4096 tokens up to 12 GiB, and 8192 tokens above that. The smaller CPU
    default keeps peak working-set growth bounded. ``EXO_PREFILL_STEP_SIZE``
    overrides the automatic profile for model- or workload-specific tuning.
    Invalid overrides are ignored so a malformed environment cannot prevent a
    worker from starting.
    """
    override = os.environ.get("EXO_PREFILL_STEP_SIZE")
    if override is not None:
        try:
            value = int(override)
        except ValueError:
            value = 0
        if value > 0:
            return value

    try:
        import mlx.core as mx

        if mx.cuda.is_available():
            return get_prefill_step_size_for_memory(_cuda_memory_size())
    except Exception:
        # A missing/incompatible driver should fall back to the CPU chunk size.
        pass

    return CPU_PREFILL_STEP_SIZE


KV_GROUP_SIZE: int | None = 32
KV_BITS: int | None = None
ATTENTION_KV_BITS: int | None = 4
MAX_TOKENS: int = 32168
MAX_KV_SIZE: int | None = 3200
KEEP_KV_SIZE: int | None = 1600
QUANTIZE_MODEL_MODE: str | None = "affine"
CACHE_GROUP_SIZE: int = 64
KV_CACHE_BITS: int | None = None


DEFAULT_TOP_LOGPROBS: int = 5

# TODO: We should really make this opt-in, but Kimi requires trust_remote_code=True
TRUST_REMOTE_CODE: bool = True
