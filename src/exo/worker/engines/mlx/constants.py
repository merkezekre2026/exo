"""Runtime and model-generation constants for the MLX engine."""

import os

CPU_PREFILL_STEP_SIZE = 4096
CUDA_PREFILL_STEP_SIZE = 8192


def get_prefill_step_size() -> int:
    """Return the prompt chunk size used by local and batched prefill.

    CUDA uses a larger default chunk to improve prompt throughput, while the
    smaller CPU default keeps peak working-set growth bounded. The
    ``EXO_PREFILL_STEP_SIZE`` override is an escape hatch for models or GPUs
    whose memory/performance profile differs from the defaults. Invalid
    overrides are ignored so a malformed environment cannot prevent a worker
    from starting.
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
        import mlx.core as mx  # pyright: ignore[reportMissingModuleSource]

        if mx.cuda.is_available():
            return CUDA_PREFILL_STEP_SIZE
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
