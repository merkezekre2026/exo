import pytest

from exo.shared.types.backends import Backend
from exo.utils.info_gatherer import info_gatherer


@pytest.mark.anyio
async def test_gather_reports_mlx_cuda_when_available(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(info_gatherer, "IS_DARWIN", False)
    monkeypatch.setattr(info_gatherer, "_has_mlx_cuda", lambda: True)

    result = await info_gatherer.NodeBackends.gather()

    assert result.backends == [Backend.MlxCpu, Backend.MlxCuda, Backend.Vllm]


@pytest.mark.anyio
async def test_gather_does_not_report_mlx_cuda_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(info_gatherer, "IS_DARWIN", False)
    monkeypatch.setattr(info_gatherer, "_has_mlx_cuda", lambda: False)

    result = await info_gatherer.NodeBackends.gather()

    assert result.backends == [Backend.MlxCpu]
