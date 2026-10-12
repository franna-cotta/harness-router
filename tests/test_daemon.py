import asyncio

import pytest

import harness_router.daemon as daemon


@pytest.mark.asyncio
async def test_provider_for_backend_creates_provider_once_under_concurrency(
    monkeypatch,
) -> None:
    created: list[str] = []

    def fake_provider_from_backend(*, backend: str, timeout_seconds: float = 5.0):
        del timeout_seconds
        created.append(backend)
        return object()

    monkeypatch.setattr(daemon, "provider_from_backend", fake_provider_from_backend)

    providers: dict[str, object] = {}
    lock = asyncio.Lock()

    async def get_provider() -> object:
        return await daemon._provider_for_backend(
            backend="azure_foundry",
            providers=providers,
            provider_lock=lock,
        )

    first, second = await asyncio.gather(get_provider(), get_provider())

    assert first is second
    assert created == ["azure_foundry"]
