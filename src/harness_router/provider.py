from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Protocol

import httpx

from .config import AzureFoundryConfig, OpenRouterConfig
from .errors import (
    InvalidProviderResponse,
    ProviderError,
    ProviderTimeoutError,
    RouterConfigurationError,
)
from .models import ChoiceDecision


_REDACTED = "[REDACTED]"


class DecisionProvider(Protocol):
    @property
    def requests_made(self) -> int:
        """Number of OpenRouter HTTP requests attempted by this provider instance."""
        return self._requests_made

    async def choose(
        self,
        *,
        state: str,
        instructions: str,
        criteria: Mapping[str, str],
    ) -> ChoiceDecision: ...


class OpenRouterJevProvider:
    """Jev provider backed by OpenRouter's Decisions API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "typesafe/jev-1.13",
        base_url: str = "https://openrouter.ai/api/alpha/decisions",
        timeout_seconds: float = 5.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty")
        self._api_key = api_key
        self._model = model
        self._requests_made = 0
        self._base_url = base_url
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=timeout_seconds,
            limits=httpx.Limits(
                max_connections=16,
                max_keepalive_connections=8,
                keepalive_expiry=30.0,
            ),
        )
        self._provider_name = "OpenRouter Jev"
        self._api_key_header = "Authorization"
        self._api_key_prefix = "Bearer "
        self._api_key_label = "OpenRouter API key"

    @property
    def requests_made(self) -> int:
        """Number of OpenRouter HTTP requests attempted by this provider."""
        return self._requests_made

    @classmethod
    def from_config(
        cls,
        config: OpenRouterConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> OpenRouterJevProvider:
        return cls(
            api_key=config.api_key(),
            model=config.model,
            base_url=config.url,
            timeout_seconds=config.timeout_seconds,
            client=client,
        )

    async def choose(
        self,
        *,
        state: str,
        instructions: str,
        criteria: Mapping[str, str],
    ) -> ChoiceDecision:
        if len(criteria) < 2:
            raise ValueError("Jev choice requires at least two criteria")

        state_payload: object = state
        if state.lstrip().startswith("{"):
            try:
                parsed_state = json.loads(state)
            except ValueError:
                pass
            else:
                if isinstance(parsed_state, dict):
                    state_payload = parsed_state

        payload = {
            "model": self._model,
            "state": state_payload,
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": instructions,
                    "criteria": dict(criteria),
                }
            },
        }
        self._assert_api_key_not_in_payload(payload)

        try:
            # Count immediately before the network request so failures/timeouts are visible.
            self._requests_made += 1
            response = await self._client.post(
                self._base_url,
                headers={
                    self._api_key_header: f"{self._api_key_prefix}{self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
        except httpx.TimeoutException:
            # Do not retain the httpx exception as __cause__: it owns the request object,
            # including request headers with credentials.
            raise ProviderTimeoutError(f"{self._provider_name} request timed out") from None
        except httpx.HTTPStatusError as exc:
            body = self._redact_api_key(exc.response.text[:500])
            status_code = exc.response.status_code
            raise ProviderError(
                f"{self._provider_name} returned HTTP {status_code}: {body}"
            ) from None
        except httpx.HTTPError as exc:
            message = self._redact_api_key(str(exc))
            raise ProviderError(f"{self._provider_name} request failed: {message}") from None

        try:
            data = response.json()
        except ValueError as exc:
            raise InvalidProviderResponse(f"{self._provider_name} response was not valid JSON") from exc

        answer = data.get("answers", {}).get("route")
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise InvalidProviderResponse("missing choice answer at answers.route")

        choice = answer.get("choice")
        probabilities_raw = answer.get("probabilities", {})
        confidence_raw = answer.get("confidence", 0.0)

        if not isinstance(choice, str):
            raise InvalidProviderResponse("choice answer did not contain a string choice")
        if choice not in criteria:
            raise InvalidProviderResponse(f"provider selected unknown criterion: {choice}")
        if not isinstance(probabilities_raw, dict):
            raise InvalidProviderResponse("choice probabilities must be an object")
        if not isinstance(confidence_raw, (int, float)):
            raise InvalidProviderResponse("choice confidence must be numeric")

        probabilities: dict[str, float] = {}
        for key, value in probabilities_raw.items():
            if not isinstance(key, str) or not isinstance(value, (int, float)):
                raise InvalidProviderResponse("choice probabilities must map strings to numbers")
            probabilities[key] = float(value)

        confidence = float(confidence_raw)
        if not 0.0 <= confidence <= 1.0:
            raise InvalidProviderResponse("choice confidence must be between 0 and 1")

        return ChoiceDecision(
            choice=choice,
            probabilities=probabilities,
            confidence=confidence,
        )

    def _assert_api_key_not_in_payload(self, payload: object) -> None:
        serialized = json.dumps(payload, ensure_ascii=False, default=str)
        if self._api_key and self._api_key in serialized:
            raise ProviderError(
                f"refusing to send a routing payload containing the {self._api_key_label}"
            )

    def _redact_api_key(self, value: str) -> str:
        if not self._api_key:
            return value
        return value.replace(self._api_key, _REDACTED)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> OpenRouterJevProvider:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()


class AzureFoundryDecisionProvider(OpenRouterJevProvider):
    """Decision provider backed by Azure Foundry's Decision API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "Decision-1",
        base_url: str = "https://aivoice-foundry-sweden.services.ai.azure.com/models/providers/microsoft/v1/systemone",
        timeout_seconds: float = 5.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            client=client,
        )
        self._provider_name = "Azure Foundry Decision"
        self._api_key_header = "api-key"
        self._api_key_prefix = ""
        self._api_key_label = "Azure API key"

    @classmethod
    def from_config(
        cls,
        config: AzureFoundryConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> AzureFoundryDecisionProvider:
        return cls(
            api_key=config.api_key(),
            model=config.model_name(),
            base_url=config.request_url(),
            timeout_seconds=config.timeout_seconds,
            client=client,
        )


def provider_from_backend(*, backend: str, timeout_seconds: float = 5.0) -> OpenRouterJevProvider:
    if backend == "openrouter":
        return OpenRouterJevProvider.from_config(OpenRouterConfig(timeout_seconds=timeout_seconds))
    if backend == "azure_foundry":
        return AzureFoundryDecisionProvider.from_config(AzureFoundryConfig(timeout_seconds=timeout_seconds))
    raise RouterConfigurationError(f"unsupported provider backend: {backend}")


def missing_key_reason_for_backend(backend: str) -> str:
    if backend == "azure_foundry":
        return "missing_azure_openai_api_key"
    if backend == "openrouter":
        return "missing_openrouter_api_key"
    return "invalid_provider_backend"
