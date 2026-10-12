import json

import pytest

import harness_router.cli as cli
from harness_router import ChoiceDecision
from harness_router.cli import build_parser


def test_cli_parser_supports_route() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "route",
            "--goal",
            "Fix tests",
            "--tools-json",
            '[{"name":"read_file","description":"Read a file"}]',
        ]
    )
    assert args.command == "route"
    assert args.goal == "Fix tests"
    assert args.hierarchical_threshold == 24
    assert args.verbose is False
    assert args.no_cache is False


def test_cli_without_subcommand_is_valid() -> None:
    parser = build_parser()
    args = parser.parse_args([])
    assert args.command is None


def test_cli_no_cache_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "route",
            "--goal",
            "Fix tests",
            "--tools-json",
            '[{"name":"read_file","description":"Read a file"},{"name":"edit_file","description":"Edit a file"}]',
            "--no-cache",
        ]
    )
    assert args.no_cache is True


def test_cli_payload_respects_provider_backend_env(monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_ROUTER_PROVIDER", "azure_foundry")
    parser = build_parser()
    args = parser.parse_args(
        [
            "route",
            "--goal",
            "Fix tests",
            "--tools-json",
            '[{"name":"read_file","description":"Read a file"}]',
        ]
    )
    payload = cli._request_payload(args)
    assert payload["provider_backend"] == "azure_foundry"


def test_cli_route_invalid_provider_backend_returns_clean_fallback(
    monkeypatch, capsys
) -> None:
    monkeypatch.setenv("HARNESS_ROUTER_PROVIDER", "azure")

    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "route",
                "--goal",
                "Fix tests",
                "--tools-json",
                '[{"name":"read_file","description":"Read a file"}]',
                "--no-daemon",
            ]
        )

    output = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "Traceback" not in output.err

    payload = json.loads(output.out)
    assert payload["fallback"] is True
    assert payload["fallback_reason"] == "invalid_provider_backend"
    assert payload["provider"] == "azure"


@pytest.mark.asyncio
async def test_cli_run_direct_uses_selected_provider_backend(monkeypatch) -> None:
    class FakeProvider:
        def __init__(self) -> None:
            self.requests_made = 0

        async def choose(self, **_: object) -> ChoiceDecision:
            self.requests_made += 1
            return ChoiceDecision(
                choice="read_file",
                probabilities={"read_file": 0.97, "__fallback__": 0.03},
                confidence=0.97,
            )

        async def aclose(self) -> None:
            return None

    provider = FakeProvider()

    def fake_provider_from_backend(*, backend: str, timeout_seconds: float = 5.0):
        del timeout_seconds
        assert backend == "azure_foundry"
        return provider

    monkeypatch.setattr(cli, "provider_from_backend", fake_provider_from_backend)

    payload = {
        "goal": "Fix tests",
        "observation": None,
        "last_action": None,
        "tools": [{"name": "read_file", "description": "Read a file"}],
        "mode": "hybrid",
        "direct_threshold": 0.85,
        "fallback_threshold": 0.60,
        "hierarchical_threshold": 24,
        "route_cache_size": 128,
        "obvious_cache_size": 256,
        "provider_backend": "azure_foundry",
    }

    response, status = await cli._run_direct(payload)

    assert status == 0
    assert response["provider"] == "azure_foundry"
    assert response["provider_requests"] == 1
