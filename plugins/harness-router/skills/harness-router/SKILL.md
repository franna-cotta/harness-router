---
name: harness-router
description: Automatically route ambiguous tool choices with harness-router hooks; manual `route` and `route_mcts` calls remain available.
---

# Harness Router (Plugin Wrapper)

This plugin installs lifecycle hooks so routing runs automatically.

Use routing only when multiple plausible tools remain after simple pruning and a wrong pick would cost extra turns.

## Preferred usage

- Use MCP tool `route` for ambiguous closed choices.
- Use MCP tool `route_mcts` only with a side-effect-free simulated graph.
- If routing returns fallback, continue with normal planner behavior.

## Azure Foundry defaults

Set these env vars in your shell/session:

- `HARNESS_ROUTER_PROVIDER=azure_foundry`
- `AZURE_ENDPOINT=https://aivoice-foundry-sweden.services.ai.azure.com`
- `AZURE_DECISION_MODEL=Decision-1`
- `AZURE_FOUNDRY_API_KEY=<your key>`

Do not print or paste secret values in tool inputs.
