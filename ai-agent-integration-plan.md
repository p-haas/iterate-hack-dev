# AI Agent Integration Plan

## Overview

With the backend and frontend now sharing a stable contract, the next milestone is to introduce the autonomous AI agent that generates insights (dataset understanding, issues, fix suggestions) instead of our current heuristics. This guide enumerates the steps required to integrate the agent safely, with actionable deliverables for each phase.

## 1. Define Agent Contracts

- Enumerate the exact prompts/inputs the agent needs (dataset metadata, sample rows, user instructions).
- Specify the structured JSON schemas the agent must return for:
  - Dataset understanding (summary, observations, columns).
  - Analysis issues (id, type, severity, affected columns/rows, suggested actions, temporal patterns).
  - Smart-fix follow-up questions.
- Document the schema in `docs/agent-contracts.md` so backend and agent code share the same contract.

**Deliverable/Test**: Contract doc reviewed by backend + prompt engineers; mock JSON validated against TypeScript types.

## 2. Agent Execution Sandbox

- Decide how the agent runs worker-side (e.g., within FastAPI using LangChain, or via an async job queue like Celery/RQ).
- Ensure long-running analysis doesn’t block HTTP requests: offload to background tasks or streaming endpoints that read from job logs.
- Add guardrails (timeouts, retry limits, logging) to keep the agent from looping or emitting invalid data.

**Deliverable/Test**: Prototype endpoint (`/agent/test`) that runs the agent on a small CSV and returns structured JSON in under N seconds.

## 3. Replace Dataset Understanding Heuristics

- Inside `app/main.py`, swap `_load_dataframe`-based summarization with an agent call that produces `DatasetUnderstanding`.
- Cache the agent output (e.g., `data/{id}/understanding.json`) so reloading the page doesn’t retrigger the agent unless forced.
- Provide a fallback (old heuristics) if the agent errors, ensuring the frontend still receives data.

**Deliverable/Test**: Upload + Understand flow shows agent-generated descriptions; backend logs confirm cached response reuse.

## 4. Agent-Driven Analysis & Issues

- Update `_run_dataset_analysis` to invoke the agent with dataset stats + context, letting it return the `issues` array (quick + smart).
- Parse the agent’s JSON, validate against the schema, and store in `analysis.json`.
- Map any free-form textual suggestions into the existing UI fields (e.g., `suggestedAction`).

**Deliverable/Test**: Running analysis produces agent-authored issue cards with diverse severities; invalid agent output triggers a fallback + log warning.

## 5. Agent-Assisted Smart Fix Dialog

- Instead of hardcoded questions & options, fetch agent-provided prompts or follow-ups per issue.
- Allow the backend to relay user answers back to the agent for iterative reasoning (e.g., kick off a secondary agent run using the response).
- Persist the dialogue history for auditing.

**Deliverable/Test**: Opening the Smart Fix dialog pulls agent-specific questions/examples; submitting an answer spawns a new agent action recorded in logs.

## 6. Streaming & Observability Enhancements

- Stream live agent thoughts/logs to `/analysis/stream` so the frontend shows meaningful progress (e.g., “Agent inspecting missing values…”).
- Emit structured telemetry (duration, tokens used, error rates) for monitoring.
- Update `scripts/smoke_test.py` to include an agent-backed run (with a very small fixture dataset).

**Deliverable/Test**: Smoke test completes using the agent path; Grafana/Logs show per-run metrics.

## 7. Safety, Cost, and Rate Control

- Implement safeguards: maximum dataset size per agent run, rate limits per user, and cost tracking (Anthropic billing).
- Add a kill switch / feature flag to disable the agent if issues arise, falling back to heuristics.
- Document incident response steps (how to inspect cached agent outputs, replay a run, etc.).

**Deliverable/Test**: Feature flag toggles between agent and heuristic modes; cost dashboard reflects each run.

## 8. Frontend UX Updates

- Display agent attribution (e.g., “Insights powered by Claude”) and confidence indicators.
- Show when outputs are cached vs newly generated, and provide a “Re-run with latest agent” button.
- Surface agent explanations in the UI (tooltips, expandable sections) for transparency.

**Deliverable/Test**: UX review confirms new messaging; manual test toggling “Re-run” demonstrates agent refresh.

## 9. QA & Rollout

- Create automated tests (unit + integration) asserting that agent JSON maps cleanly to frontend types.
- Run end-to-end scenarios (upload → context → analysis → smart fix) with multiple sample datasets.
- Gradually roll out via a beta flag before exposing to all users.

**Deliverable/Test**: QA checklist signed off; beta users complete the flow without regressions; feature flag flipped to “on” post-launch.
