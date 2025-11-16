# Backend & Frontend Integration Plan

## Overview

This document captures the concrete steps required to mature the FastAPI backend so it fully supports the DataClean AI frontend. Each step ends with an actionable, testable output to validate progress.

## 1. Align API Contract

- Inventory every frontend call defined in `frontend/src/lib/apiClient.ts`.
- Map each call to a FastAPI route, noting request payloads, query params, and expected responses that match `frontend/src/types/dataset.ts`.
- Produce `docs/api-contract.md` (or similar) that both teams review/sign off.

**Deliverable/Test**: `api-contract.md` document created, covering upload, understanding, context, streaming, analysis, and apply endpoints, with sample payloads that the frontend can consume.

## 2. Dataset Upload Service ✅

- Extend `/upload_and_analyze` or create `/datasets` POST so uploads persist immediately (no blocking analysis).
- Save the raw file under `/data/{datasetId}/raw.{ext}` and capture metadata (size, inferred delimiter, mime type).
- Return `{ datasetId, fileName }` so the UI can advance to the understanding step.

**Deliverable/Test**: `curl -F file=@sample.csv http://localhost:8000/datasets` succeeds, stores the file, and returns JSON consumed by the frontend without further changes. (✅ Verified on 2025-11-16.)

## 3. Dataset Understanding Endpoint ✅

- Implement `/datasets/{datasetId}/understanding` that loads the persisted `data/{datasetId}/raw.*` file and the accompanying `metadata.json` written in Step 2.
- Compute summary stats (row/column counts, sample values, observations) and return `Column` entries that match `frontend/src/types/dataset.ts`.
- Reuse pandas profiling logic or lightweight heuristics, but keep the response deterministic for a given dataset.

**Deliverable/Test**: Opening the “Understand” step triggers a real backend call and renders the returned schema in the UI. Automated test: GET on the endpoint returns stable fixture data for a known CSV. (✅ Verified via `curl /datasets/<id>/understanding` on 2025-11-16.)

## 4. Context Persistence ✅

- Provide `/datasets/{datasetId}/context` (POST/PUT) to store user instructions plus optional column overrides, using the dataset id generated at upload time.
- Persist context next to `metadata.json` (e.g., `data/{datasetId}/context.json`) so later analysis has access to both schema metadata and human hints.
- Optionally add GET to reload saved context when users revisit the step or refresh the browser.

**Deliverable/Test**: Saving context from the UI results in a stored record/file and reloading the page shows the same text (either via direct backend GET or manual inspection during development). (✅ Verified via POST/GET `/datasets/<id>/context` curl on 2025-11-16.)

## 5. Analysis Streaming & Final Results ✅

- Split analysis into streaming logs and final summary endpoints:
  - `/datasets/{datasetId}/analysis/stream` (SSE/chunked) reads from the stored dataset path + context file to emit `StreamMessage` updates consumed by the frontend log.
  - `/datasets/{datasetId}/analysis` (POST) loads `raw.*`, context, and metadata, runs the LLM/script pipeline, writes results to `data/{datasetId}/analysis.json`, and returns an `AnalysisResult` compatible with the UI.
- Guarantee issue objects include severity, category (`quick_fixes`/`smart_fixes`), affected rows/columns, and temporal metadata so later steps (apply/smart fix) have the necessary ids.

**Deliverable/Test**: Triggering analysis from the UI shows live log entries and eventually populates issue cards with backend-produced data; integration test covers both endpoints. (✅ Verified via curl SSE stream and POST analysis on 2025-11-16.)

## 6. Apply Fixes Workflow ✅

- Add `/datasets/{datasetId}/apply` accepting `{ issueIds: [...] }` to execute or queue deterministic quick fixes against the stored dataset from Step 2.
- Log operations per issue and update `data/{datasetId}` artifacts (new cleaned file versions, refreshed `analysis.json`, status flags) so subsequent GETs reflect applied fixes.
- Return summary `{ appliedCount, details }` for toast feedback.

**Deliverable/Test**: Selecting issues in the UI and clicking “Apply fixes” results in backend logs plus a success response; repeated GET `/analysis` reflects updated issue states if applicable. (✅ Verified via curl POST `/datasets/<id>/apply` on 2025-11-16.)

## 7. Smart Fix Responses ✅

- Implement `/datasets/{datasetId}/smart-fix` (or extend `/apply`) to capture smart-fix answers (`intentional`, `standardize`, `keep`, `custom`) tied to issues that were created in Step 5.
- Persist responses within `data/{datasetId}` (e.g., `smart_fix_responses.json`) and use them to drive follow-up remediation or human review flows.
- Validate inputs so the frontend receives meaningful errors when responses are missing or malformed, and consider reflecting stored answers when re-fetching analysis data.

**Deliverable/Test**: Submitting a smart fix answer from the dialog leads to a stored entry and optional status update; backend logs confirm receipt. (✅ Verified via POST `/datasets/<id>/smart-fix` curl on 2025-11-16.)

## 8. Health & Observability ✅

- Keep `/health` lightweight and add metrics (last analysis timestamp, queued jobs) for deployment monitoring.
- Add structured logging around each route (dataset id, duration, outcomes) to simplify debugging.
- Consider a nightly smoke script hitting health → upload → understanding → context → stream → analysis → apply to guard against regressions.

**Deliverable/Test**: Automated smoke test script exits successfully and CI/CD wiring reports failures when any step regresses. (✅ Health endpoint extended and `scripts/smoke_test.py` added on 2025-11-16.)
