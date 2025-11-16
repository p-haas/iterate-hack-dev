# Frontend ↔ Backend Integration Plan

## Overview

With the FastAPI backend endpoints now live, this plan enumerates the work needed to migrate the React frontend off mocks and onto the real API. Each step ends with a clear deliverable/test so we can verify behavior incrementally.

## 1. API Client Foundation ✅

- Replace the mock `apiClient` in `frontend/src/lib/apiClient.ts` with real fetch/axios calls pointed at the FastAPI base URL (use Vite env vars like `VITE_API_BASE_URL`).
- Standardize error handling (e.g., throwing typed errors the UI can toast) and add a helper for JSON parsing + auth headers if needed.

**Deliverable/Test**: Updating `apiClient` compiles cleanly, and hitting `/health` via a temporary method confirms connectivity. (✅ Verified via console call returning `{ status: 'up', ... }` on 2025-11-16.)

## 2. Dataset Upload Wiring ✅

- Update `uploadDataset` to POST `FormData` to `/datasets`, parse `dataset_id`, metadata, and propagate errors to the Upload step.
- Ensure drag/drop UX uses the progress/loading states driven by the real response.

**Deliverable/Test**: Uploading a CSV through the UI triggers a backend call (observable in network tab) and advances to the Understand step with the returned `datasetId`. (✅ Verified via 200 OK in uvicorn logs on 2025-11-16; Understanding step currently errors because downstream fetch isn’t wired yet, which will be addressed in Step 3.)

## 3. Understanding Step API Hook ✅

- Replace mock data in `getDatasetUnderstanding` with a GET to `/datasets/{datasetId}/understanding` and surface loading/error states in `UnderstandingStep`.
- Confirm column cards render the backend-provided fields.

**Deliverable/Test**: After uploading, refreshing the page still shows the backend summary because the step fetches live data. (✅ Verified on 2025-11-16 — page now renders backend row/column counts instead of the fallback error.)

## 4. Context Persistence ✅

- Implement `saveContext` to POST to `/datasets/{datasetId}/context` and optionally add a GET in the step initialization to preload any saved context.
- Ensure toast notifications reflect actual success/failure from the backend.

**Deliverable/Test**: Entering instructions and pressing “Save & Continue” stores them; reloading the page shows the saved text via the GET endpoint. (✅ Verified on 2025-11-16.)

## 5. Analysis Streaming ✅

- Update `streamAnalysis` to consume the SSE endpoint `/datasets/{datasetId}/analysis/stream` (using `fetch` + `ReadableStream` or EventSource polyfill) and feed log entries to the Analysis Log UI.
- Handle connection errors/retries gracefully.

**Deliverable/Test**: Clicking “Run Analysis” shows the real-time messages emitted by the backend, matching the terminal `curl -N` output. (✅ Verified on 2025-11-16 with backend SSE + POST /analysis.)

## 6. Analysis Result Fetch ✅

- Replace the mock `analyzeDataset` promise with a POST to `/datasets/{datasetId}/analysis` and map the returned JSON to `AnalysisResult`.
- Ensure issue cards show the backend IDs, categories, and severities.

**Deliverable/Test**: After the POST resolves, the issue counts in the UI match the backend response captured via the browser’s network tab. (✅ Verified on 2025-11-16.)

## 7. Apply Fixes Button

- Wire the “Apply selected issues” action to POST `/datasets/{datasetId}/apply` with the selected issue IDs, updating UI state (e.g., mark issues as applied or disable checkboxes).
- Surface backend errors in toasts.

**Deliverable/Test**: Selecting issues and applying them yields the backend confirmation (`applied` vs `skipped`); subsequent clicks skip already-applied IDs.

## 8. Smart Fix Dialog Submission

- Connect the `SmartFixDialog` submission to POST `/datasets/{datasetId}/smart-fix`, passing the selected option or custom text.
- Handle success by showing a toast and optionally marking the issue as “awaiting processing.”

**Deliverable/Test**: Submitting from the dialog results in a 200 response and the dialog closes automatically; backend logs show the saved response.

## 9. Health/Smoke Regression

- Add an npm script (e.g., `pnpm run smoke`) that executes the backend’s `scripts/smoke_test.py` via a dev command or document how to run it before releases.
- Optionally show backend health status in the UI header using `/health` for quicker diagnostics.

**Deliverable/Test**: Running the documented smoke script passes, and (if implemented) the UI health indicator reflects the backend status.
