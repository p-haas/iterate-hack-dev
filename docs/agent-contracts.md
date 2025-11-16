# Agent Contracts

## Overview

Defines the structured inputs/outputs shared between the FastAPI backend and the autonomous AI agent. All agent responses must conform to these schemas so the UI can render them safely.

## 1. Dataset Understanding Schema

**Agent Input**
- `dataset_id`: string
- `file_name`: string
- `row_count`: integer
- `column_count`: integer
- `sample_rows`: array of records (max 5 rows) with column → value mapping
- `column_summaries`: array of `{ name, inferred_type, sample_values }`
- `user_instructions`: string (may be empty)

**Agent Output** (`DatasetUnderstanding`)
```json
{
  "summary": {
    "name": "inventory_levels.csv",
    "description": "SaaS subscription dataset...",
    "rowCount": 1247,
    "columnCount": 6,
    "observations": ["Contains subscription data", "Missing values in churn_date"]
  },
  "columns": [
    {
      "name": "customer_id",
      "dataType": "string",
      "description": "Unique identifier...",
      "sampleValues": ["CUST01", "CUST02", "CUST03"]
    }
  ]
}
```
Constraints:
- `dataType` ∈ {string,numeric,date,categorical,boolean}.
- `observations` ≤ 5 items, human-readable sentences.

## 2. Analysis Issues Schema

**Agent Input**
- `dataset_id`
- `dataset_understanding` (from previous step)
- `user_instructions`
- Optional `previous_issues` (if rerunning)

**Agent Output** (`AnalysisResult`)
```json
{
  "issues": [
    {
      "id": "{dataset_id}_missing_Product",
      "type": "missing_values",
      "severity": "high",
      "description": "Product column missing 93%...",
      "affectedColumns": ["Product"],
      "suggestedAction": "Fill via forward-fill or drop rows",
      "category": "quick_fixes",
      "affectedRows": 40119,
      "temporalPattern": null
    },
    {
      "id": "{dataset_id}_context_alignment",
      "type": "discount_context",
      "severity": "medium",
      "description": "Needs business clarification",
      "affectedColumns": ["discount_amount"],
      "suggestedAction": "Ask user how to interpret discounts",
      "category": "smart_fixes",
      "affectedRows": null,
      "temporalPattern": "Starting April 2024"
    }
  ],
  "summary": "Analysis complete. Found N issues.",
  "completedAt": "ISO timestamp"
}
```
Constraints:
- `type` values map to frontend enum (`missing_values`, `duplicates`, `supplier_variations`, etc.).
- `severity` ∈ {low,medium,high}.
- `category` ∈ {quick_fixes, smart_fixes}.
- Smart-fix issues must include a follow-up question in `smart_fix_prompt` (string) so the dialog can render context.

## 3. Smart Fix Follow-up Schema

**Agent Input**
- `issue_id`
- `issue_context`: the issue object
- `user_instructions`
- `smart_fix_history`: array of previous Q/A pairs

**Agent Output**
```json
{
  "prompt": "Did the category change intentionally?",
  "options": [
    { "key": "intentional", "label": "Yes, this was intentional" },
    { "key": "standardize", "label": "No, standardize the values" }
  ],
  "examples": "Product Exputex moved from OTC to OTC:Cold&Flu",
  "onResponse": {
    "action": "queue_remediation",
    "notes": "Run standardization script if user selects 'standardize'"
  }
}
```
Constraints:
- Options array length 2-4; include `custom` flag when expecting free-form input.
- `onResponse.action` ∈ {queue_remediation, escalate_human, collect_more_info}.

## 4. Error Handling

- Agent must return `{ "error": { "type": "validation", "message": "..." } }` when it cannot comply.
- Backend will log errors and fall back to heuristics.

## 5. Validation Checklist

- All JSON must be UTF-8, no code fences.
- Numeric fields use numbers, not strings.
- IDs must be deterministic (`{dataset_id}_{slug}`) so the frontend can correlate.

