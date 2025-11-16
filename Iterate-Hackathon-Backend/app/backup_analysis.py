# app/backup_analysis.py
"""
Rule-based backup analysis used when the code agent is unavailable.

It mirrors the legacy detect_correct.py script but returns JSON payloads that
match the FastAPI IssueResponse / AnalysisResultResponse models so the
frontend can consume them without transformation.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import logging

import pandas as pd


logger = logging.getLogger(__name__)

IssuePayload = Dict[str, Any]


def _issue_id(dataset_id: str, slug: str) -> str:
    return f"{dataset_id}_{slug}"


def _severity_from_ratio(ratio: float) -> str:
    if ratio >= 0.25:
        return "high"
    if ratio >= 0.1:
        return "medium"
    return "low"


def _investigation(code: str, output: Any) -> Dict[str, Any]:
    return {
        "code": code,
        "success": True,
        "output": output,
    }


def _missing_product_names(dataset_id: str, df: pd.DataFrame) -> List[IssuePayload]:
    if "Product" not in df.columns:
        return []

    series = df["Product"].astype(str)
    missing_mask = df["Product"].isna() | series.str.strip().isin(["", "NULL", "null"])
    missing_indices = df.index[missing_mask].tolist()
    missing_count = len(missing_indices)
    if missing_count == 0:
        return []

    ratio = missing_count / max(len(df), 1)
    sample_rows = missing_indices[:5]

    issue = {
        "id": _issue_id(dataset_id, "missing_product"),
        "type": "missing_values",
        "severity": _severity_from_ratio(ratio),
        "description": (
            f"'Product' column contains {missing_count} missing or blank values "
            f"({ratio:.1%} of rows)."
        ),
        "affectedColumns": ["Product"],
        "suggestedAction": "Fill missing product names using the barcode dictionary or manual review.",
        "category": "quick_fixes",
        "affectedRows": missing_count,
        "temporalPattern": None,
        "investigation": _investigation(
            "df['Product'].isna() | df['Product'].astype(str).str.strip().isin(['', 'NULL', 'null'])",
            {
                "missing_rows": missing_count,
                "sample_row_indices": sample_rows,
            },
        ),
    }
    return [issue]


def _exact_duplicates(dataset_id: str, df: pd.DataFrame) -> List[IssuePayload]:
    duplicated_mask = df.duplicated(keep=False)
    duplicate_indices = df.index[duplicated_mask].tolist()
    duplicate_count = len(duplicate_indices)
    if duplicate_count == 0:
        return []

    ratio = duplicate_count / max(len(df), 1)
    issue = {
        "id": _issue_id(dataset_id, "exact_duplicates"),
        "type": "duplicates",
        "severity": "medium" if ratio < 0.15 else "high",
        "description": f"Detected {duplicate_count} exact duplicate rows. Keep the first instance and remove the rest.",
        "affectedColumns": list(df.columns),
        "suggestedAction": "Deduplicate rows based on full row equality to eliminate repeated sales.",
        "category": "quick_fixes",
        "affectedRows": duplicate_count,
        "temporalPattern": None,
        "investigation": _investigation(
            "df.duplicated(keep=False).sum()",
            {
                "duplicate_rows": duplicate_count,
                "example_indices": duplicate_indices[:10],
            },
        ),
    }
    return [issue]


def _product_whitespace(dataset_id: str, df: pd.DataFrame) -> List[IssuePayload]:
    if "Product" not in df.columns or not pd.api.types.is_object_dtype(df["Product"]):
        return []

    cleaned = df["Product"].dropna().astype(str)
    normalized = cleaned.apply(lambda value: " ".join(value.split()))
    mismatches = cleaned[cleaned != normalized]
    count = len(mismatches)
    if count == 0:
        return []

    ratio = count / max(len(df), 1)
    issue = {
        "id": _issue_id(dataset_id, "product_whitespace"),
        "type": "whitespace",
        "severity": _severity_from_ratio(ratio),
        "description": f"{count} product names contain stray whitespace or duplicated spaces.",
        "affectedColumns": ["Product"],
        "suggestedAction": "Trim whitespace and collapse multiple spaces in the 'Product' field.",
        "category": "quick_fixes",
        "affectedRows": count,
        "temporalPattern": None,
        "investigation": _investigation(
            "\"Product\" column stripping comparison",
            {
                "issue_count": count,
                "examples": mismatches.head(5).to_dict(),
            },
        ),
    }
    return [issue]


def _supplier_variations(dataset_id: str, df: pd.DataFrame) -> List[IssuePayload]:
    column = "OrderList"
    if column not in df.columns:
        return []

    variations: List[str] = []
    for val in df[column]:
        if pd.isna(val):
            continue
        text = str(val).strip()
        if not text:
            continue
        if text.lower().startswith("pharmax") and text not in {"Pharmax", "pharmax"}:
            variations.append(text)

    if not variations:
        return []

    unique_variations = sorted(set(variations))
    issue = {
        "id": _issue_id(dataset_id, "supplier_variations"),
        "type": "supplier_variations",
        "severity": "medium",
        "description": (
            f"Detected {len(variations)} rows where supplier names deviate from 'Pharmax' "
            f"(examples: {unique_variations[:5]})."
        ),
        "affectedColumns": [column],
        "suggestedAction": "Confirm whether all variations should be standardized to 'Pharmax'.",
        "category": "smart_fixes",
        "affectedRows": len(variations),
        "temporalPattern": None,
        "investigation": _investigation(
            "df['OrderList'].astype(str).str.lower().str.startswith('pharmax')",
            {
                "variation_count": len(variations),
                "examples": unique_variations,
            },
        ),
    }
    return [issue]


def _category_drift(dataset_id: str, df: pd.DataFrame) -> List[IssuePayload]:
    if "Product" not in df.columns or "Dept Fullname" not in df.columns:
        return []

    issues: List[IssuePayload] = []
    for product, group in df.groupby("Product"):
        if pd.isna(product):
            continue

        unique_depts = group["Dept Fullname"].dropna().unique()
        if len(unique_depts) <= 1:
            continue

        dept_list = sorted(map(str, unique_depts))
        issues.append(
            {
                "id": _issue_id(dataset_id, f"category_drift_{str(product).lower().replace(' ', '_')}"),
                "type": "category_drift",
                "severity": "medium",
                "description": (
                    f"Product '{product}' appears under multiple departments: {', '.join(dept_list)}. "
                    "Business confirmation is required."
                ),
                "affectedColumns": ["Dept Fullname"],
                "suggestedAction": "Select a canonical department for the product and reclassify inconsistent rows.",
                "category": "smart_fixes",
                "affectedRows": int(len(group)),
                "temporalPattern": None,
                "investigation": _investigation(
                    "df.groupby('Product')['Dept Fullname'].unique()",
                    {
                        "product": product,
                        "departments": dept_list,
                    },
                ),
            }
        )

    return issues


def _near_duplicate_rows(dataset_id: str, df: pd.DataFrame) -> List[IssuePayload]:
    if "Sale Date" not in df.columns:
        return []

    df_copy = df.copy()
    try:
        df_copy["Sale Date"] = pd.to_datetime(df_copy["Sale Date"])
    except Exception:
        return []

    cols_to_compare = [col for col in df_copy.columns if col != "Sale Date"]
    near_duplicate_indices: List[Tuple[int, int]] = []

    for _, group in df_copy.groupby(cols_to_compare, dropna=False):
        if len(group) < 2:
            continue

        sorted_group = group.sort_values("Sale Date")
        diffs = sorted_group["Sale Date"].diff().dt.total_seconds().abs()
        candidate_rows = sorted_group.index[(diffs <= 1) & (~diffs.isna())]

        for idx in candidate_rows:
            pos = list(sorted_group.index).index(idx)
            if pos == 0:
                continue
            previous_idx = list(sorted_group.index)[pos - 1]
            near_duplicate_indices.append((previous_idx, idx))

    if not near_duplicate_indices:
        return []

    flattened = sorted({idx for pair in near_duplicate_indices for idx in pair})
    issue = {
        "id": _issue_id(dataset_id, "near_duplicates"),
        "type": "near_duplicates",
        "severity": "medium",
        "description": (
            f"Detected {len(near_duplicate_indices)} near-duplicate row pairs "
            "(differences only within ±1 second on 'Sale Date')."
        ),
        "affectedColumns": cols_to_compare + ["Sale Date"],
        "suggestedAction": "Consolidate near-duplicates by keeping the earliest event or merging metrics.",
        "category": "quick_fixes",
        "affectedRows": len(flattened),
        "temporalPattern": "Occurrences clustered within one-second windows on Sale Date.",
        "investigation": _investigation(
            "group by non-date columns and compare Sale Date timestamps",
            {
                "pair_count": len(near_duplicate_indices),
                "example_pairs": near_duplicate_indices[:5],
            },
        ),
    }
    return [issue]


def run_backup_analysis(dataset_id: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run all backup detectors and return an analysis payload compatible with the API.
    """
    detectors = [
        _missing_product_names,
        _exact_duplicates,
        _product_whitespace,
        _supplier_variations,
        _category_drift,
        _near_duplicate_rows,
    ]

    issues: List[IssuePayload] = []
    for detector in detectors:
        try:
            issues.extend(detector(dataset_id, df))
        except Exception as exc:  # pragma: no cover - defensive guardrail
            logger.exception("Backup detector %s failed", detector.__name__)

    summary = (
        f"Backup analysis complete. Found {len(issues)} issues across "
        f"{len({issue['type'] for issue in issues})} categories."
    )

    return {
        "dataset_id": dataset_id,
        "issues": issues,
        "summary": summary,
        "completedAt": datetime.now(timezone.utc).isoformat(),
    }
