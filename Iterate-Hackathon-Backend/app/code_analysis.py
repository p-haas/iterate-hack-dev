# app/code_analysis.py
"""
Code-based dataset analysis using Claude's native code execution tool.
Replaces the custom sandbox with Anthropic's built-in secure code execution.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal, Optional

import anthropic
import pandas as pd
from pydantic import BaseModel

from .config import settings

logger = logging.getLogger(__name__)

ProgressCallback = Optional[
    Callable[[Literal["log", "progress", "issue", "complete"], str], None]
]


def _emit_progress(
    callback: ProgressCallback,
    event_type: Literal["log", "progress", "issue", "complete"],
    message: str,
) -> None:
    if callback is None:
        return
    try:
        callback(event_type, message)
    except Exception:  # pragma: no cover - defensive guardrail
        logger.debug("Progress callback failed for message: %s", message)

# Conservative constants for sizing CSV payloads sent to code execution
CSV_CHARS_PER_TOKEN = 2.0  # assume dense numerical/text CSV (tokens ~= chars / 2)
MAX_SAMPLE_TOKENS = 120_000  # keep well under Claude 200k limit (rest reserved for prompts)
MIN_SAMPLE_ROWS = 25


# ============================================================================
# Models for Code Execution Results
# ============================================================================


class CodeInvestigation(BaseModel):
    """Results from a code-based investigation."""

    code: str
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time_ms: Optional[float] = None


class EnhancedIssueModel(BaseModel):
    """Issue model with code investigation results."""

    id: str
    type: str
    severity: str
    description: str
    affectedColumns: List[str]
    suggestedAction: str
    category: str
    affectedRows: Optional[int] = None
    temporalPattern: Optional[str] = None
    investigation: Optional[CodeInvestigation] = None


# ============================================================================
# Claude Code Execution Integration
# ============================================================================





def _safe_sample_for_tokens(
    df: pd.DataFrame,
    max_rows: int,
    max_tokens: int = MAX_SAMPLE_TOKENS,
) -> tuple[pd.DataFrame, int]:
    """
    Estimate token count and reduce sample size if needed.
    
    Args:
        df: DataFrame to sample
        max_rows: Maximum rows initially requested
        max_tokens: Maximum token budget for CSV data alone
    
    Returns:
        Tuple of (possibly reduced DataFrame, actual row count)
    """
    # Take a small sample to estimate size
    sample_size = min(50, len(df))
    sample_df = df.head(sample_size)
    csv_sample = sample_df.to_csv(index=False)
    
    # Estimate tokens for full sample (conservative: 3.5 chars per token)
    chars_per_row = len(csv_sample) / sample_size if sample_size > 0 else 1000
    estimated_chars = chars_per_row * max_rows
    estimated_tokens = estimated_chars / CSV_CHARS_PER_TOKEN
    
    logger.info(f"Token estimation: {estimated_tokens:,.0f} tokens for {max_rows} rows "
                f"({chars_per_row:.0f} chars/row)")
    
    if estimated_tokens > max_tokens:
        # Reduce sample size proportionally with safety margin
        reduction_factor = max_tokens / estimated_tokens
        new_row_count = int(max_rows * reduction_factor * 0.8)  # 20% safety margin
        new_row_count = max(10, new_row_count)  # Minimum 10 rows
        logger.warning(f"Reducing sample: {max_rows} → {new_row_count} rows to stay under {max_tokens:,} token limit")
        return df, new_row_count
    
    return df, max_rows


def _fit_sample_to_prompt_budget(
    df_sample: pd.DataFrame, token_limit: int = MAX_SAMPLE_TOKENS
) -> tuple[pd.DataFrame, str, float, List[str]]:
    """
    Iteratively shrink the sample (rows/columns) until the CSV payload fits the token budget.
    
    Returns:
        (reduced_df, csv_text, estimated_tokens, dropped_columns)
    """
    if df_sample.empty:
        return df_sample, "", 0.0, []
    
    working_df = df_sample.copy()
    dropped_columns: List[str] = []
    
    while True:
        csv_text = working_df.to_csv(index=False)
        estimated_tokens = len(csv_text) / CSV_CHARS_PER_TOKEN
        
        if estimated_tokens <= token_limit:
            return working_df, csv_text, estimated_tokens, dropped_columns
        
        # Try shrinking rows first while keeping at least MIN_SAMPLE_ROWS
        if len(working_df) > MIN_SAMPLE_ROWS:
            reduction_factor = token_limit / max(estimated_tokens, 1)
            new_row_count = max(
                MIN_SAMPLE_ROWS,
                int(len(working_df) * reduction_factor * 0.85),  # leave headroom
            )
            if new_row_count >= len(working_df):
                new_row_count = max(MIN_SAMPLE_ROWS, len(working_df) - 5)
            logger.warning(
                "Reducing sample rows %s → %s to satisfy token budget (est %.0f > limit %s)",
                len(working_df),
                new_row_count,
                estimated_tokens,
                token_limit,
            )
            working_df = working_df.head(new_row_count)
            continue
        
        # Rows already minimal; progressively drop columns until we fit
        if len(working_df.columns) <= 5:
            logger.error(
                "Sample still exceeds token budget even at %s rows and %s columns (est %.0f tokens)",
                len(working_df),
                len(working_df.columns),
                estimated_tokens,
            )
            return working_df, csv_text, estimated_tokens, dropped_columns
        
        drop_count = max(1, int(len(working_df.columns) * 0.15))
        columns_to_drop = list(working_df.columns[-drop_count:])
        dropped_columns.extend(columns_to_drop)
        logger.warning(
            "Dropping %s columns (%s) to satisfy token budget",
            drop_count,
            ", ".join(columns_to_drop[:5]),
        )
        working_df = working_df.drop(columns=columns_to_drop)


async def analyze_dataset_with_code(
    dataset_id: str,
    df: pd.DataFrame,
    dataset_understanding: Dict[str, Any],
    user_instructions: str = "",
    max_sample_rows: Optional[int] = None,
    progress_callback: ProgressCallback = None,
) -> Dict[str, Any]:
    """
    Analyze dataset using Claude's code execution tool.

    Two-phase approach to handle large datasets:
    1. Send truncated dataset (max_sample_rows) to Claude for script generation
    2. Execute generated scripts on full dataset for accurate results

    Args:
        dataset_id: Unique dataset identifier
        df: Pandas DataFrame to analyze (full dataset)
        dataset_understanding: Output from dataset understanding step
        user_instructions: Optional context from user
        max_sample_rows: Maximum rows to send to Claude (default from settings.agent_sample_rows)

    Returns:
        Dict with issues array and investigation results from FULL dataset
    """
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    
    # Use config setting if not specified
    if max_sample_rows is None:
        max_sample_rows = settings.agent_sample_rows
    
    # Determine if we need to truncate
    full_row_count = len(df)
    use_sample = full_row_count > max_sample_rows
    
    if use_sample:
        logger.info(
            f"Dataset has {full_row_count:,} rows. Sampling {max_sample_rows:,} rows for Claude."
        )
        df_sample = df.head(max_sample_rows)
    else:
        logger.info(
            f"Dataset has {full_row_count:,} rows. Sending full dataset to Claude."
        )
        df_sample = df

    _emit_progress(
        progress_callback,
        "progress",
        f"Preparing {len(df_sample):,} rows for Claude (of {full_row_count:,} total)",
    )
    
    # Additional safety: estimate token count and reduce sample if needed
    df_sample, actual_sample_size = _safe_sample_for_tokens(df_sample, max_sample_rows)
    if actual_sample_size < len(df_sample):
        logger.warning(
            f"Further reduced sample to {actual_sample_size} rows to fit token limit"
        )
        df_sample = df_sample.head(actual_sample_size)
        use_sample = True
        _emit_progress(
            progress_callback,
            "progress",
            f"Token budget enforced: using {len(df_sample):,} preview rows",
        )

    # Prepare minimal dataset summary for the agent
    dataset_summary = {
        "rows": df.shape[0],
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }
    
    # Add brief description if available
    if dataset_understanding:
        desc = dataset_understanding.get("summary", {}).get("description", "")
        if desc:
            dataset_summary["description"] = desc[:100]  # Very brief

    # Create concise prompts
    system_prompt = """You are a data quality expert. Analyze the dataset and return ONLY valid JSON (no markdown):

{
  "issues": [
    {
      "id": "dataset_id_issue_slug",
      "type": "missing_values|duplicates|outliers|inconsistent_categories",
      "severity": "low|medium|high",
      "description": "Issue description with specific numbers",
      "affectedColumns": ["col1"],
      "suggestedAction": "What to do",
      "category": "quick_fixes|smart_fixes",
      "affectedRows": 123,
      "investigation": {
        "code": "result = df['col'].isna().sum()",
        "output": "actual output",
        "findings": "what was found"
      }
    }
  ],
  "summary": "Found X issues",
  "completedAt": "ISO timestamp"
}

Rules:
- Use code execution to investigate and get concrete numbers
- Write efficient, scalable code (assign results to 'result' variable)
- Return ONLY the JSON"""

    user_context = f" Context: {user_instructions}" if user_instructions else ""

    try:
        # PHASE 1: Send sample to Claude for script generation
        # Aggressively truncate long text values to save tokens
        df_sample_truncated = df_sample.copy()
        for col in df_sample_truncated.select_dtypes(include=['object']).columns:
            df_sample_truncated[col] = df_sample_truncated[col].astype(str).str[:50]  # Limit text to 50 chars
        
        df_sample_truncated, df_csv, estimated_tokens, dropped_columns = _fit_sample_to_prompt_budget(
            df_sample_truncated
        )
        df_sample = df_sample_truncated
        csv_size_kb = len(df_csv) / 1024
        use_sample = (
            use_sample
            or len(df_sample) < len(df)
            or bool(dropped_columns)
        )
        _emit_progress(
            progress_callback,
            "progress",
            (
                f"Sending {len(df_sample):,} rows and {len(df_sample.columns)} columns to Claude "
                f"(~{estimated_tokens:,.0f} tokens)"
            ),
        )

        logger.info(
            "Phase 1: Sending %s rows, %s columns to Claude (%.1f KB CSV, est %.0f tokens)",
            len(df_sample),
            len(df_sample.columns),
            csv_size_kb,
            estimated_tokens,
        )
        
        truncation_note = ""
        if len(df_sample) < len(df):
            truncation_note += (
                f"\nNOTE: Only {len(df_sample):,} of {len(df):,} rows sent to stay under the token budget."
            )
        if dropped_columns:
            dropped_preview = ", ".join(dropped_columns[:8])
            truncation_note += (
                f"\nNOTE: Dropped {len(dropped_columns)} columns for the preview: "
                f"{dropped_preview}{'...' if len(dropped_columns) > 8 else ''}"
            )

        user_prompt = f"""Dataset: {dataset_summary['rows']:,} rows, {len(dataset_summary['columns'])} columns.
Columns: {', '.join(dataset_summary['columns'][:10])}{'...' if len(dataset_summary['columns']) > 10 else ''}
Types: {json.dumps(dataset_summary['dtypes'], default=str)}{user_context}{truncation_note}

Analyze for: missing values, duplicates, outliers, inconsistencies. Use code execution, then return JSON."""
        
        response = await asyncio.to_thread(
            client.beta.messages.create,
            model=settings.claude_code_exec_model,  # Use Sonnet 4.5 for code execution
            betas=["code-execution-2025-08-25"],
            max_tokens=4096,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": f"""```python
import pandas as pd
from io import StringIO
df = pd.read_csv(StringIO('''{df_csv}'''))
```

{user_prompt}""",
                }
            ],
            tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
        )

        logger.info(f"Claude response: {response}")

        # Extract the final text response (should be JSON)
        final_text = None
        for block in response.content:
            if hasattr(block, "type") and block.type == "text":
                final_text = block.text

        if not final_text:
            raise ValueError("No text response from Claude")

        # Parse the JSON response
        result = _parse_analysis_response(final_text, dataset_id)
        _emit_progress(
            progress_callback,
            "progress",
            "Claude returned structured hypotheses; validating against full dataset",
        )
        
        # PHASE 2: Execute detection scripts on full dataset (if sampled)
        if use_sample and result.get("issues"):
            logger.info(
                f"Phase 2: Executing {len(result['issues'])} detection scripts on full dataset ({full_row_count:,} rows)"
            )
            result = await _execute_scripts_on_full_dataset(
                result,
                df,
                dataset_id,
                progress_callback=progress_callback,
            )
        elif result.get("issues"):
            # Even when not sampled, provide richer evidence on the full data
            result = await _enrich_issues_with_evidence(
                result,
                df,
                dataset_id,
                progress_callback=progress_callback,
            )

        _emit_progress(
            progress_callback,
            "log",
            f"Agent identified {len(result.get('issues', []))} issue(s)",
        )
        return result

    except Exception as e:
        logger.error(f"Code-based analysis failed: {e}")
        raise


async def _execute_scripts_on_full_dataset(
    analysis_result: Dict[str, Any],
    df_full: pd.DataFrame,
    dataset_id: str,
    progress_callback: ProgressCallback = None,
) -> Dict[str, Any]:
    """
    Execute detection scripts from analysis on the full dataset.
    Then parse results with evidence-based formatting.
    
    Args:
        analysis_result: Result from Claude with detection scripts
        df_full: Full dataset DataFrame
        dataset_id: Dataset identifier
    
    Returns:
        Updated analysis result with counts from full dataset and evidence
    """
    updated_issues = []
    _emit_progress(
        progress_callback,
        "progress",
        "Executing agent-authored scripts on full dataset",
    )

    for issue in analysis_result.get("issues", []):
        investigation = issue.get("investigation", {})
        code = investigation.get("code", "")

        if not code:
            # No code to execute, keep as-is
            updated_issues.append(issue)
            continue
        
        try:
            # Execute the detection script on full dataset
            logger.info(f"Executing script for issue {issue.get('id')}: {code[:100]}...")
            
            # Create safe execution environment
            local_vars = {"df": df_full, "pd": pd, "np": __import__("numpy")}
            exec(code, {}, local_vars)
            
            # Extract result (look for common variable names)
            result = None
            for var_name in ["result", "output", "count", "issues", "affected_rows"]:
                if var_name in local_vars:
                    result = local_vars[var_name]
                    break
            
            # Update issue with full dataset results
            if result is not None:
                # Update affected rows if it's a count
                if isinstance(result, (int, float)):
                    issue["affectedRows"] = int(result)
                    issue["description"] = issue["description"].replace(
                        str(investigation.get("output", "")), str(result)
                    ) if investigation.get("output") else issue["description"]
                
                # Update investigation output
                investigation["output"] = str(result)
                investigation["executed_on_full_dataset"] = True
                issue["investigation"] = investigation
            
            updated_issues.append(issue)
            logger.info(f"Successfully executed script for {issue.get('id')}")
            affected = issue.get("affectedRows")
            impact = (
                f"{int(affected):,} rows"
                if isinstance(affected, (int, float))
                else "multiple rows"
            )
            columns = ", ".join(issue.get("affectedColumns", [])[:3]) or "dataset"
            _emit_progress(
                progress_callback,
                "issue",
                f"Validated {issue.get('type', 'issue')} in {columns} ({impact})",
            )

        except Exception as e:
            logger.warning(f"Failed to execute script for issue {issue.get('id')}: {e}")
            # Keep original issue with note
            investigation["execution_error"] = str(e)
            investigation["executed_on_full_dataset"] = False
            issue["investigation"] = investigation
            updated_issues.append(issue)
            _emit_progress(
                progress_callback,
                "log",
                f"Script execution failed for {issue.get('id')}: {e}",
            )

    analysis_result["issues"] = updated_issues
    analysis_result["executed_on_full_dataset"] = True

    # PHASE 3: Parse results with evidence-based formatting
    logger.info("Phase 3: Enriching issues with evidence-based formatting")
    analysis_result = await _enrich_issues_with_evidence(
        analysis_result,
        df_full,
        dataset_id,
        progress_callback=progress_callback,
    )

    _emit_progress(
        progress_callback,
        "progress",
        "Finished executing detection scripts",
    )

    return analysis_result


async def _enrich_issues_with_evidence(
    analysis_result: Dict[str, Any],
    df_full: pd.DataFrame,
    dataset_id: str,
    progress_callback: ProgressCallback = None,
) -> Dict[str, Any]:
    """
    Use an output parser agent to enrich issues with evidence-based examples.
    
    For each issue, the agent will:
    - Extract concrete examples of the problem from the dataset
    - Show what the data looks like now (problematic examples)
    - Suggest what it should look like (corrected examples)
    - Provide actionable, evidence-based suggestions
    
    Args:
        analysis_result: Analysis result with executed scripts
        df_full: Full dataset DataFrame
        dataset_id: Dataset identifier
    
    Returns:
        Enhanced analysis result with evidence-based formatting
    """
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    
    enriched_issues = []
    _emit_progress(
        progress_callback,
        "progress",
        "Extracting concrete evidence examples",
    )

    for issue in analysis_result.get("issues", []):
        try:
            # Extract evidence from the dataset for this issue
            evidence = await _extract_evidence_for_issue(issue, df_full, client)
            
            # Merge evidence into the issue
            if evidence:
                issue["investigation"] = issue.get("investigation", {})
                issue["investigation"]["evidence"] = evidence
                
                # Update description and suggested action with concrete examples
                if evidence.get("examples_current"):
                    issue["description"] = f"{issue['description']}\n\nExamples found: {', '.join(evidence['examples_current'][:3])}"
                
                if evidence.get("examples_fixed"):
                    issue["suggestedAction"] = f"{issue['suggestedAction']}\n\nExample fix: {evidence['examples_current'][0] if evidence.get('examples_current') else 'N/A'} → {evidence['examples_fixed'][0]}"
            
            enriched_issues.append(issue)
            
        except Exception as e:
            logger.warning(f"Failed to enrich issue {issue.get('id')} with evidence: {e}")
            enriched_issues.append(issue)  # Keep original

    analysis_result["issues"] = enriched_issues
    _emit_progress(progress_callback, "log", "Evidence enrichment complete")
    return analysis_result


async def _extract_evidence_for_issue(
    issue: Dict[str, Any],
    df: pd.DataFrame,
    client: anthropic.Anthropic,
) -> Optional[Dict[str, Any]]:
    """
    Extract concrete evidence examples for an issue using an output parser agent.
    
    Args:
        issue: The issue to extract evidence for
        df: Full dataset DataFrame
        client: Anthropic client
    
    Returns:
        Evidence dict with examples_current and examples_fixed
    """
    # Get affected columns and sample problematic data
    affected_columns = issue.get("affectedColumns", [])
    if not affected_columns:
        return None
    
    # Extract sample of problematic rows based on issue type
    sample_data = _get_problematic_sample(issue, df, affected_columns)
    
    if sample_data.empty:
        return None
    
    # Prepare the output parser prompt
    system_prompt = """You are a data quality evidence extractor.

Your task: Given a data quality issue and sample problematic data, extract CONCRETE EXAMPLES.

Return ONLY valid JSON (no markdown, no code fences) in this format:
{
  "examples_current": ["actual problematic value 1", "actual problematic value 2", "actual problematic value 3"],
  "examples_fixed": ["how it should be fixed 1", "how it should be fixed 2", "how it should be fixed 3"],
  "pattern_description": "Brief description of the pattern/issue",
  "fix_strategy": "One-sentence description of how to fix"
}

Rules:
- examples_current: Actual values from the data showing the problem
- examples_fixed: Suggested corrected versions (be specific, not generic)
- Provide 3-5 examples max
- Be concrete: "  John Doe  " not "whitespace issues"
- Show real transformations: "ACME Inc." → "Acme Inc." not "fix capitalization"
"""

    user_prompt = f"""Issue: {issue.get('description')}
Type: {issue.get('type')}
Affected Columns: {', '.join(affected_columns)}

Sample Problematic Data:
{sample_data.to_string()}

Extract concrete examples showing:
1. What the data looks like now (with the problem)
2. What it should look like (fixed)

Return the JSON now."""

    try:
        response = await asyncio.to_thread(
            client.messages.create,
            model=settings.claude_model,  # Use regular model for evidence extraction (lighter task)
            max_tokens=1024,
            messages=[
                {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}
            ],
        )
        
        # Extract text from response
        text = response.content[0].text if response.content else ""
        
        # Strip code fences
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3:]
        text = text.strip()
        
        # Parse JSON
        evidence = json.loads(text)
        return evidence
        
    except Exception as e:
        logger.error(f"Failed to extract evidence: {e}")
        return None


def _get_problematic_sample(
    issue: Dict[str, Any],
    df: pd.DataFrame,
    affected_columns: List[str],
    max_rows: int = 10,
) -> pd.DataFrame:
    """
    Extract a sample of problematic rows based on the issue type.
    
    Args:
        issue: The issue definition
        df: Full dataset
        affected_columns: Columns affected by the issue
        max_rows: Maximum rows to sample
    
    Returns:
        DataFrame with sample problematic rows
    """
    issue_type = issue.get("type", "")
    
    try:
        if issue_type == "missing_values":
            # Get rows with missing values in affected columns
            mask = df[affected_columns].isnull().any(axis=1)
            sample = df[mask].head(max_rows)[affected_columns]
            
        elif issue_type in ["duplicates", "near_duplicates"]:
            # Get duplicate rows
            sample = df[df.duplicated(subset=affected_columns, keep=False)].head(max_rows)[affected_columns]
            
        elif issue_type == "whitespace":
            # Get rows with whitespace issues
            mask = pd.Series(False, index=df.index)
            for col in affected_columns:
                if df[col].dtype == 'object':
                    mask |= df[col].astype(str).str.strip() != df[col].astype(str)
            sample = df[mask].head(max_rows)[affected_columns]
            
        elif issue_type == "outliers":
            # Get rows with outliers (using simple IQR method)
            sample_rows = []
            for col in affected_columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
                    sample_rows.append(outliers.head(max_rows))
            sample = pd.concat(sample_rows).head(max_rows)[affected_columns] if sample_rows else pd.DataFrame()
            
        elif issue_type in ["inconsistent_categories", "category_drift", "supplier_variations"]:
            # Get rows showing variation in categorical values
            sample_rows = []
            for col in affected_columns:
                if df[col].dtype == 'object':
                    # Get rows with less common variations
                    value_counts = df[col].value_counts()
                    rare_values = value_counts[value_counts < value_counts.median()].index
                    sample_rows.append(df[df[col].isin(rare_values)].head(max_rows))
            sample = pd.concat(sample_rows).head(max_rows)[affected_columns] if sample_rows else pd.DataFrame()
            
        elif issue_type == "invalid_dates":
            # Get rows with date parsing issues
            sample_rows = []
            for col in affected_columns:
                try:
                    pd.to_datetime(df[col], errors='coerce')
                    mask = pd.to_datetime(df[col], errors='coerce').isnull() & df[col].notnull()
                    sample_rows.append(df[mask].head(max_rows))
                except Exception:
                    pass
            sample = pd.concat(sample_rows).head(max_rows)[affected_columns] if sample_rows else pd.DataFrame()
            
        else:
            # Default: just get first few rows of affected columns
            sample = df.head(max_rows)[affected_columns]
        
        return sample
        
    except Exception as e:
        logger.warning(f"Failed to extract problematic sample for {issue_type}: {e}")
        # Fallback: return sample of affected columns
        return df.head(max_rows)[affected_columns] if affected_columns else pd.DataFrame()


def _parse_analysis_response(response_text: str, dataset_id: str) -> Dict[str, Any]:
    """Parse and validate the agent's JSON response."""
    # Strip code fences if present
    text = response_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        data = json.loads(text)

        # Ensure completedAt is set
        if "completedAt" not in data:
            data["completedAt"] = datetime.now(timezone.utc).isoformat()

        # Validate issues structure
        if "issues" not in data:
            raise ValueError("Response missing 'issues' field")

        # Ensure all issues have required fields
        for issue in data["issues"]:
            if "id" not in issue:
                issue["id"] = f"{dataset_id}_{issue.get('type', 'unknown')}"
            if "category" not in issue:
                issue["category"] = "quick_fixes"
            if "severity" not in issue:
                issue["severity"] = "medium"

        return data

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}\nResponse: {text[:500]}")
        raise ValueError(f"Invalid JSON response: {e}")


# ============================================================================
# Backward Compatible API
# ============================================================================


async def generate_analysis_issues_with_code(
    dataset_id: str,
    df: pd.DataFrame,
    dataset_understanding: Dict[str, Any],
    user_instructions: str = "",
) -> Dict[str, Any]:
    """
    Generate analysis issues using code execution.

    This is a drop-in replacement for the original generate_analysis_issues
    but uses Claude's code execution tool to gather real evidence.

    Args:
        dataset_id: Unique dataset identifier
        df: Pandas DataFrame to analyze
        dataset_understanding: Output from dataset understanding step
        user_instructions: Optional context from user

    Returns:
        Dict with issues array (same format as before)
    """
    return await analyze_dataset_with_code(
        dataset_id=dataset_id,
        df=df,
        dataset_understanding=dataset_understanding,
        user_instructions=user_instructions,
    )


# ============================================================================
# Simplified Analysis for Quick Testing
# ============================================================================


async def run_quick_investigation(
    df: pd.DataFrame,
    investigation_type: str = "missing_values",
) -> CodeInvestigation:
    """
    Run a quick code-based investigation on a DataFrame.

    Args:
        df: DataFrame to investigate
        investigation_type: Type of investigation (missing_values, duplicates, etc.)

    Returns:
        CodeInvestigation with results
    """
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    prompts = {
        "missing_values": "Analyze missing values. Return dict with column names and counts.",
        "duplicates": "Find duplicate rows. Return count of duplicates.",
        "outliers": "Detect outliers using IQR method. Return count and values.",
        "temporal": "Analyze temporal patterns in the data. Return findings.",
    }

    df_csv = df.to_csv(index=False)

    try:
        response = await asyncio.to_thread(
            client.beta.messages.create,
            model=settings.claude_code_exec_model,  # Use Sonnet 4.5 for code execution
            betas=["code-execution-2025-08-25"],
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": f"""Load this dataset and {prompts.get(investigation_type, prompts["missing_values"])}

                ```python
                import pandas as pd
                from io import StringIO

                csv_data = '''
                {df_csv}
                '''

                df = pd.read_csv(StringIO(csv_data))
                ```

                Now investigate and return your findings as a dict.""",
                }
            ],
            tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
        )

        # Extract code and output from tool use
        code_used = None
        output = None

        for block in response.content:
            if hasattr(block, "type"):
                if block.type == "tool_use" and block.name == "code_execution":
                    code_used = block.input.get("code", "")
                elif block.type == "text":
                    output = block.text

        return CodeInvestigation(
            code=code_used or "No code extracted",
            success=True,
            output=output,
        )

    except Exception as e:
        logger.error(f"Quick investigation failed: {e}")
        return CodeInvestigation(
            code="",
            success=False,
            output=None,
            error=str(e),
        )
