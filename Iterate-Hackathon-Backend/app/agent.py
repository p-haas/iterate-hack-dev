# app/agent.py
"""
AI Agent execution module for dataset analysis.
Provides structured agent calls with guardrails (timeouts, retries, validation).
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from .config import settings

logger = logging.getLogger(__name__)


# ============================================================================
# Agent Response Models (matching agent-contracts.md)
# ============================================================================


class DatasetSummaryModel(BaseModel):
    name: str
    description: str
    rowCount: int
    columnCount: int
    observations: List[str]


class ColumnSummaryModel(BaseModel):
    name: str
    dataType: Literal["string", "numeric", "date", "categorical", "boolean"]
    description: str
    sampleValues: Optional[List[str]] = None


class DatasetUnderstandingModel(BaseModel):
    summary: DatasetSummaryModel
    columns: List[ColumnSummaryModel]
    suggested_context: str


class InvestigationModel(BaseModel):
    code: Optional[str] = None
    success: Optional[bool] = None
    output: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: Optional[float] = None


class IssueModel(BaseModel):
    id: str
    type: str
    severity: Literal["low", "medium", "high"]
    description: str
    affectedColumns: List[str]
    suggestedAction: str
    category: Literal["quick_fixes", "smart_fixes"]
    affectedRows: Optional[int] = None
    temporalPattern: Optional[str] = None
    investigation: Optional[InvestigationModel] = None


class AnalysisResultModel(BaseModel):
    issues: List[IssueModel]
    summary: str
    completedAt: str


class SmartFixOptionModel(BaseModel):
    key: str
    label: str


class SmartFixFollowupModel(BaseModel):
    prompt: str
    options: List[SmartFixOptionModel]
    examples: Optional[str] = None
    onResponse: Optional[Dict[str, Any]] = None


class AgentError(BaseModel):
    type: str
    message: str


class AgentErrorResponse(BaseModel):
    error: AgentError


# ============================================================================
# Agent Execution Functions
# ============================================================================


def _get_agent_llm() -> ChatAnthropic:
    """Get LangChain LLM configured for agent tasks."""
    return ChatAnthropic(
        model=settings.claude_model,
        temperature=0.1,  # Low temperature for structured outputs
        api_key=settings.anthropic_api_key,
        timeout=settings.agent_timeout_seconds,
        max_retries=0,  # We handle retries ourselves
    )


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences from LLM response."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


async def _call_agent_with_retry(
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 2,
) -> str:
    """
    Call the agent LLM with retry logic.
    
    Args:
        system_prompt: System message for the agent
        user_prompt: User message for the agent
        max_retries: Maximum number of retry attempts
    
    Returns:
        Raw string response from the agent
    
    Raises:
        Exception: If all retries fail
    """
    llm = _get_agent_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Agent call attempt {attempt + 1}/{max_retries + 1}")
            
            # Use asyncio timeout as additional safety
            response = await asyncio.wait_for(
                asyncio.to_thread(llm.invoke, messages),
                timeout=settings.agent_timeout_seconds,
            )
            
            return str(response.content)
            
        except asyncio.TimeoutError as e:
            last_error = e
            logger.warning(f"Agent timeout on attempt {attempt + 1}")
            if attempt < max_retries:
                await asyncio.sleep(1)  # Brief delay before retry
                
        except Exception as e:
            last_error = e
            logger.error(f"Agent error on attempt {attempt + 1}: {e}")
            if attempt < max_retries:
                await asyncio.sleep(1)
    
    raise last_error or Exception("Agent call failed after all retries")


# ============================================================================
# Dataset Understanding Agent
# ============================================================================


async def generate_dataset_understanding(
    dataset_id: str,
    file_name: str,
    row_count: int,
    column_count: int,
    sample_rows: List[Dict[str, Any]],
    column_summaries: List[Dict[str, Any]],
    user_instructions: str = "",
) -> DatasetUnderstandingModel:
    """
    Generate dataset understanding using AI agent.
    
    Args:
        dataset_id: Unique dataset identifier
        file_name: Name of the uploaded file
        row_count: Total number of rows
        column_count: Total number of columns
        sample_rows: List of sample row dicts (max 5)
        column_summaries: List of column info dicts
        user_instructions: Optional context from user
    
    Returns:
        DatasetUnderstandingModel with summary and column descriptions
    
    Raises:
        ValidationError: If agent output doesn't match schema
        Exception: If agent call fails
    """
    system_prompt = """You are a business data analyst helping non-technical users understand their datasets.

Generate a BUSINESS-FOCUSED analysis that explains what the data represents in plain language.

You MUST return ONLY valid JSON (no markdown, no code fences):

{
  "summary": {
    "name": "filename.csv",
    "description": "Clear business explanation of what this dataset contains",
    "rowCount": 123,
    "columnCount": 5,
    "observations": [
      "Business insight 1",
      "Business insight 2"
    ]
  },
  "columns": [
    {
      "name": "column_name",
      "dataType": "string|numeric|date|categorical|boolean",
      "description": "Business meaning (not just data type)",
      "sampleValues": ["val1", "val2", "val3"]
    }
  ],
  "suggested_context": "2-4 sentence summary of dataset purpose and patterns"
}

CRITICAL RULES:
- Use BUSINESS language, not technical jargon
- Explain WHY columns exist, not just WHAT type they are
- suggested_context: summarize dataset purpose for business users
- Return ONLY the JSON, nothing else"""

    user_prompt = f"""Analyze this business dataset:

Dataset: {file_name}
Total Rows: {row_count:,}
Total Columns: {column_count}

Sample Data (first 5 rows):
{json.dumps(sample_rows, indent=2)}

Column Statistics:
{json.dumps(column_summaries, indent=2)}

User's Context: {user_instructions or "None provided yet"}

Generate business-focused understanding JSON."""

    try:
        raw_response = await _call_agent_with_retry(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_retries=settings.agent_max_retries,
        )
        
        # Clean and parse response
        cleaned = _strip_code_fences(raw_response)
        data = json.loads(cleaned)
        
        # Validate against Pydantic model
        return DatasetUnderstandingModel(**data)
        
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Agent returned invalid JSON for dataset understanding: {e}")
        logger.error(f"Raw response: {raw_response[:500]}")
        raise
    except Exception as e:
        logger.error(f"Failed to generate dataset understanding: {e}")
        raise


# ============================================================================
# Analysis Issues Agent
# ============================================================================


async def generate_analysis_issues(
    dataset_id: str,
    dataset_understanding: Dict[str, Any],
    user_instructions: str = "",
    previous_issues: Optional[List[Dict[str, Any]]] = None,
) -> AnalysisResultModel:
    """
    Generate analysis issues using AI agent.
    
    Args:
        dataset_id: Unique dataset identifier
        dataset_understanding: Output from generate_dataset_understanding
        user_instructions: Optional context from user
        previous_issues: Optional previous issues if rerunning
    
    Returns:
        AnalysisResultModel with issues array
    
    Raises:
        ValidationError: If agent output doesn't match schema
        Exception: If agent call fails
    """
    system_prompt = """You are a data quality expert. Analyze the dataset and identify data quality issues.

You MUST return ONLY valid JSON matching this exact schema (no markdown, no code fences):

{
  "issues": [
    {
      "id": "dataset_id_issue_slug",
      "type": "missing_values|duplicates|outliers|inconsistencies|etc",
      "severity": "low|medium|high",
      "description": "Clear description of the issue",
      "affectedColumns": ["column1", "column2"],
      "suggestedAction": "What to do about this issue",
      "category": "quick_fixes|smart_fixes",
      "affectedRows": 123,
      "temporalPattern": "optional temporal pattern description",
      "investigation": {
        "code": "Python code you ran to confirm the issue",
        "success": true,
        "output": {"example": 1},
        "error": "optional error if code failed",
        "execution_time_ms": 100.0
      }
    }
  ],
  "summary": "Analysis complete. Found X issues.",
  "completedAt": "ISO timestamp"
}

CONSTRAINTS:
- severity must be: low, medium, or high
- category must be: quick_fixes or smart_fixes
- Use quick_fixes for automated issues (missing values, duplicates)
- Use smart_fixes for issues requiring user input
- investigation must capture the exact Python you ran plus its output
- IDs must be deterministic: {dataset_id}_{issue_slug}
- Return ONLY the JSON, nothing else"""

    user_prompt = f"""Analyze this dataset for data quality issues:

Dataset ID: {dataset_id}

Dataset Understanding:
{json.dumps(dataset_understanding, indent=2)}

User Instructions: {user_instructions or "None provided"}

Previous Issues: {json.dumps(previous_issues, indent=2) if previous_issues else "None"}

Generate the analysis result JSON now. Include both quick fixes (automated) and smart fixes (requiring user context)."""

    try:
        raw_response = await _call_agent_with_retry(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_retries=settings.agent_max_retries,
        )
        
        # Clean and parse response
        cleaned = _strip_code_fences(raw_response)
        data = json.loads(cleaned)
        
        # Ensure completedAt is set
        if "completedAt" not in data:
            data["completedAt"] = datetime.now(timezone.utc).isoformat()
        
        # Validate against Pydantic model
        return AnalysisResultModel(**data)
        
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Agent returned invalid JSON for analysis: {e}")
        logger.error(f"Raw response: {raw_response[:500]}")
        raise
    except Exception as e:
        logger.error(f"Failed to generate analysis issues: {e}")
        raise


# ============================================================================
# Smart Fix Agent
# ============================================================================


async def generate_smart_fix_followup(
    issue_id: str,
    issue_context: Dict[str, Any],
    user_instructions: str = "",
    smart_fix_history: Optional[List[Dict[str, Any]]] = None,
) -> SmartFixFollowupModel:
    """
    Generate smart fix follow-up questions using AI agent.
    
    Args:
        issue_id: Issue identifier
        issue_context: The issue object
        user_instructions: Context from user
        smart_fix_history: Previous Q/A pairs
    
    Returns:
        SmartFixFollowupModel with prompt and options
    
    Raises:
        ValidationError: If agent output doesn't match schema
        Exception: If agent call fails
    """
    system_prompt = """You are a data cleaning assistant. Generate a follow-up question to help resolve a data quality issue.

You MUST return ONLY valid JSON matching this exact schema (no markdown, no code fences):

{
  "prompt": "Your question to the user",
  "options": [
    {"key": "option1", "label": "First option"},
    {"key": "option2", "label": "Second option"}
  ],
  "examples": "Optional examples to clarify the question",
  "onResponse": {
    "action": "queue_remediation|escalate_human|collect_more_info",
    "notes": "What happens next"
  }
}

CONSTRAINTS:
- Include 2-4 options
- action must be: queue_remediation, escalate_human, or collect_more_info
- Return ONLY the JSON, nothing else"""

    user_prompt = f"""Generate a follow-up question for this issue:

Issue ID: {issue_id}

Issue Context:
{json.dumps(issue_context, indent=2)}

User Instructions: {user_instructions or "None provided"}

Conversation History: {json.dumps(smart_fix_history, indent=2) if smart_fix_history else "None"}

Generate the smart fix follow-up JSON now."""

    try:
        raw_response = await _call_agent_with_retry(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_retries=settings.agent_max_retries,
        )
        
        # Clean and parse response
        cleaned = _strip_code_fences(raw_response)
        data = json.loads(cleaned)
        
        # Validate against Pydantic model
        return SmartFixFollowupModel(**data)
        
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Agent returned invalid JSON for smart fix: {e}")
        logger.error(f"Raw response: {raw_response[:500]}")
        raise
    except Exception as e:
        logger.error(f"Failed to generate smart fix followup: {e}")
        raise
