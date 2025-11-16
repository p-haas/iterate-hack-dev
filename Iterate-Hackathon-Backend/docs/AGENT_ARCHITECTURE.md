# AI Agent Architecture - Code Generation System

## ⚠️ Important Architecture Update

**This document describes the code-generation agent architecture where LLMs generate Python scripts that execute locally on full datasets.**

**Key Distinction**: Unlike traditional approaches where LLMs process data directly, our agents:
1. **LLMs Generate Code**: Agents in `app/tools.py` create Python validation/correction scripts
2. **Scripts Execute Locally**: Generated code runs on the full dataset without token limits
3. **Deterministic Results**: Code execution eliminates LLM hallucination

For implementation details, see:
- **Code Generation Agents**: `app/tools.py` (`generate_error_analysis_script`, `generate_error_correction_script`)
- **System Prompts**: Detailed in `app/tools.py` guiding what validation functions to generate
- **Updated README**: Comprehensive architecture explanation with diagrams

---

## Overview

The Iterate backend implements a **code-generation multi-agent system** where specialized LLM agents generate Python scripts for dataset analysis rather than processing data directly. This innovative approach solves key limitations:

**Problems with Direct Data Processing**:
- ❌ Token limits (200k max) can't handle large datasets
- ❌ LLM hallucination creates unreliable results
- ❌ High costs for processing millions of rows
- ❌ Slow performance with multiple API calls

**Our Code-Generation Solution**:
- ✅ LLM generates script from 10-20 sample rows
- ✅ Script executes locally on full dataset (no limits)
- ✅ Deterministic, verifiable results (code doesn't hallucinate)
- ✅ Low cost (one API call generates reusable script)
- ✅ Fast (pandas processes data efficiently)

## Core Principles

### 1. Code Generation over Data Processing
**Traditional Approach**: Send data → LLM → Get analysis
**Our Approach**: Send sample → LLM → Get code → Execute locally → Get analysis

Agents generate executable Python scripts based on:
- Dataset schema and column names
- 10-20 sample rows for context
- User-provided metadata
- System prompts guiding validation logic

### 2. Separation of Concerns
Agents are specialized by function:
- **Understanding Agent** (`app/agent.py`): Dataset comprehension from samples (no code generation)
- **Error Analysis Agent** (`app/tools.py`): Generates validation scripts
- **Error Correction Agent** (`app/tools.py`): Generates remediation scripts
- **Follow-up Agent** (`app/agent.py`): Conversational guidance for ambiguous fixes

### 3. Local Execution for Scalability
Generated scripts run locally using:
- Standard Python with pandas
- Full dataset access (no sampling needed)
- Deterministic execution (same input = same output)
- subprocess execution with output capture

### 4. Contract-Driven Communication
All inter-agent and agent-API communication uses strict contracts:
- Input validation via Pydantic models (AnalyzeRequest, UnderstandingResponse, etc.)
- Output validation before returning to caller
- JSON serialization for transport
- Structured script generation with predictable function signatures

## Agent Execution Framework

### Base Infrastructure

Located in `app/agent.py`, the framework provides:

#### LLM Configuration for Code Generation
```python
def _get_agent_llm() -> ChatAnthropic:
    """Get LangChain LLM configured for agent tasks."""
    return ChatAnthropic(
        model=settings.claude_model,  # Claude 4.5 Haiku
        temperature=0.0,  # Deterministic code generation
        api_key=settings.anthropic_api_key,
        timeout=settings.agent_timeout_seconds,
        max_retries=0,  # Custom retry logic
    )
```

**Design decisions for code generation**:
- **Temperature 0.0**: Code must be deterministic and syntactically correct
- **Claude 4.5 Haiku**: Fast, efficient code generation
- **No built-in retries**: Custom retry logic with logging
- **Configurable timeout**: Prevents runaway costs

#### Retry Logic with Exponential Backoff
```python
async def _call_agent_with_retry(
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 2,
) -> str:
    """Call the agent LLM with retry logic."""
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
```

**Key features**:
- **Double timeout protection**: `asyncio.wait_for` + Claude client timeout
- **Attempt logging**: Track retry behavior for debugging
- **1-second delay**: Prevents rapid API hammering
- **Error propagation**: Raises last error if all attempts fail

#### Response Validation
```python
async def generate_dataset_understanding(...) -> DatasetUnderstandingModel:
    """Generate dataset understanding using AI agent."""
    # ... prompt construction ...
    
    try:
        raw_response = await _call_agent_with_retry(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_retries=settings.agent_max_retries,
        )
        
        # Clean markdown fences
        cleaned = _strip_code_fences(raw_response)
        
        # Parse JSON
        data = json.loads(cleaned)
        
        # Validate against Pydantic model
        return DatasetUnderstandingModel(**data)
        
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Agent returned invalid JSON: {e}")
        logger.error(f"Raw response: {raw_response[:500]}")
        raise
    except Exception as e:
        logger.error(f"Failed to generate dataset understanding: {e}")
        raise
```

**Validation pipeline**:
1. **Markdown cleaning**: Remove ```json fences
2. **JSON parsing**: Validate basic structure
3. **Pydantic validation**: Enforce schema compliance
4. **Error logging**: Capture raw response for debugging

## Specialized Agent Implementations

### 1. Dataset Understanding Agent

**File**: `app/agent.py` → `generate_dataset_understanding()`

**Purpose**: Creates human-readable dataset descriptions from small samples **(does NOT generate code)**

**Responsibilities**:
- Analyze dataset structure from 10-20 rows
- Generate business-focused descriptions
- Infer domain and use case
- Provide context for code generation agents

**Why No Code Generation**: This agent focuses on comprehension and communication, not validation logic. The understanding it produces guides the code-generation agents.

**Input Data** (Small Sample Only):
```python
user_prompt = f"""Analyze this business dataset:

Dataset: {file_name}
Total Rows: {row_count:,}  # Note: NOT sent to LLM
Total Columns: {column_count}

Sample Data (first 5-10 rows ONLY):  # Key: sample, not full data
{json.dumps(sample_rows, indent=2)}

Column Statistics:
{json.dumps(column_summaries, indent=2)}

User's Context: {user_instructions or "None provided yet"}

Generate business-focused understanding JSON."""
```

**Output Schema**:
```python
class DatasetUnderstandingModel(BaseModel):
    summary: DatasetSummaryModel  # Business description
    columns: List[ColumnSummaryModel]  # Column meanings
    suggested_context: str  # Dataset purpose
```

**Key Architectural Point**: This understanding becomes context for script-generating agents but the agent itself doesn't produce executable code.

---

### 2. Error Analysis Code Generation Agent

**File**: `app/tools.py` → `generate_error_analysis_script()`

**Purpose**: **Generates complete Python validation scripts** that execute locally on full datasets

**This is Where Code Generation Happens**:
```python
async def generate_error_analysis_script(
    dataset_id: str,
    column_names: List[str],
    sample_rows: List[Dict],  # Only 10-20 rows
    metadata: str,
) -> str:
    """
    Generate a Python script that validates the FULL dataset.
    
    LLM receives: column names + 10-20 sample rows + metadata
    LLM outputs: Complete executable Python script
    Script executes: Locally on full dataset (all rows)
    """
```

**System Prompt** (Guides What Validation Functions to Generate):
```python
system_prompt = """You are an expert data quality engineer generating Python validation scripts.

Generate a COMPLETE, EXECUTABLE Python script with these validation functions:
- check_missing_values(df) - detect null/empty cells
- check_duplicates(df) - find duplicate rows  
- check_data_types(df) - verify expected types
- check_value_ranges(df) - detect outliers/anomalies
- check_category_drifts(df) - validate categorical values
- check_date_consistency(df) - validate temporal logic

IMPORTANT:
- You MUST include these core functions
- You MAY add custom validation functions based on the dataset
- Use your intelligence to suggest domain-specific checks
- Script will run on the FULL dataset (millions of rows possible)
- Return errors as structured JSON

The script MUST:
1. Accept dataset path as command-line argument
2. Load full CSV with pandas
3. Run all validation functions
4. Print JSON results to stdout
5. Exit with code 0 (success) or 1 (validation errors found)

Return ONLY executable Python code, no explanations."""
```

**Input Data** (Sample Only):
```python
user_prompt = f"""Generate validation script for:

Dataset ID: {dataset_id}
Columns: {json.dumps(column_names)}
Sample (10-20 rows): {json.dumps(sample_rows, indent=2)}
Business Context: {metadata}

Generate script that will run on FULL dataset locally."""
```

**Generated Script Example**:
```python
# This entire script is GENERATED by the LLM
import pandas as pd
import sys
import json

def check_missing_values(df):
    """LLM generates this function based on columns."""
    results = []
    for col in df.columns:
        missing = df[col].isna().sum()
        if missing > 0:
            results.append({
                "type": "missing_value",
                "column": col,
                "count": int(missing),
                "percentage": float(missing / len(df) * 100)
            })
    return results

def check_duplicates(df):
    """LLM generates duplicate detection logic."""
    duplicates = df[df.duplicated(keep=False)]
    if len(duplicates) > 0:
        return [{
            "type": "duplicate_rows",
            "count": len(duplicates),
            "sample_indices": duplicates.index[:10].tolist()
        }]
    return []

# ... LLM generates more check functions based on dataset ...

def main():
    df = pd.read_csv(sys.argv[1])  # Load FULL dataset
    
    errors = []
    errors.extend(check_missing_values(df))
    errors.extend(check_duplicates(df))
    # ... run all validation functions ...
    
    print(json.dumps({"errors": errors}, indent=2))
    sys.exit(1 if errors else 0)

if __name__ == "__main__":
    main()
```

**Execution** (Local, Not in Anthropic Sandbox):
```python
# In app/code_analysis.py
result = subprocess.run(
    ["python", generated_script_path, full_dataset_path],
    capture_output=True,
    timeout=300
)
errors = json.loads(result.stdout)  # Structured results
```

**Key Innovation**:
- ✅ LLM sees 10-20 rows (low token cost)
- ✅ Script runs on millions of rows (no limits)
- ✅ Deterministic results (no hallucination)
- ✅ Reusable script (one generation, many executions)

---

### 3. Error Correction Code Generation Agent

**File**: `app/tools.py` → `generate_error_correction_script()`

**Purpose**: **Generates Python remediation scripts** that fix detected errors

**Similar Pattern to Analysis Agent**:
```python
async def generate_error_correction_script(
    dataset_id: str,
    errors: List[Dict],  # From validation script
    column_names: List[str],
    sample_rows: List[Dict],  # 10-20 rows
    metadata: str,
) -> str:
    """
    Generate Python script that FIXES errors in the full dataset.
    
    LLM receives: detected errors + 10-20 sample rows + columns
    LLM outputs: Complete remediation script
    Script executes: Locally, creates fixed_dataset.csv
    """
```

**System Prompt** (Guides Correction Logic):
```python
system_prompt = """You are an expert data engineer generating data remediation scripts.

Generate a COMPLETE Python script that fixes the detected errors:
- fill_missing_values(df, strategies) - impute nulls
- remove_duplicates(df, keep_strategy) - deduplicate
- fix_data_types(df, type_map) - correct types
- fix_outliers(df, method) - handle anomalies
- standardize_categories(df, mappings) - normalize values

IMPORTANT:
- Script receives the errors found by validation script
- Apply fixes conservatively (log all changes)
- Create fixed_dataset.csv with corrections
- Generate correction_report.json with details

Return ONLY executable Python code."""
```

**Generated Correction Script**:
```python
# LLM generates this entire script
import pandas as pd
import sys
import json

def fill_missing_values(df, strategies):
    """LLM decides fill strategies based on data."""
    changes = []
    for col, strategy in strategies.items():
        before = df[col].isna().sum()
        if strategy == "median":
            df[col].fillna(df[col].median(), inplace=True)
        elif strategy == "mode":
            df[col].fillna(df[col].mode()[0], inplace=True)
        # ... LLM generates logic ...
        after = df[col].isna().sum()
        changes.append({"column": col, "filled": before - after})
    return changes

# ... more fix functions ...

def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    df = pd.read_csv(input_path)  # Load FULL dataset
    
    report = {
        "original_rows": len(df),
        "changes": []
    }
    
    # Apply fixes
    report["changes"].extend(fill_missing_values(df, {...}))
    report["changes"].extend(remove_duplicates(df))
    
    # Save fixed dataset
    df.to_csv(output_path, index=False)
    
    # Save report
    with open("correction_report.json", "w") as f:
        json.dumps(report, f, indent=2)
    
    print(f"Fixed dataset saved to {output_path}")

if __name__ == "__main__":
    main()
```

**Execution**:
```python
subprocess.run([
    "python", correction_script_path,
    original_dataset_path,
    fixed_dataset_path
], capture_output=True)
```

---

### 4. Smart Fix Follow-up Agent

**File**: `app/agent.py` → `generate_smart_fix_followup()`

**Purpose**: Conversational guidance when corrections need user input **(does NOT generate code)**

**Use Case**: When error fixes have multiple valid approaches:
- "Should I drop duplicates or keep first occurrence?"
- "How should I fill missing salary values - median, mean, or zero?"
- "Should I remove outliers or cap them?"

**Prompt Strategy**:
```python
system_prompt = """You are a data quality advisor helping users decide on error corrections.

The user has dataset errors that require decisions:
- Multiple valid correction approaches exist
- User needs to understand tradeoffs
- You guide them to best solution

Ask clarifying questions, explain options, suggest best practices.
DO NOT generate code - you provide guidance only."""
```

**Output**: Natural language recommendations, not executable scripts.

**Key Distinction**: This agent communicates, doesn't generate code. Once user decides, the Error Correction Code Generation Agent creates the script.

---
```

**Design Decisions**:
- **Business-first language**: Avoids technical jargon for non-technical users
- **Limited sample data**: Max 5 rows to prevent token overflow
- **Explicit JSON instruction**: Reduces markdown fence pollution
- **Suggested context**: Helps users provide better instructions

### 2. Analysis Issues Agent

**File**: `app/agent.py` → `generate_analysis_issues()`

**Responsibilities**:
- Detect data quality issues
- Classify by severity (low/medium/high)
- Categorize by fix type (quick/smart)
- Generate actionable suggestions

**Prompt Strategy**:
```python
system_prompt = """You are a data quality expert analyzing business datasets.

Your task is to identify data quality issues and categorize them appropriately.

Return ONLY valid JSON (no markdown):
{
  "issues": [
    {
      "id": "{dataset_id}_issue_type",
      "type": "missing_values|duplicates|anomalies|...",
      "severity": "low|medium|high",
      "description": "Clear explanation of the issue",
      "affectedColumns": ["col1", "col2"],
      "suggestedAction": "Specific remediation step",
      "category": "quick_fixes|smart_fixes",
      "affectedRows": 123,
      "temporalPattern": "Optional time-based pattern"
    }
  ],
  "summary": "Overall analysis summary",
  "completedAt": "ISO timestamp"
}

CATEGORIZATION RULES:
- quick_fixes: Automated resolution (duplicates, missing values, formatting)
- smart_fixes: Requires business judgment (anomalies, context alignment)

SEVERITY RULES:
- high: Blocks analysis or critical errors (>50% missing)
- medium: May lead to incorrect insights (20-50% missing)
- low: Minor inconsistencies (<20% missing)"""
```

**Input Data**:
```python
user_prompt = f"""Analyze this dataset for quality issues:

Dataset Understanding:
{json.dumps(dataset_understanding, indent=2)}

User Instructions: {user_instructions or "None"}

Previous Issues (if rerunning):
{json.dumps(previous_issues, indent=2) if previous_issues else "None"}

Detect all data quality issues and categorize appropriately."""
```

**Output Schema**:
```python
class AnalysisResultModel(BaseModel):
    issues: List[IssueModel]
    summary: str
    completedAt: str

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
```

**Design Decisions**:
- **Two-tier categorization**: Quick fixes vs smart fixes guide UI rendering
- **Severity thresholds**: Explicit rules ensure consistency
- **Incremental analysis**: Previous issues allow iterative refinement
- **Temporal awareness**: Date-related patterns trigger time-series logic

### 3. Code Execution Investigation Agent

**File**: `app/code_analysis.py`

**Responsibilities**:
- Generate Python analysis code
- Execute in Anthropic's secure sandbox
- Validate hypotheses with real data
- Extract specific metrics and evidence

**Architecture**:

```
┌─────────────────────────────────────────────────────────────┐
│  Issue Detection (Analysis Agent)                           │
│  • Missing values hypothesis                                │
│  • Duplicate rows hypothesis                                │
│  • Anomaly detection hypothesis                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Code Generation (Investigation Agent)                      │
│  • Create pandas analysis script                            │
│  • Include hypothesis validation logic                      │
│  • Structure output as JSON                                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Token Budget Management                                    │
│  • Sample dataset to fit ~120k tokens                       │
│  • Convert to CSV for efficient encoding                    │
│  • Preserve statistical properties                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Secure Execution (Anthropic Native Sandbox)                │
│  • No network access                                        │
│  • No filesystem access                                     │
│  • Pandas + NumPy available                                 │
│  • 120-second timeout                                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Result Extraction                                          │
│  • Parse stdout for structured results                      │
│  • Extract metrics (counts, percentages, indices)           │
│  • Capture execution time                                   │
│  • Handle errors gracefully                                 │
└─────────────────────────────────────────────────────────────┘
```

**Code Generation Prompt**:
```python
system_prompt = """You are an expert data engineer generating secure Python code 
for data analysis.

CRITICAL REQUIREMENTS:
1. Use ONLY pandas and numpy (no other imports)
2. Read data from the provided CSV variable
3. Output results as JSON to stdout
4. Handle edge cases gracefully
5. Return ONLY executable Python code (no markdown, no explanations)

Example structure:
import pandas as pd
import json

# Analysis logic
df = pd.read_csv('data.csv')
result = {
    "metric": df['col'].isna().sum(),
    "percentage": df['col'].isna().sum() / len(df) * 100,
    "examples": df[df['col'].isna()].index[:5].tolist()
}

print(json.dumps(result))
"""
```

**Token Budget Management**:
```python
def _safe_sample_for_tokens(
    df: pd.DataFrame,
    max_rows: int,
    max_tokens: int = MAX_SAMPLE_TOKENS,
) -> tuple[pd.DataFrame, int]:
    """Estimate token count and reduce sample size if needed."""
    
    # Take small sample to estimate size
    sample_size = min(50, len(df))
    sample = df.head(sample_size)
    csv_preview = sample.to_csv(index=False)
    
    # Estimate tokens (conservative: chars / 2)
    estimated_tokens_per_row = len(csv_preview) / sample_size / CSV_CHARS_PER_TOKEN
    max_safe_rows = int(max_tokens / estimated_tokens_per_row)
    
    # Limit to budget
    actual_rows = min(max_rows, max_safe_rows, len(df))
    actual_rows = max(actual_rows, MIN_SAMPLE_ROWS)
    
    return df.head(actual_rows), actual_rows
```

**Execution via Anthropic Client**:
```python
async def _execute_code_with_anthropic(
    code: str,
    dataset_csv: str,
) -> Dict[str, Any]:
    """Execute Python code using Anthropic's native tool."""
    
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    
    response = await client.messages.create(
        model=settings.claude_code_exec_model,
        max_tokens=4096,
        tools=[
            {
                "type": "code_execution_20250115",
                "name": "execute_python",
            }
        ],
        messages=[
            {
                "role": "user",
                "content": f"""Execute this Python code with the provided data:

DATA (CSV format):
{dataset_csv}

CODE:
{code}

Execute and return results."""
            }
        ],
    )
    
    # Extract results from tool use
    for block in response.content:
        if block.type == "tool_use" and block.name == "execute_python":
            return {
                "success": True,
                "output": block.result,
                "execution_time_ms": response.usage.output_tokens * 0.1  # estimate
            }
    
    return {"success": False, "error": "No code execution result"}
```

**Output Schema**:
```python
class CodeInvestigation(BaseModel):
    code: str                              # Generated Python code
    success: bool                          # Execution succeeded
    output: Any                            # Parsed results (usually dict)
    error: Optional[str] = None           # Error message if failed
    execution_time_ms: Optional[float] = None  # Execution duration
```

**Design Decisions**:
- **Native sandbox**: Anthropic's tool is more secure than custom Docker
- **Token budgeting**: Prevents context overflow with large datasets
- **Pandas-only**: Restricted to built-in libraries for security
- **Structured output**: JSON stdout ensures parseable results
- **Graceful degradation**: Non-critical failures don't block analysis

### 4. Smart Fix Follow-up Agent

**File**: `app/agent.py` → `generate_smart_fix_followup()`

**Responsibilities**:
- Generate contextual questions
- Provide multiple choice options
- Guide users through complex decisions
- Adapt based on conversation history

**Prompt Strategy**:
```python
system_prompt = """You are a data remediation advisor helping users fix 
complex data quality issues.

Generate a follow-up question to clarify the user's intent and guide them 
toward a solution.

Return ONLY valid JSON:
{
  "prompt": "Clear question about the issue",
  "options": [
    {"key": "option1", "label": "User-friendly option description"},
    {"key": "option2", "label": "Alternative approach"}
  ],
  "examples": "Optional: examples to clarify the question",
  "onResponse": {
    "option1": {
      "action": "execute_fix",
      "parameters": {"method": "forward_fill"}
    }
  }
}

GUIDELINES:
- Ask ONE clear question at a time
- Provide 2-4 actionable options
- Include examples if helpful
- Map responses to specific actions when possible"""
```

**Input Data**:
```python
user_prompt = f"""Guide the user through fixing this issue:

Issue:
{json.dumps(issue_context, indent=2)}

User's Latest Response: {user_response}

Conversation History:
{json.dumps(smart_fix_history, indent=2)}

Generate the next follow-up question."""
```

**Output Schema**:
```python
class SmartFixFollowupModel(BaseModel):
    prompt: str
    options: List[SmartFixOptionModel]
    examples: Optional[str] = None
    onResponse: Optional[Dict[str, Any]] = None

class SmartFixOptionModel(BaseModel):
    key: str
    label: str
```

**Example Conversation Flow**:
```
Issue: "Product name variations: 'iPhone 13', 'iphone 13', 'IPHONE 13'"

Agent Q1:
{
  "prompt": "Should these variations be treated as the same product?",
  "options": [
    {"key": "same", "label": "Yes, consolidate to one name"},
    {"key": "different", "label": "No, keep as separate products"}
  ]
}

User: "same"

Agent Q2:
{
  "prompt": "Which format should be the canonical name?",
  "options": [
    {"key": "most_frequent", "label": "Most common: 'iPhone 13'"},
    {"key": "title_case", "label": "Title case: 'iPhone 13'"},
    {"key": "custom", "label": "Let me specify"}
  ],
  "onResponse": {
    "most_frequent": {
      "action": "standardize_values",
      "parameters": {"column": "Product", "canonical": "iPhone 13"}
    }
  }
}
```

**Design Decisions**:
- **One question at a time**: Prevents overwhelming users
- **Actionable options**: Each choice leads to specific next step
- **Conversation history**: Agent adapts based on previous answers
- **Terminal actions**: `onResponse` maps final choices to API calls

## Agent Orchestration Patterns

### Sequential Pipeline

For full dataset analysis, agents execute in sequence:

```python
async def full_analysis_pipeline(dataset_id: str):
    # Step 1: Understanding
    understanding = await generate_dataset_understanding(
        dataset_id=dataset_id,
        file_name=metadata["file_name"],
        row_count=metadata["row_count"],
        ...
    )
    
    # Step 2: Issue Detection
    issues = await generate_analysis_issues(
        dataset_id=dataset_id,
        dataset_understanding=understanding.dict(),
        ...
    )
    
    # Step 3: Code Investigations (parallel within)
    enhanced_issues = await run_code_investigations(
        issues=issues,
        dataset_id=dataset_id,
    )
    
    return enhanced_issues
```

**Advantages**:
- Each stage builds on previous results
- Clear dependency chain
- Easy to debug and log
- Graceful degradation if one stage fails

### Parallel Investigations

Code execution runs in parallel for multiple issues:

```python
async def run_code_investigations(issues: List[IssueModel], dataset_id: str):
    """Run code investigations for all smart-fix issues in parallel."""
    
    tasks = [
        investigate_issue(issue, dataset_id)
        for issue in issues
        if issue.category == "smart_fixes"
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Merge results back into issues
    for issue, result in zip(issues, results):
        if isinstance(result, Exception):
            logger.error(f"Investigation failed for {issue.id}: {result}")
            issue.investigation = None
        else:
            issue.investigation = result
    
    return issues
```

**Advantages**:
- Reduces total latency (5 investigations: 30s → 30s instead of 150s)
- Isolated failures don't block other investigations
- Better resource utilization

## Error Handling & Fallbacks

### Agent Failure Cascade

```
Agent Call
    │
    ├─ Timeout (>30s)
    │   └─ Retry (attempt 2)
    │       ├─ Timeout again
    │       │   └─ Retry (attempt 3)
    │       │       └─ Fail → Fallback
    │       └─ Success → Continue
    │
    ├─ API Error (rate limit, etc.)
    │   └─ Retry with backoff
    │       └─ Fail → Fallback
    │
    ├─ Invalid JSON
    │   └─ Log response
    │       └─ Fail → Fallback
    │
    └─ Validation Error
        └─ Log schema mismatch
            └─ Fail → Fallback
```

### Fallback to Heuristics

When code generation agents fail, the system uses rule-based analysis in `app/backup_analysis.py`:

```python
if settings.agent_enabled:
    try:
        script = await generate_error_analysis_script(...)
        # Execute generated script locally
        issues = execute_validation_script(script, dataset_path)
    except Exception as e:
        logger.error(f"Code generation failed, falling back: {e}")
        # Simple pandas-based heuristics
        issues = run_backup_analysis(dataset_id)
else:
    # Agent disabled, use heuristics directly
    issues = run_backup_analysis(dataset_id)
```

**Heuristic Fallback** (`app/backup_analysis.py`):
- Simple pandas checks (null counts, duplicate counts)
- No LLM involvement
- Fast but less intelligent than generated scripts
- Ensures system always returns results

---

## Architecture Advantages Summary

### Why Code Generation > Direct Data Processing

| Aspect | Direct LLM Processing | Our Code Generation Approach |
|--------|----------------------|------------------------------|
| **Dataset Size** | Limited by token window (200k max) | Unlimited (scripts run locally) |
| **Cost** | High (every row costs tokens) | Low (one script generation) |
| **Reliability** | Hallucination risk | Deterministic code execution |
| **Speed** | Slow (API latency per operation) | Fast (pandas processes locally) |
| **Reusability** | Generate analysis each time | Script reusable across runs |
| **Verification** | Hard to audit LLM decisions | Code is inspectable/testable |
| **Scalability** | Doesn't scale to millions of rows | Scales to any dataset size |

### Token Budget Management

**Sample-Based Generation**:
```
10-20 rows + column names + metadata = ~5k tokens (input)
Generated script = ~2k tokens (output)
Total per dataset: ~7k tokens ≈ $0.001
```

**vs. Full Dataset Processing**:
```
1M rows × average tokens = 50M+ tokens
Cost: $50+ per analysis
Impossible due to context limits
```

### Code Intelligence + LLM Creativity

Our architecture combines:
- **LLM Intelligence**: Suggests domain-specific validation functions based on dataset context
- **Code Determinism**: Executes validation consistently without hallucination
- **System Prompts**: Guide what functions to generate while allowing LLM creativity
- **Local Execution**: No API limits, full dataset access, fast processing

**Example**: For an e-commerce dataset, the LLM might generate:
- Standard checks: `check_missing_values()`, `check_duplicates()`
- Domain-specific: `check_price_reasonableness()`, `check_sku_format()`, `check_inventory_logic()`

The system prompt ensures core functions exist, but the LLM adds valuable dataset-specific logic.

---
    # Agent disabled via config
    issues = run_backup_analysis(dataset_id)
```

**Heuristic capabilities** (see `app/backup_analysis.py`):
- Missing value detection
- Duplicate row detection
- Type inconsistency detection
- Basic outlier detection

**Limitations**:
- No business context understanding
- No smart fix categorization
- No code-based investigations
- Generic remediation suggestions

## Performance Optimization

### Token Budget Management

**Challenge**: Claude has a 200k token limit, but datasets can be millions of rows.

**Solution**: Multi-tier sampling strategy

```python
# Tier 1: Sample for understanding (small)
sample_rows = df.head(5).to_dict('records')  # ~500 tokens

# Tier 2: Sample for code generation (medium)
sample_df = df.head(1000)  # ~20k tokens

# Tier 3: Sample for code execution (large)
sample_df, actual_rows = _safe_sample_for_tokens(
    df, 
    max_rows=10000, 
    max_tokens=120000  # Leave 80k for prompts/results
)
```

### Caching Strategies

**Dataset Understanding**: Cache per dataset_id
```python
# Check cache first
cached = load_dataset_understanding(dataset_id)
if cached:
    return cached

# Generate if not cached
understanding = await generate_dataset_understanding(...)

# Save to cache
save_dataset_understanding(dataset_id, understanding)
```

**Analysis Results**: Cache per (dataset_id, user_instructions hash)
```python
cache_key = f"{dataset_id}_{hash(user_instructions)}"
cached_issues = load_cached_analysis(cache_key)
if cached_issues:
    return cached_issues
```

### Async Execution

All agent calls use `async/await` for non-blocking I/O:

```python
# Bad: Sequential blocking
understanding = generate_dataset_understanding()  # 5s
issues = generate_analysis_issues()              # 10s
# Total: 15s

# Good: Async when possible
async def analyze():
    understanding = await generate_dataset_understanding()  # 5s
    issues = await generate_analysis_issues(understanding) # 10s (can't parallelize)
    # Total: 15s (but doesn't block other requests)
```

## Monitoring & Observability

### Agent Call Logging

Every agent call logs:
- Agent type (understanding, analysis, investigation)
- Input token count (estimated)
- Output token count (from API)
- Execution time
- Success/failure
- Retry attempts

```python
logger.info(
    "Agent call complete",
    extra={
        "agent_type": "dataset_understanding",
        "dataset_id": dataset_id,
        "input_tokens": len(user_prompt) // 4,  # Rough estimate
        "output_tokens": response.usage.output_tokens,
        "duration_ms": elapsed_ms,
        "attempts": attempt + 1,
        "success": True,
    }
)
```

### Progress Callbacks

For long-running analysis, emit progress events:

```python
async def analyze_with_progress(dataset_id: str, callback: Callable):
    callback("log", "Starting dataset understanding...")
    understanding = await generate_dataset_understanding(...)
    
    callback("progress", "Understanding complete. Detecting issues...")
    issues = await generate_analysis_issues(...)
    
    callback("progress", f"Found {len(issues)} issues. Running investigations...")
    for i, issue in enumerate(issues):
        callback("issue", f"Investigating issue {i+1}/{len(issues)}")
        result = await investigate_issue(issue)
    
    callback("complete", "Analysis complete!")
```

### Error Rate Tracking

Track agent failure rates to detect degradation:

```python
# Metrics to monitor
agent_call_total.inc(agent_type=type)
agent_call_failures.inc(agent_type=type, error_type=error)
agent_call_duration.observe(duration_ms, agent_type=type)
```

## Testing Strategies

### Unit Tests

Test individual agent functions with mocked LLM:

```python
@pytest.mark.asyncio
async def test_dataset_understanding_agent():
    # Mock LLM response
    mock_response = {
        "summary": {...},
        "columns": [...],
        "suggested_context": "..."
    }
    
    with patch('app.agent._call_agent_with_retry') as mock:
        mock.return_value = json.dumps(mock_response)
        
        result = await generate_dataset_understanding(
            dataset_id="test123",
            file_name="test.csv",
            ...
        )
        
        assert isinstance(result, DatasetUnderstandingModel)
        assert result.summary.name == "test.csv"
```

### Integration Tests

Test full pipeline with real API (requires API key):

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_analysis_pipeline():
    # Upload real dataset
    dataset_id = await upload_dataset("test_data.csv")
    
    # Run full pipeline
    result = await full_analysis_pipeline(dataset_id)
    
    # Validate structure
    assert len(result.issues) > 0
    assert all(issue.severity in ["low", "medium", "high"] for issue in result.issues)
```

### Contract Tests

Validate agent outputs against schemas:

```python
def test_analysis_result_contract():
    """Ensure AnalysisResultModel matches agent-contracts.md"""
    sample_output = {
        "issues": [{
            "id": "test_issue",
            "type": "missing_values",
            "severity": "high",
            ...
        }],
        "summary": "...",
        "completedAt": "2025-11-16T10:00:00Z"
    }
    
    # Should not raise ValidationError
    result = AnalysisResultModel(**sample_output)
    assert result.completedAt is not None
```

## Future Enhancements

### 1. Agent Memory & Learning
- Store successful prompt patterns
- Learn from user feedback on suggestions
- Personalize based on domain (healthcare, finance, retail)

### 2. Multi-Agent Collaboration
- Debate patterns: Multiple agents propose solutions, vote on best
- Verification agents: Validate other agents' outputs
- Specialist agents: Domain-specific (time-series, NLP, geospatial)

### 3. Streaming Responses
- Stream agent responses token-by-token
- Show intermediate reasoning steps
- Allow user to interrupt long-running analyses

### 4. Advanced Code Execution
- Support additional libraries (scikit-learn, matplotlib)
- Persistent execution environment across investigations
- Interactive debugging (user can modify generated code)

### 5. Feedback Loops
- User rates suggestion quality
- Agents adapt prompts based on low-rated responses
- A/B test different prompt strategies

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-16  
**Maintainer**: Backend Team
