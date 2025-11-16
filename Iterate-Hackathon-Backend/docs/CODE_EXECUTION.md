# Code Execution System - Local Script Execution

## Overview

The Iterate backend uses **local Python script execution** to run AI-generated validation and remediation scripts on full datasets. Unlike traditional approaches where LLMs process data directly (limited by token windows), our system generates code that executes locally, providing unlimited scalability and deterministic results.

## Why Local Code Execution?

### The Token Limit Problem

**Traditional LLM Data Processing**:
```
User uploads 1M row dataset
  ↓
Send all rows to LLM (IMPOSSIBLE - exceeds token limit)
  ↓
Alternative: Sample 1000 rows (INCOMPLETE - misses rare issues)
  ↓
LLM analyzes sample (UNRELIABLE - hallucination risk)
```

**Problems**:
- ❌ Token limits (200k max) can't handle large datasets
- ❌ Sampling misses edge cases and rare errors
- ❌ LLM hallucination creates false positives
- ❌ High cost ($50+ for 1M rows at $0.05/1k tokens)
- ❌ Slow (multiple API calls for iteration)

### Our Code Generation Solution

```
User uploads 1M row dataset
  ↓
Extract 10-20 row sample + metadata
  ↓
LLM generates Python validation script (~$0.001)
  ↓
Script executes LOCALLY on full 1M rows (~5 seconds)
  ↓
Deterministic, verifiable results
```

**Advantages**:
- ✅ **No token limits**: Script runs on unlimited rows
- ✅ **Complete analysis**: Every row validated
- ✅ **Deterministic**: Code doesn't hallucinate
- ✅ **Low cost**: One generation, unlimited executions
- ✅ **Fast**: Pandas processes millions of rows efficiently
- ✅ **Auditable**: Generated code is inspectable

## Architecture

### Code-Generation Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: Sample Extraction                                      │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Full Dataset (1M rows) → Extract 10-20 sample rows       │  │
│  │ + column names + metadata                                 │  │
│  └───────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: Code Generation Agent (LLM)                            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Input:                                                    │  │
│  │  - 10-20 sample rows                                      │  │
│  │  - Column names and types                                 │  │
│  │  - User-provided metadata                                 │  │
│  │  - System prompt (guides validation functions)           │  │
│  │                                                           │  │
│  │ LLM Generates:                                            │  │
│  │  - Complete executable Python script                     │  │
│  │  - Validation functions (missing values, duplicates...)  │  │
│  │  - Domain-specific checks based on data context          │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Example Generated Script:                                      │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ # LLM generates this entire script                        │  │
│  │ import pandas as pd                                       │  │
│  │ import sys                                                │  │
│  │ import json                                               │  │
│  │                                                           │  │
│  │ def check_missing_values(df):                             │  │
│  │     results = []                                          │  │
│  │     for col in df.columns:                                │  │
│  │         missing = df[col].isna().sum()                    │  │
│  │         if missing > 0:                                    │  │
│  │             results.append({                              │  │
│  │                 "type": "missing_value",                  │  │
│  │                 "column": col,                            │  │
│  │                 "count": int(missing)                     │  │
│  │             })                                            │  │
│  │     return results                                        │  │
│  │                                                           │  │
│  │ def check_duplicates(df):                                 │  │
│  │     duplicates = df[df.duplicated(keep=False)]            │  │
│  │     if len(duplicates) > 0:                               │  │
│  │         return [{"type": "duplicates",                    │  │
│  │                  "count": len(duplicates)}]               │  │
│  │     return []                                             │  │
│  │                                                           │  │
│  │ # LLM may add domain-specific checks                     │  │
│  │ def check_price_reasonableness(df):                      │  │
│  │     # Custom logic based on e-commerce context           │  │
│  │     outliers = df[df['price'] > 10000]                   │  │
│  │     return [{"type": "price_outlier",                    │  │
│  │              "count": len(outliers)}]                    │  │
│  │                                                           │  │
│  │ def main():                                               │  │
│  │     df = pd.read_csv(sys.argv[1])  # FULL dataset        │  │
│  │     errors = []                                           │  │
│  │     errors.extend(check_missing_values(df))               │  │
│  │     errors.extend(check_duplicates(df))                   │  │
│  │     errors.extend(check_price_reasonableness(df))         │  │
│  │     print(json.dumps({"errors": errors}, indent=2))       │  │
│  │     sys.exit(1 if errors else 0)                          │  │
│  │                                                           │  │
│  │ if __name__ == "__main__":                                │  │
│  │     main()                                                │  │
│  └───────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: Local Script Execution                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ subprocess.run([                                          │  │
│  │     "python", "generated_script.py",                      │  │
│  │     "data/full_dataset.csv"  # 1M rows                    │  │
│  │ ])                                                        │  │
│  │                                                           │  │
│  │ Script Output (stdout):                                   │  │
│  │ {                                                         │  │
│  │   "errors": [                                             │  │
│  │     {"type": "missing_value", "column": "email",          │  │
│  │      "count": 1234},                                      │  │
│  │     {"type": "duplicates", "count": 567}                  │  │
│  │   ]                                                       │  │
│  │ }                                                         │  │
│  └───────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 4: Results Processing                                     │
│  Parse JSON output → Store in database → Return to user         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Details

### 1. Code Generation Agents

**Location**: `app/tools.py`

#### Error Analysis Script Generator

```python
async def generate_error_analysis_script(
    dataset_id: str,
    column_names: List[str],
    sample_rows: List[Dict],  # Only 10-20 rows
    metadata: str,
) -> str:
    """
    Generate Python validation script.
    
    Key Innovation:
    - LLM sees tiny sample (low cost)
    - Script runs on full dataset (unlimited scale)
    - Deterministic results (no hallucination)
    """
    
    system_prompt = """You are an expert data quality engineer.

Generate a COMPLETE Python script with these validation functions:

REQUIRED FUNCTIONS:
- check_missing_values(df) - detect null/empty values
- check_duplicates(df) - find duplicate rows
- check_data_types(df) - verify type consistency
- check_value_ranges(df) - detect outliers/anomalies
- check_category_drifts(df) - validate categorical values
- check_date_consistency(df) - validate temporal logic

OPTIONAL FUNCTIONS (based on dataset context):
- Add domain-specific validation functions
- Use your intelligence to suggest relevant checks
- Examples: check_email_format(), check_sku_pattern(), etc.

The script MUST:
1. Accept CSV path as sys.argv[1]
2. Load FULL dataset with pandas
3. Run all validation functions
4. Print JSON results to stdout
5. Exit with code 0 (no errors) or 1 (errors found)

CRITICAL: This script will process the ENTIRE dataset (millions of rows).
Use efficient pandas operations, not row-by-row iteration.

Return ONLY executable Python code, no markdown or explanations."""

    user_prompt = f"""Generate validation script for:

Dataset ID: {dataset_id}
Columns: {json.dumps(column_names)}

Sample Data (first 10-20 rows):
{json.dumps(sample_rows, indent=2)}

Business Context: {metadata}

Remember: Script will run on FULL dataset locally."""

    # Call LLM to generate code
    script_content = await _call_agent_with_retry(
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )
    
    return script_content
```

#### Error Correction Script Generator

```python
async def generate_error_correction_script(
    dataset_id: str,
    errors: List[Dict],  # From validation script
    column_names: List[str],
    sample_rows: List[Dict],
    metadata: str,
) -> str:
    """
    Generate Python remediation script.
    
    Takes detected errors and creates script to fix them.
    """
    
    system_prompt = """You are an expert data engineer.

Generate a Python script that FIXES the detected errors:

COMMON CORRECTION FUNCTIONS:
- fill_missing_values(df, strategies) - impute nulls
- remove_duplicates(df, keep_strategy) - deduplicate
- fix_data_types(df, type_map) - correct types
- fix_outliers(df, method) - handle anomalies
- standardize_categories(df, mappings) - normalize values

The script MUST:
1. Accept input_path and output_path as arguments
2. Load original dataset
3. Apply corrections conservatively
4. Save fixed_dataset.csv
5. Generate correction_report.json with details

Log all changes for auditability.

Return ONLY executable Python code."""

    user_prompt = f"""Generate correction script for:

Dataset ID: {dataset_id}
Detected Errors:
{json.dumps(errors, indent=2)}

Columns: {json.dumps(column_names)}
Sample: {json.dumps(sample_rows, indent=2)}
Context: {metadata}"""

    script_content = await _call_agent_with_retry(
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )
    
    return script_content
```

---

### 2. Local Script Execution

**Location**: `app/code_analysis.py`

#### Script Execution Function

```python
import subprocess
import json
import tempfile
from pathlib import Path

def execute_validation_script(
    script_content: str,
    dataset_path: str,
    timeout_seconds: int = 300
) -> Dict:
    """
    Execute generated Python script locally on full dataset.
    
    Args:
        script_content: LLM-generated Python code
        dataset_path: Path to full CSV file
        timeout_seconds: Max execution time
    
    Returns:
        Parsed JSON results from script output
    """
    
    # Create temporary file for script
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.py',
        delete=False
    ) as script_file:
        script_file.write(script_content)
        script_path = script_file.name
    
    try:
        # Execute script with full dataset
        result = subprocess.run(
            ["python", script_path, dataset_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False  # Don't raise on non-zero exit
        )
        
        # Parse JSON output
        if result.stdout:
            return json.loads(result.stdout)
        else:
            logger.error(f"Script error: {result.stderr}")
            raise ValueError(f"Script failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        logger.error(f"Script timeout after {timeout_seconds}s")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON output: {result.stdout}")
        raise
    finally:
        # Cleanup temp file
        Path(script_path).unlink(missing_ok=True)
```

#### Correction Script Execution

```python
def execute_correction_script(
    script_content: str,
    input_dataset_path: str,
    output_dataset_path: str,
    timeout_seconds: int = 600
) -> Dict:
    """
    Execute data remediation script locally.
    
    Returns correction report with details of changes.
    """
    
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.py',
        delete=False
    ) as script_file:
        script_file.write(script_content)
        script_path = script_file.name
    
    try:
        result = subprocess.run(
            ["python", script_path, input_dataset_path, output_dataset_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=True
        )
        
        # Read correction report
        report_path = Path("correction_report.json")
        if report_path.exists():
            with open(report_path) as f:
                return json.load(f)
        
        return {"status": "success", "output": result.stdout}
        
    finally:
        Path(script_path).unlink(missing_ok=True)
        Path("correction_report.json").unlink(missing_ok=True)
```

---

### 3. Safety and Security

#### Sandboxing Considerations

**Current Implementation**: Local execution without strict sandboxing
- ✅ Suitable for trusted single-user deployment
- ✅ Scripts generated by LLM (limited malicious potential)
- ✅ Timeout protection prevents infinite loops
- ⚠️ Not suitable for multi-tenant production without additional isolation

**Production Recommendations**:
1. **Docker Containers**: Run scripts in isolated containers
2. **Resource Limits**: CPU/memory limits via cgroups
3. **Filesystem Restrictions**: Read-only access except output directory
4. **Network Isolation**: No network access for validation scripts

#### Script Validation

```python
def validate_generated_script(script_content: str) -> bool:
    """
    Basic safety checks on generated code.
    
    Prevents obvious malicious patterns.
    """
    
    # Forbidden patterns
    forbidden = [
        "import os",  # Prevent OS manipulation
        "import subprocess",  # Prevent subprocess spawning
        "import socket",  # Prevent network access
        "open(",  # Only allow pandas read_csv
        "eval(",  # Prevent code injection
        "exec(",  # Prevent code execution
    ]
    
    for pattern in forbidden:
        if pattern in script_content:
            logger.warning(f"Rejected script with forbidden: {pattern}")
            return False
    
    # Required patterns
    required = [
        "import pandas as pd",
        "def main():",
        "pd.read_csv"
    ]
    
    for pattern in required:
        if pattern not in script_content:
            logger.warning(f"Script missing required: {pattern}")
            return False
    
    return True
```

#### Timeout Protection

```python
# In subprocess.run call
timeout=300  # 5 minutes max

try:
    result = subprocess.run(..., timeout=timeout)
except subprocess.TimeoutExpired:
    logger.error("Script exceeded timeout, terminating")
    # Cleanup and return error
│  │     if len(duplicates) > 0:                               │  │
│  │         return [{"type": "duplicates",                    │  │
│  │                  "count": len(duplicates)}]               │  │
│  │     return []                                             │  │
│  │                                                           │  │
│  │ def main():                                               │  │
│  │     df = pd.read_csv(sys.argv[1])  # Load FULL dataset   │  │
│  │     errors = []                                           │  │
│  │     errors.extend(check_missing_values(df))               │  │
│  │     errors.extend(check_duplicates(df))                   │  │
│  │     print(json.dumps({"errors": errors}))                 │  │
│  │                                                           │  │
│  │ if __name__ == "__main__":                                │  │
│  │     main()                                                │  │
│  │     "percentage": float(missing.sum() / len(df) * 100),│   │
│  │     "examples": df[missing].index[:5].tolist()         │   │
│  │ }                                                       │   │
│  │                                                         │   │
│  │ print(json.dumps(result))                              │   │
│  └────────────────────────────────────────────────────────┘   │
└────────────────────────────┬──────────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────────┐
│              Token Budget Manager                             │
│  • Estimates CSV token cost                                   │
│  • Samples dataset to fit budget (~120k tokens)               │
│  • Preserves statistical properties                           │
└────────────────────────────┬──────────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────────┐
│         Anthropic Code Execution Tool (Sandbox)               │
│                                                                │
│  Security Constraints:                                        │
│  • No network access                                          │
│  • No filesystem access (only in-memory data)                 │
│  • 60-second timeout                                          │
│  • Limited to pandas, numpy, json, datetime                   │
│                                                                │
│  Execution:                                                   │
│  1. Inject dataset as CSV string                              │
│  2. Run generated Python code                                 │
│  3. Capture stdout/stderr                                     │
│  4. Return results + execution metadata                       │
└────────────────────────────┬──────────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────────┐
│              Result Parser & Validator                        │
│  • Parse JSON from stdout                                     │
│  • Validate against expected schema                           │
│  • Handle execution errors gracefully                         │
│  • Attach results to issue object                             │
└───────────────────────────────────────────────────────────────┘
```

## Implementation Details

### File: `app/code_analysis.py`

#### Token Budget Management

**Challenge**: Claude's 200k token context limit requires careful dataset sampling.

**Solution**: Adaptive sampling based on estimated token cost

```python
# Conservative constants
CSV_CHARS_PER_TOKEN = 2.0  # Dense CSV: ~2 chars per token
MAX_SAMPLE_TOKENS = 120_000  # Leave 80k for prompts + results
MIN_SAMPLE_ROWS = 25  # Always include at least 25 rows

def _safe_sample_for_tokens(
    df: pd.DataFrame,
    max_rows: int,
    max_tokens: int = MAX_SAMPLE_TOKENS,
) -> tuple[pd.DataFrame, int]:
    """Estimate token count and reduce sample size if needed."""
    
    # Step 1: Take small sample to estimate size
    sample_size = min(50, len(df))
    sample = df.head(sample_size)
    csv_preview = sample.to_csv(index=False)
    
    # Step 2: Estimate tokens per row
    estimated_tokens_per_row = (
        len(csv_preview) / sample_size / CSV_CHARS_PER_TOKEN
    )
    
    # Step 3: Calculate max safe rows
    max_safe_rows = int(max_tokens / estimated_tokens_per_row)
    
    # Step 4: Apply limits
    actual_rows = min(max_rows, max_safe_rows, len(df))
    actual_rows = max(actual_rows, MIN_SAMPLE_ROWS)
    
    logger.info(
        f"Sampled {actual_rows:,} rows "
        f"(~{actual_rows * estimated_tokens_per_row:,.0f} tokens)"
    )
    
    return df.head(actual_rows), actual_rows
```

**Example scaling**:
- Small dataset (100 rows, 5 cols): Use all 100 rows
- Medium dataset (10k rows, 20 cols): Sample 5k rows
- Large dataset (1M rows, 50 cols): Sample 500 rows

#### Code Generation

**Prompt Engineering**: The agent must generate secure, executable code.

```python
async def _generate_investigation_code(
    issue: IssueModel,
    dataset_schema: Dict[str, str],
) -> str:
    """Generate Python code to investigate a specific issue."""
    
    system_prompt = """You are an expert data engineer generating secure 
Python code for data quality analysis.

CRITICAL REQUIREMENTS:
1. Import ONLY: pandas, numpy, json, datetime
2. Read data using: df = pd.read_csv('data.csv')
3. Output results as JSON to stdout using: print(json.dumps(result))
4. Handle edge cases (empty dataframes, missing columns, etc.)
5. Return ONLY executable Python code (no markdown, no explanations)

OUTPUT FORMAT:
- Print a single JSON object with analysis results
- Include specific metrics (counts, percentages, examples)
- Use Python native types (int, float, list, str) for JSON compatibility

EXAMPLE:
import pandas as pd
import json

df = pd.read_csv('data.csv')
missing_count = df['revenue'].isna().sum()

result = {
    "missing_count": int(missing_count),
    "missing_percentage": float(missing_count / len(df) * 100),
    "affected_rows": df[df['revenue'].isna()].index[:10].tolist()
}

print(json.dumps(result))
"""

    user_prompt = f"""Generate code to investigate this issue:

Issue Type: {issue.type}
Description: {issue.description}
Affected Columns: {issue.affectedColumns}

Dataset Schema:
{json.dumps(dataset_schema, indent=2)}

Generate Python code that:
1. Validates the hypothesis in the description
2. Calculates specific metrics (counts, percentages)
3. Finds example row indices demonstrating the issue
4. Outputs results as JSON
"""

    llm = ChatAnthropic(
        model=settings.claude_code_exec_model,
        temperature=0.0,  # Deterministic for code generation
        api_key=settings.anthropic_api_key,
    )
    
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    
    # Clean markdown fences
    code = _strip_code_fences(response.content)
    
    return code
```

**Key design decisions**:
- **Temperature 0.0**: Code must be consistent and correct
- **Library restrictions**: Only pandas/numpy to prevent security issues
- **JSON output**: Structured results enable parsing and validation
- **Example-driven**: Show desired code structure in prompt

#### Secure Execution

**Using Anthropic's Native Tool**:

```python
async def _execute_code_with_anthropic(
    code: str,
    dataset_csv: str,
) -> Dict[str, Any]:
    """Execute Python code using Anthropic's native tool."""
    
    client = anthropic.AsyncAnthropic(
        api_key=settings.anthropic_api_key
    )
    
    # Create message with code execution tool
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
                "content": f"""Execute this Python code with the provided dataset.

DATASET (CSV format):
```csv
{dataset_csv}
```

CODE:
```python
{code}
```

Save the CSV data to 'data.csv' and execute the code. Return the output.
"""
            }
        ],
    )
    
    # Extract execution results
    for block in response.content:
        if block.type == "tool_use" and block.name == "execute_python":
            if hasattr(block, 'result'):
                # Successful execution
                return {
                    "success": True,
                    "output": block.result,
                    "execution_time_ms": _estimate_execution_time(response),
                }
        elif block.type == "error":
            # Execution error
            return {
                "success": False,
                "error": str(block.error),
            }
    
    # No tool use found
    return {
        "success": False,
        "error": "Code execution tool not invoked",
    }
```

**Security guarantees** (provided by Anthropic):
- ✅ **Sandboxed environment**: No access to host system
- ✅ **No network**: Can't make external API calls
- ✅ **No filesystem**: Can't read/write files (except in-memory)
- ✅ **Timeout protection**: 60-second max execution time
- ✅ **Limited libraries**: Only pandas, numpy, stdlib

#### Result Parsing

**Extracting structured results from stdout**:

```python
def _parse_code_output(raw_output: str) -> Any:
    """Parse JSON output from code execution."""
    
    # Find JSON in stdout (may have debug prints)
    lines = raw_output.strip().split('\n')
    
    # Try to parse each line as JSON
    for line in reversed(lines):  # Last line most likely
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    
    # Fallback: try entire output
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse code output as JSON: {e}")
        logger.error(f"Raw output: {raw_output[:500]}")
        return {"raw_output": raw_output, "parse_error": str(e)}
```

#### Full Investigation Flow

**Main entry point** for code-based investigations:

```python
async def run_code_investigation(
    issue: IssueModel,
    dataset_id: str,
    progress_callback: Optional[Callable] = None,
) -> CodeInvestigation:
    """Run code-based investigation for a single issue."""
    
    try:
        # Step 1: Load dataset
        _emit_progress(progress_callback, "log", f"Loading dataset for {issue.id}")
        df = load_dataset(dataset_id)
        
        # Step 2: Sample for token budget
        sampled_df, row_count = _safe_sample_for_tokens(
            df, 
            max_rows=settings.agent_sample_rows,
        )
        _emit_progress(
            progress_callback, 
            "log", 
            f"Sampled {row_count} rows for investigation"
        )
        
        # Step 3: Generate code
        _emit_progress(progress_callback, "log", "Generating analysis code")
        code = await _generate_investigation_code(
            issue=issue,
            dataset_schema={col: str(dtype) for col, dtype in df.dtypes.items()},
        )
        
        # Step 4: Convert to CSV
        dataset_csv = sampled_df.to_csv(index=False)
        
        # Step 5: Execute in sandbox
        _emit_progress(progress_callback, "log", "Executing analysis in sandbox")
        start_time = time.time()
        
        result = await _execute_code_with_anthropic(
            code=code,
            dataset_csv=dataset_csv,
        )
        
        execution_time = (time.time() - start_time) * 1000
        
        # Step 6: Parse output
        if result["success"]:
            parsed_output = _parse_code_output(result["output"])
            _emit_progress(
                progress_callback, 
                "issue", 
                f"Investigation complete: {parsed_output}"
            )
            
            return CodeInvestigation(
                code=code,
                success=True,
                output=parsed_output,
                execution_time_ms=execution_time,
            )
        else:
            logger.error(f"Code execution failed: {result['error']}")
            return CodeInvestigation(
                code=code,
                success=False,
                output=None,
                error=result["error"],
                execution_time_ms=execution_time,
            )
            
    except Exception as e:
        logger.error(f"Investigation failed for {issue.id}: {e}")
        return CodeInvestigation(
            code="",
            success=False,
            output=None,
            error=str(e),
        )
```

## Use Cases

### 1. Missing Value Investigation

**Hypothesis**: "Revenue column has 23% missing values"

**Generated Code**:
```python
import pandas as pd
import json

df = pd.read_csv('data.csv')

missing_mask = df['revenue'].isna()
missing_count = missing_mask.sum()
total_count = len(df)

result = {
    "missing_count": int(missing_count),
    "missing_percentage": float(missing_count / total_count * 100),
    "affected_rows": df[missing_mask].index[:10].tolist(),
    "missing_by_date": df.groupby('date')['revenue'].apply(
        lambda x: x.isna().sum()
    ).to_dict() if 'date' in df.columns else None
}

print(json.dumps(result))
```

**Output**:
```json
{
  "missing_count": 1247,
  "missing_percentage": 23.4,
  "affected_rows": [0, 5, 12, 18, 23, 45, 67, 89, 102, 134],
  "missing_by_date": {
    "2024-01-01": 5,
    "2024-01-02": 12,
    "2024-01-03": 3
  }
}
```

**Business Impact**: Reveals temporal pattern (more missing on specific dates) that triggers smart-fix question: "Did data collection fail on these dates?"

### 2. Duplicate Detection

**Hypothesis**: "Dataset contains duplicate transaction IDs"

**Generated Code**:
```python
import pandas as pd
import json

df = pd.read_csv('data.csv')

# Find duplicates on transaction_id
duplicates = df[df.duplicated(subset=['transaction_id'], keep=False)]

# Group by transaction_id to see variations
dup_groups = duplicates.groupby('transaction_id').agg({
    'amount': lambda x: list(x.unique()),
    'customer_id': lambda x: list(x.unique()),
}).head(5).to_dict('index')

result = {
    "duplicate_count": int(len(duplicates)),
    "unique_duplicate_ids": int(duplicates['transaction_id'].nunique()),
    "example_duplicates": dup_groups,
    "are_exact_duplicates": bool((duplicates.duplicated(keep=False).sum() == len(duplicates)))
}

print(json.dumps(result))
```

**Output**:
```json
{
  "duplicate_count": 48,
  "unique_duplicate_ids": 12,
  "example_duplicates": {
    "TXN001": {
      "amount": [100.0, 100.0, 100.0],
      "customer_id": ["C001", "C001", "C001"]
    },
    "TXN002": {
      "amount": [50.0, 75.0],
      "customer_id": ["C002", "C003"]
    }
  },
  "are_exact_duplicates": false
}
```

**Business Impact**: Reveals that duplicates aren't exact copies (different amounts/customers), triggering smart-fix: "Are these refunds, corrections, or errors?"

### 3. Anomaly Detection

**Hypothesis**: "Product prices show unusual spikes"

**Generated Code**:
```python
import pandas as pd
import numpy as np
import json

df = pd.read_csv('data.csv')

prices = df['price'].dropna()

# Calculate statistics
mean_price = prices.mean()
std_price = prices.std()
z_scores = np.abs((prices - mean_price) / std_price)

# Find outliers (Z-score > 3)
outliers = df[z_scores > 3]

result = {
    "mean_price": float(mean_price),
    "std_price": float(std_price),
    "outlier_count": int(len(outliers)),
    "outlier_percentage": float(len(outliers) / len(df) * 100),
    "example_outliers": outliers[['product', 'price']].head(5).to_dict('records'),
    "max_price": float(prices.max()),
    "min_price": float(prices.min()),
}

print(json.dumps(result))
```

**Output**:
```json
{
  "mean_price": 49.99,
  "std_price": 25.30,
  "outlier_count": 8,
  "outlier_percentage": 0.6,
  "example_outliers": [
    {"product": "Premium Widget", "price": 499.99},
    {"product": "Data Entry Error", "price": 9999.99}
  ],
  "max_price": 9999.99,
  "min_price": 0.01
}
```

**Business Impact**: One clear error (9999.99), others may be legitimate premium products. Smart-fix asks: "Should prices above $500 be flagged for review?"

### 4. Categorical Variation Detection

**Hypothesis**: "Supplier names have inconsistent formatting"

**Generated Code**:
```python
import pandas as pd
import json
from difflib import SequenceMatcher

df = pd.read_csv('data.csv')

suppliers = df['supplier'].dropna().unique()

# Find similar supplier names
similar_groups = []
checked = set()

for i, s1 in enumerate(suppliers):
    if s1 in checked:
        continue
    
    group = [s1]
    for s2 in suppliers[i+1:]:
        similarity = SequenceMatcher(None, s1.lower(), s2.lower()).ratio()
        if similarity > 0.85:  # 85% similar
            group.append(s2)
            checked.add(s2)
    
    if len(group) > 1:
        similar_groups.append({
            "variants": group,
            "count": int(df['supplier'].isin(group).sum()),
        })

result = {
    "total_unique_suppliers": int(len(suppliers)),
    "variation_groups": similar_groups[:5],  # Top 5 groups
    "total_variation_groups": len(similar_groups),
}

print(json.dumps(result))
```

**Output**:
```json
{
  "total_unique_suppliers": 47,
  "variation_groups": [
    {
      "variants": ["Acme Corp", "ACME CORP", "Acme Corporation"],
      "count": 234
    },
    {
      "variants": ["TechSupply Inc", "Tech Supply Inc.", "TechSupply"],
      "count": 156
    }
  ],
  "total_variation_groups": 12
}
```

**Business Impact**: Standardizing would reduce 47 suppliers to ~35, improving reporting accuracy.

## Error Handling

### Execution Failures

**Common failure modes**:

1. **Syntax Error in Generated Code**
   ```python
   # Bad code (missing import)
   df = pd.read_csv('data.csv')  # NameError: pd not defined
   ```
   
   **Recovery**: Log error, return empty investigation, don't block analysis
   
2. **Runtime Error (Invalid Column)**
   ```python
   df['revenu'].isna()  # KeyError: 'revenu' not in columns
   ```
   
   **Prevention**: Include schema in code generation prompt
   
3. **Timeout (>60s)**
   ```python
   # Infinite loop or expensive operation
   for i in range(1000000000):
       pass
   ```
   
   **Recovery**: Anthropic sandbox kills process, returns timeout error

4. **Memory Overflow**
   ```python
   # Creating huge data structures
   df['new_col'] = [list(range(1000000)) for _ in range(len(df))]
   ```
   
   **Prevention**: Token budgeting limits dataset size

### Graceful Degradation

```python
async def investigate_issue_with_fallback(issue: IssueModel, dataset_id: str):
    """Run investigation with fallback to heuristics."""
    
    try:
        # Try code-based investigation
        investigation = await run_code_investigation(issue, dataset_id)
        
        if investigation.success:
            return investigation
        else:
            logger.warning(f"Code investigation failed: {investigation.error}")
            # Fall through to heuristic
            
    except Exception as e:
        logger.error(f"Investigation exception: {e}")
        # Fall through to heuristic
    
    # Fallback: Use heuristic analysis
    return heuristic_investigation(issue, dataset_id)
```

## Performance Characteristics

### Execution Times

**Typical latency breakdown** for code investigation:

| Phase | Duration | Notes |
|-------|----------|-------|
| Code Generation | 2-5s | LLM generates Python script |
| Dataset Sampling | 0.1-0.5s | Load + sample CSV |
| Code Execution | 0.5-3s | Pandas operations in sandbox |
| Result Parsing | <0.1s | JSON extraction |
| **Total** | **3-9s** | Per investigation |

**Parallel execution** for multiple issues:
- 5 investigations sequentially: ~40s
- 5 investigations in parallel: ~9s (limited by slowest)

### Token Costs

**Per investigation**:
- Code generation prompt: ~1,500 tokens
- Dataset CSV (sampled): 10k-120k tokens
- Generated code: ~200 tokens
- Execution result: ~500 tokens
- **Total**: ~15k-125k tokens

**Monthly cost estimates** (at $0.25 per 1M input tokens):
- 1000 investigations/month × 50k tokens avg = 50M tokens
- Cost: ~$12.50/month (input only, output adds ~$5)

## Best Practices

### 1. Validate Before Executing
```python
# Check for required columns before generating code
if not all(col in df.columns for col in issue.affectedColumns):
    logger.warning(f"Missing columns for {issue.id}, skipping investigation")
    return None
```

### 2. Limit Sample Size
```python
# Never send full dataset to avoid token overflow
MAX_SAMPLE_ROWS = min(10_000, len(df))
sampled_df = df.head(MAX_SAMPLE_ROWS)
```

### 3. Structure Outputs
```python
# Always output JSON for parseable results
print(json.dumps({
    "metric1": value1,
    "metric2": value2,
    "examples": [...]
}))
```

### 4. Handle Edge Cases
```python
# Guard against empty dataframes
if len(df) == 0:
    result = {"error": "Empty dataset"}
else:
    result = {"count": len(df)}
```

### 5. Log Everything
```python
logger.info(f"Investigating {issue.id}")
logger.debug(f"Generated code:\n{code}")
logger.info(f"Execution result: {result}")
```

## Security Considerations

### What Anthropic's Sandbox Prevents
- ✅ Network access (no API calls)
- ✅ Filesystem access (no file reads/writes)
- ✅ Subprocess execution (no shell commands)
- ✅ Import of arbitrary packages
- ✅ Infinite loops (60s timeout)

### What We Still Validate
- ❌ Code quality (syntax, logic errors)
- ❌ Output structure (JSON schema)
- ❌ Sensitive data in outputs (PII redaction)
- ❌ Malicious prompts (prompt injection attempts)

### Data Privacy
- Code execution happens in Anthropic's secure environment
- No data persisted after execution
- CSV data transmitted over encrypted API
- Results logged but never include raw PII

## Future Enhancements

### 1. Interactive Debugging
Allow users to modify generated code before execution:
```python
investigation = await run_code_investigation(issue, dataset_id)
# User edits code in UI
modified_code = user_edited_code()
investigation = await re_run_investigation(modified_code, dataset_id)
```

### 2. Code Templates
Pre-built templates for common patterns:
```python
TEMPLATES = {
    "missing_values": "...",
    "duplicates": "...",
    "outliers": "...",
}

code = TEMPLATES[issue.type].format(
    column=issue.affectedColumns[0],
    threshold=settings.outlier_threshold,
)
```

### 3. Multi-Step Investigations
Chain multiple code executions:
```python
# Step 1: Identify outliers
outliers = await investigate("outliers", dataset_id)

# Step 2: Analyze outlier patterns
patterns = await investigate("patterns", dataset_id, context=outliers)
```

### 4. Visualization Generation
Generate matplotlib charts in sandbox:
```python
import matplotlib.pyplot as plt

plt.hist(df['price'], bins=50)
plt.savefig('price_distribution.png')

# Return base64-encoded image
```

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-16  
**Maintainer**: Backend Team
