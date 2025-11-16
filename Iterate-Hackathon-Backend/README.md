# Iterate Data Quality Analysis Platform - Backend

[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Anthropic Claude](https://img.shields.io/badge/Claude-4.5_Haiku-8B5CF6?style=flat)](https://www.anthropic.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C?style=flat)](https://www.langchain.com/)

> An intelligent data quality analysis backend powered by autonomous AI agents for automated dataset understanding, error detection, and guided data remediation.

## 🎯 Overview

The Iterate Backend is a FastAPI-based service that leverages **specialized code-generation AI agents** to provide intelligent, context-aware data quality analysis for tabular datasets (CSV/Excel). Unlike traditional approaches that pass entire datasets to LLMs or use simple rule-based heuristics, our innovative architecture:

**Uses LLMs as Code Generators, Not Data Processors**:
- **Specialized Agent System**: Multiple LLM agents (error analysis, correction, dataset summary) guided by system prompts
- **Code Generation over Data Ingestion**: Agents generate Python analysis scripts based on dataset metadata and small samples (10-20 rows)
- **Local Script Execution**: Generated scripts run locally on the **full dataset**, avoiding token window limitations
- **Deterministic Analysis**: Code-based approach eliminates LLM hallucination—scripts produce consistent, verifiable results
- **Scalability**: Analyzes datasets of any size without context limits (millions of rows supported)

**Key Capabilities**:
- **Dataset Understanding Agent**: Generates business-focused summaries from schema and samples
- **Error Analysis Agent**: Creates comprehensive validation scripts to detect data quality issues
- **Error Correction Agent**: Generates remediation scripts based on detected errors
- **Guided Remediation**: Smart fix workflows with conversational user guidance
- **Contextual Chat**: MongoDB-backed interface with dataset context awareness

## 🏗️ Architecture

### Code-Generation Agent Pipeline

The system employs a **multi-agent code-generation architecture** where LLMs generate Python scripts instead of processing data directly:

```
┌──────────────────────────────────────────────────────────────┐
│                     Dataset Upload                            │
│              (CSV/Excel → Pandas → Storage)                   │
│              Extract: Schema + 10-20 Sample Rows              │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│         🤖 Dataset Understanding Agent (LLM)                  │
│  Input: Column names, types, 10-20 sample rows, metadata     │
│  Process: LLM analyzes structure and business context        │
│  Output: Business-focused summary (no code generation)       │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│         🤖 Error Analysis Code Generator (LLM)                │
│  Input: Column names, dataset summary, sample rows           │
│  Process: LLM generates complete Python validation script    │
│  Output: detect_errors.py (executable script)                │
│                                                               │
│  Generated script includes:                                  │
│  • check_missing_values()   • check_duplicates()             │
│  • check_basic_types()      • check_value_ranges()           │
│  • check_category_drifts()  • check_id_consistency()         │
│  • + LLM's custom validation functions                       │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│              ⚙️ Local Script Execution Engine                │
│  • Runs generated script on FULL dataset (all rows)          │
│  • No token limits - processes millions of rows              │
│  • Deterministic results (no LLM hallucination)              │
│  • Captures: Error counts, affected rows, examples           │
│  Output: Structured error report with evidence               │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│      🤖 Error Correction Code Generator (LLM)                 │
│  Input: Error report, column names, conversation context     │
│  Process: LLM generates Python remediation script            │
│  Output: fix_errors.py (executable correction script)        │
│                                                               │
│  Generated script includes:                                  │
│  • Data loading and validation                               │
│  • Targeted fixes for detected issues                        │
│  • Data integrity checks                                     │
│  • Export corrected dataset                                  │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│              🤖 Smart Fix Follow-up Agent (LLM)               │
│  • Generates contextual questions for ambiguous issues       │
│  • Guides users through correction decisions                 │
│  • Adapts generated code based on user responses             │
└──────────────────────────────────────────────────────────────┘
```

**Key Innovation**: LLMs generate analysis code from small samples (10-20 rows), then scripts execute locally on full datasets—combining LLM intelligence with code determinism.

### Why This Architecture Matters

**The Problem with Traditional Approaches**:

| Approach | Method | Limitations |
|----------|--------|-------------|
| **Rule-Based** | Hardcoded validation rules | ❌ Inflexible, can't adapt to different domains<br>❌ Misses context-specific issues<br>❌ No business intelligence |
| **Full Data to LLM** | Send entire dataset to AI | ❌ Token limits (200k max)<br>❌ Can't handle large datasets<br>❌ LLM hallucination risks<br>❌ Expensive and slow |
| **Sampling to LLM** | Send data samples to AI | ❌ May miss issues in unsampled rows<br>❌ Still limited by tokens<br>❌ Hallucination on statistical claims |

**Our Code-Generation Approach**:

| Advantage | How It Works | Benefits |
|-----------|--------------|----------|
| **Unlimited Scale** | LLM generates code from small sample<br>→ Script runs on full dataset locally | ✅ Analyze millions of rows<br>✅ No token limits<br>✅ Fast execution |
| **Zero Hallucination** | LLM writes deterministic Python code<br>→ Pandas processes actual data | ✅ Verifiable results<br>✅ Consistent outputs<br>✅ Code is reviewable |
| **Intelligent + Deterministic** | LLM adds custom validation logic<br>→ Code executes deterministically | ✅ Domain-aware checks<br>✅ Reliable execution<br>✅ Best of both worlds |
| **Cost Effective** | One LLM call generates script<br>→ Script runs unlimited times | ✅ Low API costs<br>✅ Reusable scripts<br>✅ No per-row charges |
| **Transparent** | Generated code is visible<br>→ Users can review/modify | ✅ Explainable AI<br>✅ User control<br>✅ Trust through transparency |

**Real-World Impact**:
- **Dataset Size**: 1M rows × 50 columns
- **Traditional LLM**: Impossible (exceeds token limit)
- **Our Approach**: ✅ LLM sees 20 rows, generates script, script processes all 1M rows in seconds

### Technology Stack

**Core Framework**
- **FastAPI**: High-performance async API framework
- **Pydantic**: Runtime type validation and settings management
- **Pandas**: Dataframe processing and CSV/Excel handling

**AI & Agent Infrastructure**
- **Anthropic Claude 4.5 Haiku**: Primary LLM for code generation (analysis & correction scripts)
- **LangChain**: Agent orchestration and prompt management
- **Local Python Execution**: Generated scripts run on full datasets without token limits
- **Specialized Agents**: Located in `app/tools.py` with guided system prompts

**Data Persistence**
- **MongoDB**: Chat history and conversation state
- **File Storage**: Local filesystem for dataset persistence
- **PostgreSQL (optional)**: Vector embeddings for semantic search

**Agent Guardrails**
- Timeout protection (configurable, default 30s)
- Automatic retry logic with exponential backoff
- Response validation with Pydantic schemas
- Token budget management for large datasets

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- MongoDB instance (local or Atlas)
- Anthropic API key

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Iterate-Hackathon-Backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```bash
# Required: Anthropic API
ANTHROPIC_API_KEY=sk-ant-...

# Required: MongoDB
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=chat_history
MONGODB_COLLECTION_NAME=message_store

# Optional: Model selection
CLAUDE_MODEL=claude-haiku-4-5-20251001
CLAUDE_CODE_EXEC_MODEL=claude-haiku-4-5-20251001

# Optional: Agent tuning
AGENT_TIMEOUT_SECONDS=30.0
AGENT_MAX_RETRIES=2
AGENT_MAX_DATASET_ROWS=100000
AGENT_SAMPLE_ROWS=1000
AGENT_ENABLED=true
```

### Running the Server

```bash
# Development mode with auto-reload
uvicorn app.main:app --reload --port 8000

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at `http://localhost:8000`

API documentation: `http://localhost:8000/docs`

## 📚 API Reference

### Core Endpoints

#### `POST /upload-dataset`
Upload and initialize dataset analysis.

**Request** (multipart/form-data):
```
file: <CSV or Excel file>
user_instructions: string (optional)
```

**Response**:
```json
{
  "dataset_id": "dataset_abc123",
  "file_name": "sales_data.csv",
  "file_type": "csv",
  "file_size_bytes": 1048576,
  "delimiter": ",",
  "storage_path": "/data/dataset_abc123/sales_data.csv",
  "uploaded_at": "2025-11-16T10:30:00Z"
}
```

#### `POST /understand-dataset`
Trigger Dataset Understanding Agent to analyze uploaded data.

**Request**:
```json
{
  "dataset_id": "dataset_abc123",
  "user_instructions": "This is quarterly sales data"
}
```

**Response**: `DatasetUnderstandingModel` (see Agent Contracts)

#### `POST /analyze-dataset`
Run full analysis pipeline with code-based investigations.

**Request**:
```json
{
  "dataset_id": "dataset_abc123",
  "user_instructions": "Focus on revenue anomalies",
  "run_code_investigations": true
}
```

**Response**: Streaming JSON with progress updates and `AnalysisResultModel`

#### `POST /chat`
Conversational interface with dataset context.

**Request**:
```json
{
  "session_id": "user_session_123",
  "message": "What are the main quality issues?",
  "dataset_id": "dataset_abc123"
}
```

**Response**:
```json
{
  "reply": "Based on the analysis, there are 3 main issues..."
}
```

### Smart Fix Endpoints

#### `POST /smart-fix-followup`
Generate contextual follow-up questions for complex issues.

**Request**:
```json
{
  "dataset_id": "dataset_abc123",
  "issue_id": "issue_supplier_variations",
  "user_response": "I want to standardize supplier names"
}
```

**Response**: `SmartFixFollowupModel` with next question and options

## 🤖 AI Agent System - Code Generation Architecture

### Why Code Generation Instead of Direct Data Processing?

Traditional approaches send entire datasets to LLMs, which creates problems:
- **Token Limits**: Claude's 200k token limit can't handle large datasets
- **Hallucination**: LLMs may invent patterns or values that don't exist
- **Cost**: Processing millions of rows through LLM API is expensive
- **Speed**: Multiple LLM calls for large datasets are slow

**Our Solution**: LLMs generate Python code that runs locally on full datasets
- ✅ Analyze unlimited dataset sizes (millions of rows)
- ✅ Deterministic results (code doesn't hallucinate)
- ✅ Low cost (one LLM call generates script, script runs locally)
- ✅ Fast execution (pandas processes data efficiently)
- ✅ Verifiable (users can review/modify generated scripts)

---

### Agent Architecture Details

#### 1. Dataset Understanding Agent

**Purpose**: Generate business-focused understanding from minimal dataset samples

**Input**:
- Dataset metadata (rows, columns, file type)
- **10-20 sample rows only** (critical: avoids token limits)
- Column names, types, basic statistics
- User-provided context

**Process** (No Code Generation):
1. LLM analyzes structure and patterns from minimal sample
2. Infers business domain and use case
3. Generates human-readable descriptions
4. Identifies key observations

**Output**: Structured `DatasetUnderstandingModel`

**Why Small Samples Work**:
- Schema and column types visible in 10-20 rows
- Business context inferable from column names + samples
- Fast (<5s) and low-cost (<2k tokens)

---

#### 2. Error Analysis Code Generation Agent

**Location**: `app/tools.py` → `generate_error_analysis_script()`

**Purpose**: Generate comprehensive Python validation scripts

**Input to LLM**:
- Exact column names
- Dataset metadata/description from understanding agent
- **10-20 sample rows** (not full dataset!)
- File type and delimiter

**LLM Task** (Guided by System Prompts):
Generates complete, executable Python script with:

**Standard Validation Functions** (LLM implements these):
- `check_missing_values()` - NaN detection with row indices
- `check_duplicates()` - Exact duplicate rows
- `check_basic_types()` - Type consistency validation
- `check_value_ranges()` - Age/quantity/price range checks
- `check_allowed_categories()` - Rare/suspicious category detection
- `check_id_consistency()` - Same ID must have consistent values
- `check_product_whitespace()` - Leading/trailing/multiple spaces
- `check_category_drifts()` - Category changes over time
- `check_near_duplicate_rows()` - Near-duplicates (timestamp diffs)

**LLM Freedom**: Add custom validation functions based on dataset understanding

**Output from LLM**: Complete `detect_errors.py` script (as string)

**Local Execution** (NOT in LLM):
```bash
python detect_errors.py /path/to/full_dataset.csv
```

**Script Runs On**:
- **Full dataset** (no row limits, millions of rows supported)
- Local Python environment with pandas
- Deterministic - same dataset = same results

**Script Returns**:
Structured error report:
```
[MISSING_VALUES] ERROR: Column 'Product' has 234 missing values. Example rows: [12, 45, 67]
[DUPLICATE_ROWS] ERROR: 15 exact duplicate rows found. Example rows: [100, 101, 205]
[CATEGORY_DRIFT] WARNING: Product 'ABC123' changed category from 'OTC' to 'OTC:Cold&Flu' at rows [45, 78]
[ID_INCONSISTENCY] ERROR: ID '42' has inconsistent 'price' values: [9.99 (row 10), 12.50 (row 25)]
```

**Key Advantages**:
- ✅ No token limits - analyzes entire dataset
- ✅ Deterministic - code produces consistent results
- ✅ LLM adds intelligence (custom checks) without seeing all data
- ✅ Verifiable - script is reviewable, modifiable, re-runnable

---

#### 3. Error Correction Code Generation Agent

**Location**: `app/tools.py` → `generate_error_correction_script()`

**Purpose**: Generate Python scripts to fix detected issues

**Input to LLM**:
- Exact column names
- **Error report from executed analysis script**
- Conversation history (user decisions on fixes)
- Dataset metadata

**LLM Task**:
Generates correction script that:
- Takes input/output file paths as CLI arguments
- Loads dataset with pandas
- Applies targeted fixes:
  - Fill missing values (forward-fill, mean, mode, custom logic)
  - Remove or consolidate duplicates
  - Standardize formats (dates, categories, whitespace)
  - Fix type inconsistencies
  - Correct value ranges
  - Merge category variations
- Validates corrections
- Exports cleaned dataset

**Output from LLM**: Complete `fix_errors.py` script (as string)

**Local Execution**:
```bash
python fix_errors.py input.csv output_cleaned.csv
```

**Example Generated Code**:
```python
import pandas as pd
import sys

def fix_missing_products(df):
    # Fill using barcode dictionary
    df['Product'] = df['Product'].fillna(
        df['Barcode'].map(barcode_lookup)
    )
    return df

def standardize_categories(df):
    # Merge variations
    df['Dept'] = df['Dept'].replace({
        'OTC': 'OTC:General',
        'OTC:Cold&Flu': 'OTC:General'
    })
    return df

def main():
    df = pd.read_csv(sys.argv[1])
    df = fix_missing_products(df)
    df = remove_duplicates(df)
    df = standardize_categories(df)
    df.to_csv(sys.argv[2], index=False)

if __name__ == "__main__":
    main()
```

**Benefits**:
- ✅ Corrections applied to full dataset deterministically
- ✅ User can review/modify script before running
- ✅ Reproducible - same script = same corrections
- ✅ No data sent to LLM during correction

---

#### 4. Smart Fix Follow-up Agent

**Purpose**: Guide users through complex remediation decisions

**Interaction Pattern**:
1. User selects a "smart fix" issue
2. Agent generates contextual question
3. Provides multiple choice options
4. Adapts follow-up based on response
5. Culminates in actionable fix recommendation

**Example Flow**:
```
Issue: "Supplier name variations detected (95% similarity)"

Agent Q1: "Did the supplier rebrand or change names?"
Options: [Yes - Intentional | No - Data entry errors | Unsure]

User: "No - Data entry errors"

Agent Q2: "Which should be the canonical name?"
Options: [Most frequent | Most recent | Let me specify]

User: "Most frequent"

Agent: Generates merge script to consolidate variants
```

### Agent Communication Contracts

All agents follow strict input/output schemas defined in `docs/agent-contracts.md`. This ensures:
- Type safety between frontend and backend
- Predictable error handling
- Easy testing and validation
- Clear API evolution path

**Key Contract Types**:
- `DatasetUnderstandingModel`: Dataset summary and column descriptions
- `AnalysisResultModel`: List of categorized issues
- `IssueModel`: Individual issue with metadata and suggested action
- `InvestigationModel`: Code execution results
- `SmartFixFollowupModel`: Contextual questions and options

See [`docs/AGENT_ARCHITECTURE.md`](docs/AGENT_ARCHITECTURE.md) for detailed agent design patterns.

## 🔧 Configuration & Tuning

### Agent Performance Tuning

**Timeout Settings**:
```bash
# Balance between thoroughness and responsiveness
AGENT_TIMEOUT_SECONDS=30.0  # Increase for complex datasets
AGENT_MAX_RETRIES=2         # Retry failed agent calls
```

**Dataset Size Limits**:
```bash
# Prevent token overflow
AGENT_MAX_DATASET_ROWS=100000   # Max rows for full processing
AGENT_SAMPLE_ROWS=1000          # Rows sent to Claude for code gen
```

**Model Selection**:
```bash
# Trade cost vs. quality
CLAUDE_MODEL=claude-haiku-4-5-20251001        # Fast, cost-effective
CLAUDE_CODE_EXEC_MODEL=claude-sonnet-4-5      # More capable for code
```

### Fallback Strategies

When `AGENT_ENABLED=false` or agent calls fail:
1. System falls back to `backup_analysis.py` heuristics
2. Uses rule-based detection (missing values, duplicates)
3. No code execution or smart fixes
4. Reduced quality but guaranteed availability

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Test specific module
pytest tests/test_agent.py -v

# Integration tests (requires .env)
pytest tests/integration/ -v
```

## 📂 Project Structure

```
Iterate-Hackathon-Backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app and endpoints
│   ├── agent.py                # Core agent execution logic
│   ├── code_analysis.py        # Code execution agent
│   ├── chat.py                 # Conversational interface
│   ├── tools.py                # Code generation tools
│   ├── config.py               # Settings and environment
│   ├── db.py                   # MongoDB client
│   ├── dataset_store.py        # File storage management
│   ├── backup_analysis.py      # Fallback heuristics
│   ├── excel_context.py        # Dataset context building
│   └── sampling.py             # Data sampling utilities
├── data/                       # Dataset storage (gitignored)
├── scripts/                    # Generated analysis scripts
├── docs/
│   ├── AGENT_ARCHITECTURE.md   # Detailed agent design
│   ├── agent-contracts.md      # Input/output schemas
│   └── CODE_EXECUTION.md       # Code sandbox details
├── tests/
│   ├── test_agent.py
│   ├── test_code_analysis.py
│   └── integration/
├── requirements.txt
├── .env.example
└── README.md
```

## 🔒 Security Considerations

- **API Key Protection**: Store `ANTHROPIC_API_KEY` in environment, never commit
- **Code Execution**: Uses Anthropic's native sandbox (no arbitrary code execution)
- **Input Validation**: Pydantic schemas validate all agent inputs/outputs
- **Rate Limiting**: Implement at reverse proxy level (nginx/Cloudflare)
- **CORS**: Configured for specific frontend origins
- **File Upload**: Limit file sizes, validate CSV/Excel formats

## 🚧 Known Limitations

- **Dataset Size**: Claude context limits dataset samples to ~120k tokens
- **Code Execution**: Limited to pandas/numpy operations (no external libraries)
- **Processing Time**: Large datasets (>50k rows) may take 30-60 seconds
- **Cost**: Each analysis run consumes ~50k-200k tokens depending on dataset size
- **Concurrent Users**: MongoDB connection pooling required for high traffic

## 🛣️ Roadmap

- [ ] **Multi-tenant Support**: Workspace isolation and user management
- [ ] **Advanced Agents**: Time-series analysis, correlation detection
- [ ] **Auto-remediation**: One-click fixes for common issues
- [ ] **Vector Search**: Semantic similarity detection for categorical data
- [ ] **Streaming Analysis**: Real-time progress updates via WebSockets
- [ ] **Custom Rules**: User-defined validation rules via DSL
- [ ] **Caching Layer**: Redis for dataset summaries and analysis results

## 📄 License

[Specify your license here]

## 🤝 Contributing

Contributions welcome! Please read our contributing guidelines and submit pull requests.

## 📞 Support

For issues and questions:
- GitHub Issues: [Link to issues]
- Documentation: [Link to full docs]
- Contact: [Your contact info]

---

**Built with ❤️ using Claude 4.5, FastAPI, and LangChain**
