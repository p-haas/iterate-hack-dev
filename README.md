# Iterate: AI-Powered Data Cleaning & Quality Platform

> **Accelerating data cleaning processes with AI-generated code execution—tailored to any dataset with high accuracy**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-61DAFB?style=flat&logo=react&logoColor=black)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5+-3178C6?style=flat&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Anthropic Claude](https://img.shields.io/badge/Claude-4.5_Haiku-8B5CF6?style=flat)](https://www.anthropic.com/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)

---

## 🎯 The Problem: Data Cleaning is a Universal Bottleneck

Data professionals across industries spend **60-80% of their time** on data cleaning and preparation—not on analysis or insights. This affects:

- **Data Scientists & Analysts**: Cleaning datasets before building models or dashboards
- **Operations Teams**: Preparing clean data for reporting and process optimization
- **Marketing & Sales**: Ensuring CRM data quality for campaigns and forecasting
- **Finance Analysts**: Validating financial data for accurate reporting
- **Researchers**: Standardizing datasets for reproducible analysis
- **Business Intelligence Teams**: Maintaining data warehouse integrity

Traditional approaches fall short:
- ❌ **Manual cleaning**: Time-consuming, error-prone, not scalable
- ❌ **Rule-based tools**: Inflexible, require constant maintenance for new datasets
- ❌ **LLM-based analysis**: Token limits prevent processing large datasets; prone to hallucination
- ❌ **Generic solutions**: Don't understand business context or domain-specific patterns

**The Cost**: Organizations lose millions in productivity, delayed insights, and poor data-driven decisions.

---

## 💡 Our Solution: Intelligent, Adaptive Data Cleaning

Iterate leverages **AI-generated code execution** to provide automated, context-aware data quality analysis that adapts to any dataset structure. Our platform:

✅ **Analyzes datasets of any size** (millions of rows) without token limits  
✅ **Generates deterministic, verifiable results** by executing code locally  
✅ **Understands business context** through intelligent dataset comprehension  
✅ **Provides guided remediation** with conversational AI assistance  
✅ **Scales instantly** to new dataset types without manual configuration  

### How It Works

Unlike traditional approaches, **we use LLMs as code generators, not data processors**:

```
Traditional Approach                    Iterate's Approach
─────────────────────                   ─────────────────────
Send full dataset to LLM    →           Send 10-20 sample rows to LLM
     ↓                                       ↓
LLM processes data          →           LLM generates Python script
(200k token limit)                      (no size limits)
     ↓                                       ↓
Returns analysis            →           Script executes locally on full dataset
(may hallucinate)                       (deterministic results)
     ↓                                       ↓
❌ Fails on large datasets              ✅ Processes millions of rows accurately
```

**Key Innovation**: AI generates intelligent validation scripts from small samples, then executes them locally on complete datasets—combining AI reasoning with code reliability.

---

## 🏗️ Technical Architecture

### Multi-Agent Code Generation System

Our platform employs **specialized AI agents** that generate executable Python code instead of processing data directly:

#### 1. **Dataset Understanding Agent**
- **Input**: Column schema + 10-20 sample rows
- **Process**: LLM analyzes structure and infers business context
- **Output**: Business-focused dataset summary (no code generation)

#### 2. **Error Analysis Code Generator**
- **Input**: Dataset metadata, column types, sample rows
- **Process**: LLM generates comprehensive validation script with functions like:
  - `check_missing_values()` - Identifies null patterns
  - `check_duplicates()` - Detects duplicate records
  - `check_value_ranges()` - Validates numeric/date ranges
  - `check_category_drifts()` - Finds inconsistent categorical values
  - `check_id_consistency()` - Verifies identifier integrity
  - Custom validation functions based on dataset context
- **Output**: `detect_errors.py` (executable Python script)

#### 3. **Local Script Execution Engine**
- Runs generated validation scripts on **full dataset** (all rows)
- No token limits—processes datasets with millions of rows
- Captures structured error reports with evidence
- Deterministic results (same input = same output)

#### 4. **Error Correction Code Generator**
- **Input**: Error report + conversation context
- **Process**: LLM generates targeted remediation script
- **Output**: `fix_errors.py` (executable correction script)

#### 5. **Smart Fix Follow-up Agent**
- Generates contextual questions for ambiguous data issues
- Guides users through correction decisions
- Adapts generated code based on user responses

### Architecture Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Dataset Upload                            │
│         (CSV/Excel → Pandas → Storage + Sampling)            │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│          🤖 Dataset Understanding Agent                      │
│  Analyzes: Schema + 10-20 sample rows                       │
│  Outputs: Business context summary                          │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│          🤖 Error Analysis Code Generator                    │
│  Generates: Complete Python validation script               │
│  Based on: Dataset structure + domain patterns              │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              ⚙️ Local Script Execution                      │
│  Executes: Generated script on FULL dataset                 │
│  Processes: Millions of rows without limits                 │
│  Returns: Structured error report with evidence             │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│          🤖 Error Correction Generator                       │
│  Generates: Targeted remediation scripts                    │
│  Adapts: Based on user guidance via Smart Fix Agent         │
└─────────────────────────────────────────────────────────────┘
```

### Why This Architecture Matters

**Solving LLM Limitations**:
- **Token Limits**: LLMs can't process large datasets directly (200k max tokens ≈ 50k rows)
- **Hallucination**: Direct LLM analysis can produce unreliable results
- **Cost**: Processing millions of rows through LLM APIs is prohibitively expensive

**Our Code-Generation Approach**:
- **Unlimited Scale**: Generated scripts process any dataset size
- **Deterministic**: Code execution produces verifiable, consistent results
- **Cost-Efficient**: One API call generates reusable, executable scripts
- **Transparent**: Users can inspect and modify generated validation logic

---

## 🛠️ Technology Stack

### Backend
- **Framework**: FastAPI (async Python web framework)
- **AI/ML**: 
  - LangChain + Anthropic Claude 4.5 Haiku (code generation)
  - Sentence Transformers (embeddings for context retrieval)
- **Data Processing**: Pandas, OpenPyXL (CSV/Excel handling)
- **Database**: 
  - MongoDB (conversation history, dataset metadata)
  - PostgreSQL + pgvector (vector embeddings)
- **Execution**: subprocess (sandboxed Python script execution)

### Frontend
- **Framework**: React 18 + TypeScript
- **Build Tool**: Vite
- **UI Components**: Radix UI + shadcn/ui (accessible component library)
- **Styling**: Tailwind CSS
- **State Management**: TanStack Query (data fetching/caching)
- **Routing**: React Router

### Infrastructure
- **Package Management**: Bun (frontend), pip (backend)
- **Development**: Hot-reload dev servers, TypeScript strict mode
- **API Communication**: RESTful endpoints + Server-Sent Events (streaming)

---

## 🚀 Key Features

### 1. **Intelligent Dataset Understanding**
- Automatic schema detection and type inference
- Business context extraction from column names and values
- Smart sampling for representative data analysis

### 2. **Comprehensive Error Detection**
- Missing value analysis with pattern recognition
- Duplicate detection (exact and fuzzy matching)
- Type inconsistency identification
- Range validation (dates, numerics, categories)
- Cross-column relationship checks
- Temporal pattern analysis

### 3. **Guided Data Remediation**
- **Quick Fixes**: One-click corrections for common issues
- **Smart Fixes**: AI-guided interactive workflows for complex scenarios
- Conversational clarification for ambiguous data problems
- Preview changes before applying

### 4. **Context-Aware Chat Interface**
- Natural language queries about your dataset
- Dataset-specific Q&A powered by MongoDB memory
- Embedding-based context retrieval for relevant responses

### 5. **Production-Ready Design**
- Async processing for large datasets
- Streaming progress updates via SSE
- Error handling and fallback mechanisms
- Comprehensive logging and observability

---

## 📊 Real-World Impact

### Use Cases

**E-commerce Operations**
- Clean product catalogs with inconsistent categories
- Standardize supplier names and SKU formats
- Validate pricing and inventory data

**Marketing Analytics**
- Deduplicate CRM contacts across systems
- Standardize campaign tracking parameters
- Fix date formatting in event logs

**Financial Reporting**
- Validate transaction data for completeness
- Detect anomalies in expense reports
- Ensure regulatory compliance in audit trails

**Research & Academia**
- Standardize survey response formats
- Clean experimental data before analysis
- Merge datasets from multiple sources

**Sales Operations**
- Clean opportunity data in CRM systems
- Validate lead scoring attributes
- Fix contact information formatting

---

## 🔧 Getting Started

### Prerequisites
- **Backend**: Python 3.11+, pip
- **Frontend**: Node.js 18+, Bun
- **Services**: MongoDB, PostgreSQL (with pgvector)
- **API Keys**: Anthropic API key for Claude

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd iterate-hack
```

2. **Backend Setup**
```bash
cd Iterate-Hackathon-Backend

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys and database URLs

# Run migrations (if applicable)
# Start the server
uvicorn app.main:app --reload --port 8000
```

3. **Frontend Setup**
```bash
cd frontend

# Install dependencies
bun install

# Start development server
bun run dev
```

4. **Access the Application**
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## 📁 Project Structure

```
iterate-hack/
├── Iterate-Hackathon-Backend/      # FastAPI backend
│   ├── app/
│   │   ├── agent.py               # Agent orchestration & LLM config
│   │   ├── tools.py               # Code generation agents
│   │   ├── main.py                # API endpoints
│   │   ├── chat.py                # Conversational interface
│   │   ├── dataset_store.py       # Dataset persistence
│   │   ├── excel_context.py       # Excel/CSV context building
│   │   └── config.py              # Settings management
│   ├── data/                      # Uploaded datasets (gitignored)
│   ├── scripts/                   # Generated validation scripts
│   ├── docs/                      # Architecture documentation
│   └── requirements.txt
│
├── frontend/                       # React TypeScript frontend
│   ├── src/
│   │   ├── components/            # UI components (shadcn/ui)
│   │   ├── pages/                 # Route components
│   │   ├── context/               # React context providers
│   │   ├── hooks/                 # Custom React hooks
│   │   ├── lib/                   # Utilities
│   │   └── types/                 # TypeScript definitions
│   └── package.json
│
└── docs/                          # Project documentation
    ├── agent-contracts.md         # Agent I/O specifications
    ├── ai-agent-integration-plan.md
    └── code-execution-agent-requirements.md
```

---

## 🎓 Documentation

- **[Agent Architecture](Iterate-Hackathon-Backend/docs/AGENT_ARCHITECTURE.md)**: Deep dive into code-generation system
- **[Agent Contracts](docs/agent-contracts.md)**: API schemas and data structures
- **[Integration Plan](docs/ai-agent-integration-plan.md)**: Implementation roadmap
- **[Backend README](Iterate-Hackathon-Backend/README.md)**: Detailed backend documentation

---

## 🧪 Development

### Running Tests
```bash
# Backend tests
cd Iterate-Hackathon-Backend
pytest

# Frontend tests (if configured)
cd frontend
bun test
```

### Code Quality
```bash
# Backend linting
ruff check app/

# Frontend linting
cd frontend
bun run lint
```

---

## 🌟 What Makes Iterate Different

| Traditional Tools | LLM-Direct Analysis | **Iterate (Code Generation)** |
|-------------------|---------------------|-------------------------------|
| Manual rules      | Sends data to LLM   | **LLM generates validation code** |
| Inflexible        | 200k token limit    | **Unlimited dataset size** |
| No context        | May hallucinate     | **Deterministic execution** |
| Slow updates      | High API costs      | **Cost-efficient (one API call)** |
| Generic           | Black box           | **Transparent, inspectable scripts** |

**Iterate combines the best of both worlds**: AI reasoning for context understanding + code execution for reliability and scale.

---

## 🛣️ Roadmap

- [x] Core code-generation agent architecture
- [x] Dataset understanding & error detection
- [x] Smart fix workflows with conversational guidance
- [x] Streaming progress updates
- [ ] Multi-dataset comparison and merging
- [ ] Custom validation rule templates
- [ ] Scheduled data quality monitoring
- [ ] Team collaboration features
- [ ] API for programmatic access
- [ ] Cloud deployment (Azure/AWS)

---

## 👥 Team

Built during the Iterate Hackathon by a team passionate about solving real-world data quality challenges.

---

## 📄 License

[Add your license here]

---

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details on how to:
- Report bugs
- Suggest features
- Submit pull requests

---

## 📞 Support

For questions, issues, or feedback:
- Open an issue on GitHub
- [Contact information]

---

**Iterate**: *Because clean data shouldn't be this hard.*
