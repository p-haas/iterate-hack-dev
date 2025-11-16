# app/main.py
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .chat import init_llm, chat_with_user
from .config import settings
from .dataset_store import (
    dataset_dir_path,
    generate_dataset_id,
    infer_delimiter,
    load_dataset_metadata,
    load_dataset_context,
    persist_dataset_file,
    save_dataset_context,
    save_smart_fix_response,
    resolve_raw_path,
)
from .backup_analysis import run_backup_analysis
from .excel_context import build_excel_context
from .tools import generate_error_analysis_script, format_error_report_to_json
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
import subprocess
import sys
import asyncio
import json
import logging

logger = logging.getLogger(__name__)

# Dossiers pour stocker fichiers + scripts
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SCRIPTS_DIR = BASE_DIR / "scripts"

DATA_DIR.mkdir(parents=True, exist_ok=True)
SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)


app = FastAPI(title="Claude Excel Context API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str


class ColumnSummary(BaseModel):
    name: str
    dataType: Literal["string", "numeric", "date", "categorical", "boolean"]
    description: str
    sampleValues: Optional[List[str]] = None


class DatasetSummary(BaseModel):
    name: str
    description: str
    rowCount: int
    columnCount: int
    observations: List[str]


class DatasetUnderstandingResponse(BaseModel):
    summary: DatasetSummary
    columns: List[ColumnSummary]
    suggested_context: Optional[str] = None


class UploadDatasetResponse(BaseModel):
    dataset_id: str
    file_name: str
    file_type: str
    file_size_bytes: int
    delimiter: Optional[str]
    storage_path: str
    uploaded_at: str


class DatasetContextRequest(BaseModel):
    instructions: str
    column_edits: Optional[Any] = None


class DatasetContextResponse(BaseModel):
    dataset_id: str
    instructions: str
    column_edits: Optional[Any] = None
    suggested_context: Optional[str] = None
    updated_at: str


class InvestigationResult(BaseModel):
    code: Optional[str] = None
    success: Optional[bool] = None
    output: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: Optional[float] = None
    evidence: Optional[Dict[str, Any]] = None  # Evidence-based examples


class IssueResponse(BaseModel):
    id: str
    type: Literal[
        "missing_values",
        "outliers",
        "inconsistent_categories",
        "invalid_dates",
        "duplicates",
        "whitespace",
        "near_duplicates",
        "supplier_variations",
        "discount_context",
        "category_drift",
    ]
    severity: Literal["low", "medium", "high"]
    description: str
    affectedColumns: List[str]
    suggestedAction: str
    category: Literal["quick_fixes", "smart_fixes"]
    accepted: Optional[bool] = False
    affectedRows: Optional[int] = None
    temporalPattern: Optional[str] = None
    investigation: Optional[InvestigationResult] = None


class AnalysisResultResponse(BaseModel):
    dataset_id: str
    issues: List[IssueResponse]
    summary: str
    completedAt: str


class StreamMessageResponse(BaseModel):
    type: Literal["log", "progress", "issue", "complete"]
    message: str
    timestamp: str


class ApplyIssuesRequest(BaseModel):
    issueIds: List[str]


class ApplyIssuesResponse(BaseModel):
    dataset_id: str
    applied: List[str]
    skipped: List[str]
    message: str


class SmartFixRequest(BaseModel):
    issueId: str
    response: str


class SmartFixResponse(BaseModel):
    dataset_id: str
    issue_id: str
    response: str
    updated_at: str


class ChatDatasetRequest(BaseModel):
    session_id: str
    dataset_id: str
    message: str


@app.on_event("startup")
def startup_event():
    # Initialiser le LLM (Claude)
    init_llm()


@app.post("/datasets", response_model=UploadDatasetResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """Persist an uploaded dataset and capture metadata for later steps."""

    if not file.filename:
        raise HTTPException(
            status_code=400, detail="Le fichier doit avoir un nom valide."
        )

    extension = Path(file.filename).suffix.lower()
    if extension not in [".csv", ".xlsx", ".xls"]:
        raise HTTPException(
            status_code=400, detail="Le fichier doit être un CSV ou un Excel."
        )

    file_bytes = await file.read()
    delimiter = infer_delimiter(file_bytes) if extension == ".csv" else None

    dataset_id = generate_dataset_id()
    metadata = persist_dataset_file(
        DATA_DIR,
        dataset_id,
        file.filename,
        file_bytes,
        file.content_type,
        delimiter,
    )

    return UploadDatasetResponse(
        dataset_id=dataset_id,
        file_name=file.filename,
        file_type=metadata["file_type"],
        file_size_bytes=metadata["file_size_bytes"],
        delimiter=metadata["delimiter"],
        storage_path=metadata["stored_file"],
        uploaded_at=metadata["uploaded_at"],
    )


def _infer_column_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    if pd.api.types.is_object_dtype(series):
        unique_ratio = 0
        try:
            unique_ratio = series.nunique(dropna=True) / max(len(series), 1)
        except Exception:
            unique_ratio = 0
        if unique_ratio <= 0.2:
            return "categorical"
    return "string"


def _build_column_summary(series: pd.Series) -> ColumnSummary:
    data_type = _infer_column_type(series)
    total = len(series)
    missing = int(series.isna().sum())
    unique = int(series.nunique(dropna=True))
    description = (
        f"Detected as {data_type}. {unique} unique values"
        f" with {missing} missing entries out of {total}."
    )

    samples: List[str] = []
    for value in series.dropna().head(3).tolist():
        samples.append(str(value))

    return ColumnSummary(
        name=str(series.name),
        dataType=data_type,  # type: ignore[arg-type]
        description=description,
        sampleValues=samples or None,
    )


def _load_dataframe(metadata: dict) -> pd.DataFrame:
    raw_path = resolve_raw_path(DATA_DIR, metadata)
    file_type = metadata.get("file_type", "csv")
    delimiter = metadata.get("delimiter") or ","

    if file_type == "excel":
        return pd.read_excel(raw_path)
    return pd.read_csv(raw_path, delimiter=delimiter)


def _persist_analysis_result(dataset_dir: Path, result: AnalysisResultResponse) -> None:
    analysis_path = dataset_dir / "analysis.json"
    analysis_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")


def _load_analysis_result(dataset_id: str) -> AnalysisResultResponse:
    analysis_path = dataset_dir_path(DATA_DIR, dataset_id) / "analysis.json"
    if not analysis_path.exists():
        raise FileNotFoundError(
            f"Aucune analyse enregistrée pour dataset_id={dataset_id}. Lancez /analysis avant d'appliquer des corrections."
        )
    data = json.loads(analysis_path.read_text(encoding="utf-8"))
    return AnalysisResultResponse(**data)


def _persist_applied_issues(dataset_id: str, applied: List[str]) -> None:
    path = dataset_dir_path(DATA_DIR, dataset_id) / "applied_issues.json"
    payload = {
        "dataset_id": dataset_id,
        "applied": applied,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_applied_issues(dataset_id: str) -> List[str]:
    path = dataset_dir_path(DATA_DIR, dataset_id) / "applied_issues.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("applied", [])


def _persist_smart_fix_response(
    dataset_id: str, issue_id: str, response: str
) -> SmartFixResponse:
    saved = save_smart_fix_response(DATA_DIR, dataset_id, issue_id, response)
    return SmartFixResponse(
        dataset_id=dataset_id,
        issue_id=issue_id,
        response=saved["response"],
        updated_at=saved["updated_at"],
    )


@app.get(
    "/datasets/{dataset_id}/understanding", response_model=DatasetUnderstandingResponse
)
async def get_dataset_understanding(dataset_id: str, force_refresh: bool = False):
    """
    Get dataset understanding using AI agent or cached result.

    Args:
        dataset_id: Dataset identifier
        force_refresh: Force agent to re-analyze (bypass cache)

    Returns:
        DatasetUnderstandingResponse with business-focused descriptions
    """
    from .agent import generate_dataset_understanding
    from .sampling import (
        smart_sample_dataframe,
        prepare_sample_rows,
        prepare_column_summaries,
    )
    from .config import settings

    try:
        metadata = load_dataset_metadata(DATA_DIR, dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    dataset_dir = dataset_dir_path(DATA_DIR, dataset_id)
    understanding_cache_path = dataset_dir / "understanding.json"

    # Try to load from cache first
    if not force_refresh and understanding_cache_path.exists():
        try:
            cached_data = json.loads(
                understanding_cache_path.read_text(encoding="utf-8")
            )
            logger.info(f"Returning cached understanding for {dataset_id}")
            return DatasetUnderstandingResponse(**cached_data)
        except Exception as e:
            logger.warning(f"Failed to load cached understanding: {e}")
            # Continue to regenerate

    # Load and sample dataset
    try:
        df = _load_dataframe(metadata)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Erreur de lecture du dataset: {exc}"
        ) from exc

    # Smart sampling for large datasets
    original_row_count = len(df)
    sampled_df = smart_sample_dataframe(df, max_sample_rows=300)
    logger.info(
        f"Sampled {len(sampled_df)} rows from {original_row_count} for analysis"
    )

    # Prepare inputs for agent
    sample_rows = prepare_sample_rows(sampled_df, max_rows=5)
    column_summaries = prepare_column_summaries(sampled_df)

    # Try agent-powered understanding if enabled
    if settings.agent_enabled:
        try:
            logger.info(f"Calling agent for dataset understanding: {dataset_id}")
            agent_result = await generate_dataset_understanding(
                dataset_id=dataset_id,
                file_name=metadata.get("original_filename", dataset_id),
                row_count=original_row_count,
                column_count=len(df.columns),
                sample_rows=sample_rows,
                column_summaries=column_summaries,
                user_instructions="",
            )

            # Convert to response model
            response = DatasetUnderstandingResponse(
                summary=DatasetSummary(
                    name=agent_result.summary.name,
                    description=agent_result.summary.description,
                    rowCount=agent_result.summary.rowCount,
                    columnCount=agent_result.summary.columnCount,
                    observations=agent_result.summary.observations,
                ),
                columns=[
                    ColumnSummary(
                        name=col.name,
                        dataType=col.dataType,
                        description=col.description,
                        sampleValues=col.sampleValues,
                    )
                    for col in agent_result.columns
                ],
                suggested_context=agent_result.suggested_context,
            )

            # Cache the result
            understanding_cache_path.write_text(
                response.model_dump_json(indent=2), encoding="utf-8"
            )
            logger.info(f"Cached agent understanding for {dataset_id}")

            return response

        except Exception as e:
            logger.error(f"Agent failed for understanding: {e}")
            logger.warning("Falling back to heuristic analysis")
            # Fall through to heuristics

    # Fallback: Heuristic-based understanding
    logger.info(f"Using heuristic understanding for {dataset_id}")
    columns = [_build_column_summary(df[col]) for col in df.columns]

    missing_counts = df.isna().sum().sort_values(ascending=False)
    observations: List[str] = []
    for col, count in missing_counts.head(3).items():
        if count > 0:
            observations.append(f"{col} contient {int(count)} valeurs manquantes")

    if not observations:
        observations.append("Aucun signal de qualité majeur détecté")

    summary = DatasetSummary(
        name=metadata.get("original_filename", dataset_id),
        description=f"Dataset importé via {metadata.get('file_type', 'csv').upper()} le {metadata.get('uploaded_at', '')}",
        rowCount=int(df.shape[0]),
        columnCount=int(df.shape[1]),
        observations=observations,
    )

    response = DatasetUnderstandingResponse(
        summary=summary,
        columns=columns,
        suggested_context=None,
    )

    # Cache heuristic result too
    understanding_cache_path.write_text(
        response.model_dump_json(indent=2), encoding="utf-8"
    )

    return response


@app.get("/datasets/{dataset_id}/context", response_model=DatasetContextResponse)
def get_dataset_context(dataset_id: str):
    try:
        metadata = load_dataset_metadata(DATA_DIR, dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    context = load_dataset_context(DATA_DIR, dataset_id)

    # Try to get suggested_context from cached understanding
    dataset_dir = dataset_dir_path(DATA_DIR, dataset_id)
    understanding_cache_path = dataset_dir / "understanding.json"
    suggested_context = None

    if understanding_cache_path.exists():
        try:
            cached_understanding = json.loads(
                understanding_cache_path.read_text(encoding="utf-8")
            )
            suggested_context = cached_understanding.get("suggested_context")
        except Exception:
            pass

    if context is None:
        context = {
            "dataset_id": dataset_id,
            "instructions": "",
            "column_edits": None,
            "suggested_context": suggested_context,
            "updated_at": metadata.get(
                "uploaded_at", datetime.now(timezone.utc).isoformat()
            ),
        }
    else:
        # Add suggested_context to existing context
        context["suggested_context"] = suggested_context

    return DatasetContextResponse(**context)


@app.post("/datasets/{dataset_id}/context", response_model=DatasetContextResponse)
def save_dataset_context_endpoint(dataset_id: str, payload: DatasetContextRequest):
    try:
        load_dataset_metadata(DATA_DIR, dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        context = save_dataset_context(
            DATA_DIR,
            dataset_id,
            payload.instructions,
            payload.column_edits,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return DatasetContextResponse(**context)


def _format_sse_message(message: StreamMessageResponse) -> str:
    return f"data: {message.model_dump_json() }\n\n"


async def _analysis_stream_generator(
    messages: List[StreamMessageResponse],
) -> AsyncGenerator[str, None]:
    for message in messages:
        yield _format_sse_message(message)
        await asyncio.sleep(0.4)


def _build_stream_messages(
    dataset_id: str, df: pd.DataFrame
) -> List[StreamMessageResponse]:
    now = datetime.now(timezone.utc)
    messages: List[StreamMessageResponse] = []

    def push(msg_type: str, text: str) -> None:
        messages.append(
            StreamMessageResponse(
                type=msg_type,  # type: ignore[arg-type]
                message=text,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        )

    push("log", "Starting dataset analysis...")
    push("progress", "Loading dataset into memory")
    push("progress", f"Detecting missing values across {len(df.columns)} columns")
    missing_columns = [col for col in df.columns if df[col].isna().any()]
    if missing_columns:
        push("issue", f"Missing data detected in {', '.join(missing_columns[:3])}")
    duplicates = int(df.duplicated().sum())
    if duplicates:
        push("issue", f"Detected {duplicates} duplicate rows")
    push("progress", "Generating cleaning recommendations")
    push("complete", "Analysis complete")
    return messages


@app.get("/datasets/{dataset_id}/analysis/stream")
async def stream_dataset_analysis(dataset_id: str):
    try:
        metadata = load_dataset_metadata(DATA_DIR, dataset_id)
        df = _load_dataframe(metadata)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Erreur de lecture du dataset: {exc}"
        ) from exc

    messages = _build_stream_messages(dataset_id, df)
    return StreamingResponse(
        _analysis_stream_generator(messages),
        media_type="text/event-stream",
    )


async def _run_dataset_analysis(dataset_id: str) -> AnalysisResultResponse:
    """
    Run dataset analysis using Claude code execution (if enabled) or fallback to heuristics.
    """
    metadata = load_dataset_metadata(DATA_DIR, dataset_id)
    try:
        df = _load_dataframe(metadata)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Erreur de lecture du dataset: {exc}"
        ) from exc

    context = load_dataset_context(DATA_DIR, dataset_id)

    # Try code-based analysis if agent is enabled
    if settings.agent_enabled:
        try:
            from .code_analysis import analyze_dataset_with_code

            # Load dataset understanding
            try:
                understanding_path = DATA_DIR / dataset_id / "understanding.json"
                if understanding_path.exists():
                    understanding = json.loads(understanding_path.read_text())
                else:
                    raise FileNotFoundError()
            except (FileNotFoundError, json.JSONDecodeError):
                # If no understanding yet, create minimal one
                understanding = {
                    "summary": {
                        "name": metadata.file_name,
                        "rowCount": len(df),
                        "columnCount": len(df.columns),
                        "description": "Dataset uploaded for analysis",
                    },
                    "columns": [],
                }

            # Run code-based analysis
            logger.info(f"Running code-based analysis for {dataset_id}")
            analysis_result = await analyze_dataset_with_code(
                dataset_id=dataset_id,
                df=df,
                dataset_understanding=understanding,
                user_instructions=context.get("instructions", ""),
            )

            # Convert to expected response format
            issues = [
                IssueResponse(**issue) for issue in analysis_result.get("issues", [])
            ]
            result = AnalysisResultResponse(
                dataset_id=dataset_id,
                issues=issues,
                summary=analysis_result.get("summary", "Analysis complete"),
                completedAt=analysis_result.get(
                    "completedAt", datetime.now(timezone.utc).isoformat()
                ),
            )

            _persist_analysis_result(DATA_DIR / dataset_id, result)
            logger.info(f"Code-based analysis complete: {len(issues)} issues found")
            return result

        except Exception as e:
            logger.error(f"Code-based analysis failed: {e}, falling back to heuristics")
            # Fall through to heuristics

    # Fallback: rule-based backup analysis
    logger.info(f"Using backup rule-based analysis for {dataset_id}")
    backup_payload = run_backup_analysis(dataset_id, df)
    issues = [IssueResponse(**issue) for issue in backup_payload["issues"]]
    result = AnalysisResultResponse(
        dataset_id=dataset_id,
        issues=issues,
        summary=backup_payload["summary"],
        completedAt=backup_payload["completedAt"],
    )

    _persist_analysis_result(DATA_DIR / dataset_id, result)
    return result


@app.post("/datasets/{dataset_id}/analysis", response_model=AnalysisResultResponse)
async def analyze_dataset_endpoint(dataset_id: str):
    try:
        load_dataset_metadata(DATA_DIR, dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return await _run_dataset_analysis(dataset_id)


@app.get("/datasets/{dataset_id}/analysis", response_model=AnalysisResultResponse)
def get_dataset_analysis(dataset_id: str):
    try:
        load_dataset_metadata(DATA_DIR, dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        return _load_analysis_result(dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Aucune analyse disponible pour ce dataset. Lancez une analyse d'abord.",
        ) from exc


@app.post("/datasets/{dataset_id}/apply", response_model=ApplyIssuesResponse)
def apply_dataset_changes(dataset_id: str, payload: ApplyIssuesRequest):
    try:
        load_dataset_metadata(DATA_DIR, dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not payload.issueIds:
        raise HTTPException(status_code=400, detail="Aucun issueId fourni.")

    try:
        analysis = _load_analysis_result(dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    applied_before = set(_load_applied_issues(dataset_id))
    available_ids = {issue.id for issue in analysis.issues}

    applied_now: List[str] = []
    skipped: List[str] = []

    for issue_id in payload.issueIds:
        if issue_id not in available_ids:
            skipped.append(issue_id)
            continue
        if issue_id in applied_before:
            skipped.append(issue_id)
            continue
        applied_before.add(issue_id)
        applied_now.append(issue_id)

    _persist_applied_issues(dataset_id, list(applied_before))

    message = f"Applied {len(applied_now)} issues; {len(skipped)} skipped"

    return ApplyIssuesResponse(
        dataset_id=dataset_id,
        applied=applied_now,
        skipped=skipped,
        message=message,
    )


@app.post("/datasets/{dataset_id}/smart-fix", response_model=SmartFixResponse)
def submit_smart_fix_response(dataset_id: str, payload: SmartFixRequest):
    try:
        load_dataset_metadata(DATA_DIR, dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not payload.issueId or not payload.response.strip():
        raise HTTPException(status_code=400, detail="issueId et response sont requis.")

    try:
        analysis = _load_analysis_result(dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    issue_ids = {
        issue.id for issue in analysis.issues if issue.category == "smart_fixes"
    }
    if payload.issueId not in issue_ids:
        raise HTTPException(
            status_code=400, detail="issueId inconnu ou non éligible aux smart fixes."
        )

    saved = _persist_smart_fix_response(
        dataset_id, payload.issueId, payload.response.strip()
    )
    return saved


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    """
    Chat simple sans fichier Excel.
    L'historique est stocké dans MongoDB.
    """
    reply = chat_with_user(
        session_id=payload.session_id,
        user_message=payload.message,
        excel_context=None,
    )
    return ChatResponse(reply=reply)


@app.post("/chat_excel", response_model=ChatResponse)
async def chat_excel(
    session_id: str = Form(...),
    message: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Chat avec contexte Excel.

    - Le fichier Excel est lu à chaque requête
    - Son contenu est transformé en texte
    - Ce texte est passé comme contexte au LLM
    - L'historique de la conversation est stocké dans MongoDB
    """
    if not file.filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(
            status_code=400,
            detail="Le fichier doit être un Excel (.xlsx ou .xls).",
        )

    file_bytes = await file.read()
    excel_context = build_excel_context(file_bytes, filename=file.filename)

    reply = chat_with_user(
        session_id=session_id,
        user_message=message,
        excel_context=excel_context,
    )

    return ChatResponse(reply=reply)


@app.post("/upload_and_analyze")
async def upload_and_analyze(
    dataset_id: str = Form(...),
    file: UploadFile = File(...),
    file_type: str = Form("excel"),  # "excel" ou "csv"
    delimiter: str = Form(","),
    meta_data: str = Form(""),
):
    """
    Upload le dataset ET lance l'analyse en une seule étape.

    Étapes :
    1) Sauvegarde le fichier dans /data/{dataset_id}.{ext}
    2) Charge le fichier avec pandas pour récupérer colonnes + metadata
    3) Appelle le tool LLM generate_error_analysis_script pour générer un script Python
    4) Sauvegarde ce script dans /scripts/detect_errors_{dataset_id}.py
    5) Exécute le script sur le fichier
    6) Sauvegarde le rapport d'erreurs dans /data/{dataset_id}_errors.txt
    7) Retourne le rapport d'erreurs + chemins utiles
    """

    suffix = Path(file.filename).suffix.lower()
    if suffix not in [".xlsx", ".xls", ".csv"]:
        raise HTTPException(
            status_code=400, detail="Le fichier doit être un Excel ou un CSV."
        )

    # 1) Sauvegarder le fichier
    dataset_path = DATA_DIR / f"{dataset_id}{suffix}"
    print(f"Sauvegarde du fichier dataset à : {dataset_path}")
    file_bytes = await file.read()
    dataset_path.write_bytes(file_bytes)

    # 2) Charger avec pandas pour récupérer les colonnes + metadata
    if suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(dataset_path)
        effective_file_type = "excel"
    else:
        df = pd.read_csv(dataset_path, delimiter=delimiter)
        effective_file_type = "csv"

    column_names = df.columns.tolist()

    metadata = (
        f"Dataset '{dataset_id}' - shape: {df.shape[0]} rows x {df.shape[1]} columns.\n"
        f"Premières lignes (head):\n{df.head(5).to_markdown(index=False)}",
        f"\nMetadata additionnelle fournie : {meta_data}" if meta_data else "",
    )

    # 3) Appeler le tool LLM pour générer le script d'analyse
    script_code = generate_error_analysis_script(
        column_names,
        metadata,
        effective_file_type,
        delimiter,
    )

    # 4) Sauvegarder le script généré
    script_path = SCRIPTS_DIR / f"detect_errors_{dataset_id}.py"
    script_path.write_text(script_code, encoding="utf-8")

    # 5) Exécuter le script sur le dataset
    result = subprocess.run(
        [sys.executable, str(script_path), str(dataset_path)],
        capture_output=True,
        text=True,
        cwd=BASE_DIR,
    )

    error_report = result.stdout
    if not error_report.strip() and result.stderr:
        error_report = "SCRIPT RUNTIME ERROR:\n" + result.stderr

    formatted_report = None
    try:
        formatted_report = format_error_report_to_json(dataset_id, error_report)
    except Exception as exc:  # pragma: no cover - best-effort formatting
        logger.warning("Failed to format error report with LLM: %s", exc)

    # 6) Sauvegarder le rapport d'erreurs
    error_report_path = DATA_DIR / f"{dataset_id}_errors.txt"
    error_report_path.write_text(error_report, encoding="utf-8")

    # 7) Retourner infos
    return {
        "status": "ok",
        "dataset_id": dataset_id,
        "dataset_path": str(dataset_path),
        "analysis_script_path": str(script_path),
        "error_report_path": str(error_report_path),
        "error_report": error_report,
        "analysis_json": formatted_report,
    }


@app.post("/chat_dataset", response_model=ChatResponse)
def chat_dataset(payload: ChatDatasetRequest):
    """
    Discute avec l'IA à propos d'un dataset précis (dataset_id).
    Utilise le rapport d'erreurs texte comme contexte.
    """

    error_report_path = DATA_DIR / f"{payload.dataset_id}_errors.txt"
    if not error_report_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Aucun rapport d'erreurs trouvé pour ce dataset_id. "
            "Appelle d'abord /upload_and_analyze.",
        )

    dataset_path = DATA_DIR / f"{payload.dataset_id}.csv"
    if not dataset_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Aucun fichier dataset CSV trouvé pour ce dataset_id.",
        )

    # 2) Charger avec pandas pour récupérer les colonnes + metadata
    df = pd.read_csv(dataset_path)
    effective_file_type = "csv"

    column_names = df.columns.tolist()

    error_report = error_report_path.read_text(encoding="utf-8")

    meta_data = f"""
    Metadata du dataset '{payload.dataset_id}':
        metadata = (
        f"Dataset '{payload.dataset_id}' - shape: {df.shape[0]} rows x {df.shape[1]} columns.\n"
        f"Premières lignes (head):\n{df.head(5).to_markdown(index=False)}",
    )

    Voici le rapport d'erreurs détectées dans le dataset :
    {error_report}
    """

    reply = chat_with_user(
        session_id=payload.session_id,
        user_message=payload.message,
        excel_context=error_report,  # on passe le rapport comme contexte
    )

    return ChatResponse(reply=reply)


# ============================================================================
# AGENT TEST ENDPOINT (Step 2: Agent Execution Sandbox)
# ============================================================================


class AgentTestRequest(BaseModel):
    """Request model for agent testing endpoint."""

    file_content: Optional[str] = None
    sample_csv_rows: Optional[int] = 10


class AgentTestResponse(BaseModel):
    """Response model for agent testing endpoint."""

    success: bool
    agent_enabled: bool
    execution_time_seconds: float
    understanding: Optional[Dict[str, Any]] = None
    analysis: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@app.post("/agent/test", response_model=AgentTestResponse)
async def test_agent_execution(request: AgentTestRequest = AgentTestRequest()):
    """
    Prototype endpoint to test agent execution on a small dataset.

    This endpoint demonstrates the agent sandbox with:
    - Timeout enforcement
    - Retry logic
    - JSON validation
    - Structured error handling

    Returns agent-generated dataset understanding and analysis.
    """
    import time
    from .agent import (
        generate_dataset_understanding,
        generate_analysis_issues,
    )
    from .config import settings

    start_time = time.time()

    # Check if agent is enabled
    if not settings.agent_enabled:
        return AgentTestResponse(
            success=False,
            agent_enabled=False,
            execution_time_seconds=time.time() - start_time,
            error="Agent is disabled via AGENT_ENABLED flag",
        )

    try:
        # Use provided CSV or create a small sample dataset
        if request.file_content:
            # Parse user-provided CSV content
            import io

            df = pd.read_csv(io.StringIO(request.file_content))
        else:
            # Create a tiny sample dataset for testing
            sample_data = {
                "id": [1, 2, 3, 4, 5],
                "name": ["Alice", "Bob", None, "David", "Eve"],
                "age": [25, 30, 35, 30, 28],
                "department": ["Sales", "Engineering", "Sales", "Engineering", "HR"],
                "salary": [50000, 75000, 60000, 75000, 55000],
            }
            df = pd.DataFrame(sample_data)

        # Limit rows for testing
        max_rows = min(request.sample_csv_rows or 10, len(df))
        df = df.head(max_rows)

        # Prepare inputs for agent
        dataset_id = "test_dataset_001"
        file_name = "test_sample.csv"
        row_count = len(df)
        column_count = len(df.columns)

        # Get sample rows
        sample_rows = df.head(5).to_dict(orient="records")

        # Build column summaries
        column_summaries = []
        for col in df.columns:
            col_summary = {
                "name": col,
                "inferred_type": str(df[col].dtype),
                "sample_values": df[col].dropna().head(3).astype(str).tolist(),
                "missing_count": int(df[col].isna().sum()),
            }
            column_summaries.append(col_summary)

        # Step 1: Generate dataset understanding
        understanding = await generate_dataset_understanding(
            dataset_id=dataset_id,
            file_name=file_name,
            row_count=row_count,
            column_count=column_count,
            sample_rows=sample_rows,
            column_summaries=column_summaries,
            user_instructions="This is a test dataset for agent validation",
        )

        # Step 2: Generate analysis issues
        analysis = await generate_analysis_issues(
            dataset_id=dataset_id,
            dataset_understanding=understanding.model_dump(),
            user_instructions="Identify any data quality issues",
        )

        execution_time = time.time() - start_time

        return AgentTestResponse(
            success=True,
            agent_enabled=True,
            execution_time_seconds=round(execution_time, 2),
            understanding=understanding.model_dump(),
            analysis=analysis.model_dump(),
        )

    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Agent test failed: {e}")

        return AgentTestResponse(
            success=False,
            agent_enabled=settings.agent_enabled,
            execution_time_seconds=round(execution_time, 2),
            error=str(e),
        )


@app.get("/health")
def health():
    dataset_dirs = [p for p in DATA_DIR.iterdir() if p.is_dir()]
    return {
        "status": "up",
        "datasets": len(dataset_dirs),
        "latest_dataset": max((p.stat().st_mtime for p in dataset_dirs), default=None),
    }
