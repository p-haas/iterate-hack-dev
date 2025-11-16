"""Utilities for persisting uploaded datasets and capturing metadata."""

from __future__ import annotations

import csv
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

ALLOWED_EXTENSIONS = {".csv", ".xls", ".xlsx"}
CSV_DELIMITERS = [",", ";", "\t", "|"]


def generate_dataset_id() -> str:
    """Return a unique dataset identifier."""

    return f"dataset_{uuid.uuid4().hex}"


def normalize_extension(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported file extension. Allowed: .csv, .xls, .xlsx")
    return suffix


def infer_file_type(extension: str) -> Literal["csv", "excel"]:
    return "csv" if extension == ".csv" else "excel"


def infer_delimiter(file_bytes: bytes, fallback: str = ",") -> str:
    """Try to guess delimiter for CSV files, fallback to comma if unknown."""

    try:
        sample = file_bytes[:4096].decode("utf-8", errors="ignore")
        if not sample.strip():
            return fallback
        dialect = csv.Sniffer().sniff(sample, delimiters=CSV_DELIMITERS)
        return dialect.delimiter
    except csv.Error:
        return fallback


def persist_dataset_file(
    data_dir: Path,
    dataset_id: str,
    original_filename: str,
    file_bytes: bytes,
    content_type: Optional[str],
    delimiter: Optional[str],
) -> dict:
    """Store the uploaded bytes under /data/{dataset_id}/raw.{ext} and write metadata."""

    extension = normalize_extension(original_filename)
    dataset_dir = data_dir / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)

    raw_path = dataset_dir / f"raw{extension}"
    raw_path.write_bytes(file_bytes)

    metadata = {
        "dataset_id": dataset_id,
        "original_filename": original_filename,
        "stored_file": str(raw_path.relative_to(data_dir.parent)),
        "raw_filename": raw_path.name,
        "extension": extension,
        "file_size_bytes": len(file_bytes),
        "file_type": infer_file_type(extension),
        "delimiter": delimiter if extension == ".csv" else None,
        "content_type": content_type,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }

    metadata_path = dataset_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return metadata


def load_dataset_metadata(data_dir: Path, dataset_id: str) -> dict:
    metadata_path = data_dir / dataset_id / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Aucun metadata.json pour dataset_id={dataset_id}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.setdefault("dataset_id", dataset_id)
    metadata.setdefault("raw_filename", f"raw{metadata.get('extension', '')}")
    metadata["_metadata_path"] = str(metadata_path)
    return metadata


def resolve_raw_path(data_dir: Path, metadata: dict) -> Path:
    dataset_id = metadata.get("dataset_id")
    dataset_dir = data_dir / dataset_id

    raw_filename = metadata.get("raw_filename")
    if raw_filename:
        candidate = dataset_dir / raw_filename
        if candidate.exists():
            return candidate

    stored_file = metadata.get("stored_file")
    if stored_file:
        stored_path = Path(stored_file)
        if not stored_path.is_absolute():
            stored_path = data_dir.parent / stored_path
        if stored_path.exists():
            return stored_path

    raise FileNotFoundError(f"Impossible de localiser le fichier brut pour dataset_id={dataset_id}")


def save_dataset_context(
    data_dir: Path,
    dataset_id: str,
    instructions: str,
    column_edits: Optional[Any] = None,
) -> dict:
    dataset_dir = data_dir / dataset_id
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset {dataset_id} introuvable. Upload requis avant de sauvegarder du contexte.")

    payload = {
        "dataset_id": dataset_id,
        "instructions": instructions,
        "column_edits": column_edits,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    context_path = dataset_dir / "context.json"
    context_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_dataset_context(data_dir: Path, dataset_id: str) -> Optional[dict]:
    context_path = data_dir / dataset_id / "context.json"
    if not context_path.exists():
        return None
    context = json.loads(context_path.read_text(encoding="utf-8"))
    context.setdefault("dataset_id", dataset_id)
    context.setdefault("instructions", "")
    return context


def dataset_dir_path(data_dir: Path, dataset_id: str) -> Path:
    path = data_dir / dataset_id
    if not path.exists():
        raise FileNotFoundError(f"Dataset {dataset_id} introuvable")
    return path
