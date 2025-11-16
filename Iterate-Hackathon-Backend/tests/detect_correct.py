import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from app.backup_analysis import run_backup_analysis  # noqa: E402


def load_dataset(file_path: Path) -> pd.DataFrame:
    if file_path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    return pd.read_csv(file_path, delimiter=",")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the rule-based backup analysis on a dataset and emit frontend-ready JSON.",
    )
    parser.add_argument("dataset_path", type=Path, help="Path to the CSV/XLSX dataset")
    parser.add_argument(
        "--dataset-id",
        dest="dataset_id",
        help="Optional dataset identifier (defaults to file stem)",
    )
    parser.add_argument(
        "--output",
        dest="output",
        type=Path,
        help="Optional path to write the JSON payload (defaults to stdout only)",
    )
    args = parser.parse_args()

    dataset_path: Path = args.dataset_path
    if not dataset_path.exists():
        print(f"File not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    dataset_id = args.dataset_id or dataset_path.stem

    try:
        df = load_dataset(dataset_path)
    except Exception as exc:  # pragma: no cover - CLI guard
        print(f"Unable to load dataset: {exc}", file=sys.stderr)
        sys.exit(1)

    payload = run_backup_analysis(dataset_id, df)
    output_text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(output_text)

    if args.output:
        args.output.write_text(output_text, encoding="utf-8")


if __name__ == "__main__":
    main()
