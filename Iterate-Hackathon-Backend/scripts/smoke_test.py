"""Simple smoke test hitting the key backend endpoints."""

from __future__ import annotations

import json
import pathlib
import sys
import requests

BASE_URL = "http://127.0.0.1:8000"


def main(dataset_path: str) -> None:
    file = pathlib.Path(dataset_path)
    if not file.exists():
        raise SystemExit(f"File {dataset_path} not found")

    print("[1] Health check...", end=" ")
    resp = requests.get(f"{BASE_URL}/health", timeout=10)
    resp.raise_for_status()
    print("ok")

    print("[2] Upload dataset...", end=" ")
    with file.open("rb") as fh:
        resp = requests.post(
            f"{BASE_URL}/datasets",
            files={"file": (file.name, fh, "text/csv")},
            timeout=60,
        )
    resp.raise_for_status()
    dataset_id = resp.json()["dataset_id"]
    print(dataset_id)

    print("[3] Understanding...", end=" ")
    resp = requests.get(f"{BASE_URL}/datasets/{dataset_id}/understanding", timeout=30)
    resp.raise_for_status()
    print("ok")

    print("[4] Save context...", end=" ")
    resp = requests.post(
        f"{BASE_URL}/datasets/{dataset_id}/context",
        json={"instructions": "Smoke test context"},
        timeout=10,
    )
    resp.raise_for_status()
    print("ok")

    print("[5] Analysis...")
    resp = requests.post(f"{BASE_URL}/datasets/{dataset_id}/analysis", timeout=60)
    resp.raise_for_status()
    analysis = resp.json()
    issues = [issue["id"] for issue in analysis.get("issues", [])]
    print(f"    Found {len(issues)} issues")

    if issues:
        print("[6] Apply first issue...", end=" ")
        resp = requests.post(
            f"{BASE_URL}/datasets/{dataset_id}/apply",
            json={"issueIds": issues[:1]},
            timeout=10,
        )
        resp.raise_for_status()
        print("ok")

    smart_issue = next((issue for issue in analysis.get("issues", []) if issue["category"] == "smart_fixes"), None)
    if smart_issue:
        print("[7] Smart fix response...", end=" ")
        resp = requests.post(
            f"{BASE_URL}/datasets/{dataset_id}/smart-fix",
            json={"issueId": smart_issue["id"], "response": "intentional"},
            timeout=10,
        )
        resp.raise_for_status()
        print("ok")

    print("Smoke test completed successfully.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/smoke_test.py <dataset.csv>")
    main(sys.argv[1])
