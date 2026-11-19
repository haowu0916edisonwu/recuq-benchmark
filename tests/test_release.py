from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_full_release_validator(tmp_path: Path) -> None:
    report = tmp_path / "validation.json"
    subprocess.run(
        [sys.executable, "scripts/validate_release.py", "--output", str(report)],
        cwd=ROOT,
        check=True,
    )
    payload = json.loads(report.read_text())
    assert payload["status"] == "passed"
    assert payload["base_artifacts"]["llama"]["rows"] == 45163
    assert payload["base_artifacts"]["llama"]["eligible_rows"] == 42475
    assert payload["base_artifacts"]["qwen"]["rows"] == 4000
    assert len(payload["uq_artifacts"]) == 16
