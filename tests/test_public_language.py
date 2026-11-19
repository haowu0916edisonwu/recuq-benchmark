from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_release_manifest import release_files


def public_text() -> str:
    paths = [ROOT / "README.md", ROOT / "DATA_CARD.md", ROOT / "REPRODUCIBILITY.md"]
    paths.extend((ROOT / "docs").glob("*.md"))
    return "\n".join(path.read_text(encoding="utf-8") for path in paths).lower()


def test_final_paper_terminology_is_present() -> None:
    text = public_text()
    assert "ranking-failure detection" in text
    assert "tail-target subset" in text
    assert "paired-control response" in text
    assert "operational dominant control" in text
    assert "not causal attribution" in text or "no causal attribution" in text


def test_release_docs_avoid_causal_ground_truth_claims() -> None:
    text = public_text()
    assert "scalar reliability detection" not in text
    assert "ground-truth source" not in text
    assert "true source" not in text
    assert "true long-tail" not in text


def test_no_legacy_project_name_or_chinese_text() -> None:
    legacy_name = "unc" + "rec"
    text_suffixes = {".py", ".md", ".yaml", ".yml", ".json", ".csv", ".txt", ".toml"}
    for path in release_files(ROOT):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        text = path.read_text(encoding="utf-8", errors="strict")
        assert legacy_name not in text.lower(), path
        assert not any("\u3400" <= char <= "\u9fff" for char in text), path


def test_no_absolute_workstation_paths_in_text_release() -> None:
    excluded = {
        ROOT / "tests/test_public_language.py",
        ROOT / "scripts/validate_release.py",
    }
    for path in release_files(ROOT):
        if not path.is_file() or path in excluded:
            continue
        if path.suffix.lower() not in {".py", ".md", ".yaml", ".yml", ".json", ".csv", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        assert "/nvme/" not in text, path
        assert "/data_nvme/" not in text, path
