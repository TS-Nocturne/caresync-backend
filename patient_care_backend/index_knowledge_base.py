"""CLI entry point for indexing the curated pland PDF knowledge base."""

from pathlib import Path

from patient_care_backend.config import get_settings
from patient_care_backend.services.knowledge_base import build_knowledge_base_indexer


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    source_dir = repo_root / "pland"
    result = build_knowledge_base_indexer(get_settings()).index_directory(source_dir)
    print(result)


if __name__ == "__main__":
    main()
