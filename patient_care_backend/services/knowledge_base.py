"""Index curated PDF documents into Pinecone for LangGraph retrieval."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid5, NAMESPACE_URL

if TYPE_CHECKING:
    from patient_care_backend.config import Settings

SUPPORTED_SUFFIXES = {".pdf"}
MAX_CHARS_PER_CHUNK = 1_800
CHUNK_OVERLAP = 180
MAX_FILE_SIZE_BYTES = 75 * 1024 * 1024


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunk_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> list[str]:
    text = _clean_text(text)
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        split_at = text.rfind("\n\n", start, end)
        if split_at <= start + max_chars // 2:
            split_at = text.rfind(". ", start, end)
        if split_at <= start:
            split_at = end

        chunk = text[start:split_at].strip()
        if chunk:
            chunks.append(chunk)

        if split_at >= len(text):
            break

        next_start = max(0, split_at - CHUNK_OVERLAP)
        if next_start <= start:
            next_start = split_at
        start = next_start

    return chunks


def _safe_document_paths(source_dir: Path) -> list[Path]:
    root = source_dir.resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Knowledge source directory not found: {root}")

    paths: list[Path] = []
    for path in sorted(root.iterdir()):
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            continue
        if not resolved.is_file() or resolved.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if resolved.stat().st_size > MAX_FILE_SIZE_BYTES:
            raise ValueError(f"PDF is too large to index safely: {resolved.name}")
        paths.append(resolved)
    return paths


def extract_pdf_chunks(path: Path) -> list[dict[str, Any]]:
    import fitz  # PyMuPDF

    chunks: list[dict[str, Any]] = []
    
    try:
        doc = fitz.open(str(path))
        for page_index, page in enumerate(doc, start=1):
            page_text = page.get_text() or ""
            for chunk_index, text in enumerate(_chunk_text(page_text), start=1):
                chunks.append(
                    {
                        "source": path.name,
                        "page": page_index,
                        "chunk": chunk_index,
                        "text": text,
                    }
                )
    except Exception as e:
        print(f"Error reading {path.name}: {e}")
    finally:
        if 'doc' in locals():
            doc.close()

    return chunks


def build_knowledge_chunks(source_dir: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for path in _safe_document_paths(source_dir):
        chunks.extend(extract_pdf_chunks(path))
    return chunks


class KnowledgeBaseIndexer:
    def __init__(self, settings) -> None:
        self._settings = settings
        self._index = None
        self._embeddings = None

    def _connect(self) -> None:
        if self._index is not None and self._embeddings is not None:
            return

        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from pinecone import Pinecone

        pc = Pinecone(api_key=self._settings.pinecone_api_key)
        self._index = pc.Index(self._settings.pinecone_index_name)
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=self._settings.gemini_embedding_model,
            google_api_key=self._settings.google_api_key,
        )

    def index_directory(self, source_dir: Path) -> dict[str, Any]:
        chunks = build_knowledge_chunks(source_dir)
        documents = sorted({chunk["source"] for chunk in chunks})

        if not (
            self._settings.enable_external_ai
            and self._settings.pinecone_api_key
            and self._settings.google_api_key
        ):
            return {
                "indexed": False,
                "reason": "external_ai_not_configured",
                "document_count": len(documents),
                "chunk_count": len(chunks),
                "documents": documents,
                "preview": [chunk["text"][:240] for chunk in chunks[:3]],
            }

        self._connect()
        assert self._index is not None
        assert self._embeddings is not None

        import time
        vectors = []
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            texts = [c["text"] for c in batch_chunks]
            try:
                embeddings = self._embeddings.embed_documents(texts)
            except Exception as e:
                print(f"Embedding batch error, retrying: {e}")
                time.sleep(10)
                embeddings = self._embeddings.embed_documents(texts)
            
            for chunk, emb in zip(batch_chunks, embeddings):
                stable_id = uuid5(
                    NAMESPACE_URL,
                    f"{chunk['source']}:{chunk['page']}:{chunk['chunk']}:{chunk['text'][:120]}",
                ).hex
                vectors.append(
                    {
                        "id": f"kb-{stable_id}",
                        "values": emb,
                        "metadata": {
                            "scope": "knowledge_base",
                            "source": chunk["source"],
                            "page": chunk["page"],
                            "chunk": chunk["chunk"],
                            "text": chunk["text"],
                        },
                    }
                )
            time.sleep(1)

        if vectors:
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                self._index.upsert(vectors=vectors[i:i+batch_size])

        return {
            "indexed": True,
            "document_count": len(documents),
            "chunk_count": len(vectors),
            "documents": documents,
        }


def build_knowledge_base_indexer(settings: Settings) -> KnowledgeBaseIndexer:
    return KnowledgeBaseIndexer(settings)
