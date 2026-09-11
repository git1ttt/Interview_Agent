"""
Document Loader — ingests PDF, DOCX, and plain-text resumes / job descriptions
into the ChromaDB RAG vector store so Granite can retrieve from them.

Supports:
  - PDF  (PyPDF2)
  - DOCX (python-docx)
  - TXT  (plain text)
  - Raw string (e.g. pasted text from Streamlit textarea)

Usage:
    loader = DocumentLoader()
    loader.ingest_file("resume.pdf",  source_label="Priya_Resume")
    loader.ingest_text("5 years Python …", source_label="JD_SWE_IBM")
    docs = loader.search("distributed systems experience")
"""

from __future__ import annotations

import hashlib
import os
import re
import textwrap
from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Optional PDF / DOCX — imported lazily so missing packages raise only on use
try:
    import PyPDF2
    _HAS_PDF = True
except ImportError:
    _HAS_PDF = False

try:
    import docx as python_docx
    _HAS_DOCX = True
except ImportError:
    _HAS_DOCX = False

from dotenv import load_dotenv
load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./vector_store/chroma_db")
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"
DOCS_COLLECTION    = "user_documents"
CHUNK_SIZE         = 400    # characters per chunk
CHUNK_OVERLAP      = 80


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping character-level chunks."""
    text   = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start  = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def _stable_id(source: str, chunk_index: int) -> str:
    """Deterministic chunk ID so re-ingesting the same file is idempotent."""
    raw = f"{source}::chunk_{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


class DocumentLoader:
    """
    Ingests user-supplied documents into a dedicated ChromaDB collection
    separate from the curated knowledge base.
    """

    def __init__(self):
        self.embedder   = SentenceTransformer(EMBEDDING_MODEL)
        self.chroma     = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma.get_or_create_collection(
            name=DOCS_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Public ingestion methods
    # ------------------------------------------------------------------
    def ingest_file(self, file_path: str, source_label: Optional[str] = None) -> int:
        """
        Parse and embed a PDF, DOCX, or TXT file.
        Returns the number of chunks ingested.
        """
        path  = Path(file_path)
        label = source_label or path.stem
        ext   = path.suffix.lower()

        if ext == ".pdf":
            text = self._read_pdf(path)
        elif ext in (".docx", ".doc"):
            text = self._read_docx(path)
        elif ext == ".txt":
            text = path.read_text(encoding="utf-8", errors="ignore")
        else:
            raise ValueError(f"Unsupported file type: {ext}. Use PDF, DOCX, or TXT.")

        return self._ingest_text_internal(text, label)

    def ingest_text(self, text: str, source_label: str = "pasted_text") -> int:
        """Ingest raw text string (e.g. from a Streamlit textarea)."""
        return self._ingest_text_internal(text, source_label)

    def ingest_bytes(self, file_bytes: bytes, filename: str, source_label: Optional[str] = None) -> int:
        """
        Ingest from bytes (Streamlit UploadedFile.read()).
        Writes a temp file, ingests, then removes it.
        """
        import tempfile
        suffix = Path(filename).suffix
        label  = source_label or Path(filename).stem

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            return self.ingest_file(tmp_path, label)
        finally:
            os.unlink(tmp_path)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def search(self, query: str, n_results: int = 4) -> List[dict]:
        """Semantic search over ingested user documents."""
        if self.collection.count() == 0:
            return []

        q_emb    = self.embedder.encode(query).tolist()
        results  = self.collection.query(
            query_embeddings=[q_emb],
            n_results=min(n_results, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        out = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            out.append({
                "text":       doc,
                "source":     meta.get("source", ""),
                "chunk":      meta.get("chunk_index", 0),
                "similarity": round(1 - dist, 4),
            })
        return out

    def list_sources(self) -> List[str]:
        """Return distinct source labels that have been ingested."""
        if self.collection.count() == 0:
            return []
        results = self.collection.get(include=["metadatas"])
        return sorted({m.get("source", "") for m in results["metadatas"]})

    def clear_source(self, source_label: str) -> None:
        """Remove all chunks for a given source label."""
        results = self.collection.get(
            where={"source": {"$eq": source_label}},
            include=["metadatas"],
        )
        ids = results.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)
            print(f"[DocumentLoader] Removed {len(ids)} chunks for source '{source_label}'.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ingest_text_internal(self, text: str, label: str) -> int:
        chunks     = _chunk_text(text)
        embeddings = self.embedder.encode(chunks, show_progress_bar=False).tolist()
        ids        = [_stable_id(label, i) for i in range(len(chunks))]
        metadatas  = [{"source": label, "chunk_index": i} for i in range(len(chunks))]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        print(f"[DocumentLoader] Ingested {len(chunks)} chunks from '{label}'.")
        return len(chunks)

    def _read_pdf(self, path: Path) -> str:
        if not _HAS_PDF:
            raise ImportError("PyPDF2 not installed. Run: pip install PyPDF2")
        text = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text.append(page.extract_text() or "")
        return "\n".join(text)

    def _read_docx(self, path: Path) -> str:
        if not _HAS_DOCX:
            raise ImportError("python-docx not installed. Run: pip install python-docx")
        doc   = python_docx.Document(str(path))
        paras = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paras)
