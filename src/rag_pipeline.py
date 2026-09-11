"""
RAG Pipeline — Vector Store + Retriever
Loads the interview knowledge base JSON files, embeds them with
sentence-transformers, stores in a persistent ChromaDB collection,
and exposes a retrieve() method used by the agent.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE_DIR = Path(__file__).parent.parent / "data" / "knowledge_base"
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./vector_store/chroma_db")
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"          # fast, 384-dim, runs fully locally
COLLECTION_NAME    = "interview_questions"
MAX_RETRIEVED_DOCS = int(os.getenv("MAX_RETRIEVED_DOCS", "5"))


class InterviewRetriever:
    """
    Loads all JSON knowledge-base files into ChromaDB and retrieves
    semantically similar documents for a given query.

    Usage:
        retriever = InterviewRetriever()
        retriever.build_index()           # once, or on startup
        docs = retriever.retrieve(query, role="Software Engineer", n_results=4)
    """

    def __init__(self):
        self.embedder  = SentenceTransformer(EMBEDDING_MODEL)
        self.chroma    = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------
    def build_index(self, force_rebuild: bool = False) -> int:
        """
        Load all JSON files from the knowledge base directory and upsert
        into ChromaDB. Returns the total number of documents indexed.
        """
        if not force_rebuild and self.collection.count() > 0:
            print(f"[Retriever] Index already contains {self.collection.count()} docs. Skipping rebuild.")
            return self.collection.count()

        print("[Retriever] Building vector index from knowledge base …")
        all_docs: List[Dict[str, Any]] = []

        for json_file in KNOWLEDGE_BASE_DIR.glob("*.json"):
            with open(json_file, "r", encoding="utf-8") as f:
                docs = json.load(f)
                all_docs.extend(docs)

        if not all_docs:
            raise FileNotFoundError(f"No JSON files found in {KNOWLEDGE_BASE_DIR}")

        ids, texts, metadatas = [], [], []

        for doc in all_docs:
            # Build rich text for embedding – question + key points
            text = (
                f"Role: {doc.get('role', 'All')}\n"
                f"Category: {doc.get('category', '')}\n"
                f"Question: {doc.get('question', '')}\n"
                f"Key Points: {', '.join(doc.get('key_points', []))}"
            )
            ids.append(doc["id"])
            texts.append(text)
            metadatas.append({
                "role":       doc.get("role", "All"),
                "category":   doc.get("category", ""),
                "difficulty": doc.get("difficulty", ""),
                "question":   doc.get("question", ""),
                "model_answer":     doc.get("model_answer", ""),
                "improvement_tip":  doc.get("improvement_tip", ""),
            })

        embeddings = self.embedder.encode(texts, show_progress_bar=True).tolist()

        # Upsert in one shot
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        print(f"[Retriever] Indexed {len(all_docs)} documents.")
        return len(all_docs)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        role: Optional[str] = None,
        category: Optional[str] = None,
        n_results: int = MAX_RETRIEVED_DOCS,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-n most semantically similar interview questions.

        Args:
            query:    User's query (e.g., job role, topic, raw question)
            role:     Filter by role (e.g., 'Software Engineer')
            category: Filter by category ('technical', 'behavioral', 'hr')
            n_results: Number of results to return

        Returns:
            List of metadata dicts with question, model_answer, improvement_tip.
        """
        query_embedding = self.embedder.encode(query).tolist()

        where_filter: Optional[Dict] = None
        if role and category:
            where_filter = {"$and": [
                {"role":     {"$in": [role, "All"]}},
                {"category": {"$eq": category}},
            ]}
        elif role:
            where_filter = {"role": {"$in": [role, "All"]}}
        elif category:
            where_filter = {"category": {"$eq": category}}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["metadatas", "distances"],
        )

        retrieved = []
        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
            retrieved.append({
                "question":        meta["question"],
                "model_answer":    meta["model_answer"],
                "improvement_tip": meta["improvement_tip"],
                "category":        meta["category"],
                "difficulty":      meta["difficulty"],
                "similarity":      round(1 - dist, 4),   # cosine distance → similarity
            })

        return retrieved

    def get_index_stats(self) -> Dict[str, int]:
        """Return a summary of indexed document counts."""
        count = self.collection.count()
        return {"total_documents": count, "collection": COLLECTION_NAME}
