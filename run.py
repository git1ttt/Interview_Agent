"""
run.py — Convenience launcher for the Interview Trainer Agent.

Usage:
    python run.py            → launch Streamlit UI
    python run.py --test     → run smoke test (CLI, no UI)
    python run.py --index    → rebuild RAG vector index only
    python run.py --health   → check IBM Granite connectivity
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_ui():
    """Launch the Streamlit app."""
    os.system("streamlit run app.py")


def run_test():
    """Run the CLI smoke test."""
    import test_agent
    test_agent.main()


def run_index():
    """Force-rebuild the ChromaDB vector index from the knowledge base."""
    from dotenv import load_dotenv
    load_dotenv()
    from src.rag_pipeline import InterviewRetriever
    retriever = InterviewRetriever()
    n = retriever.build_index(force_rebuild=True)
    print(f"\n✅ Index rebuilt: {n} documents.")


def run_health():
    """Check IBM Granite connectivity and index stats."""
    from dotenv import load_dotenv
    load_dotenv()

    print("Checking IBM Granite (watsonx.ai) connectivity …\n")
    from src.granite_llm import GraniteLLM
    llm = GraniteLLM()
    ok  = llm.health_check()
    print(f"  IBM Granite: {'✅ Connected' if ok else '❌ Not reachable'}")

    from src.rag_pipeline import InterviewRetriever
    retriever = InterviewRetriever()
    stats     = retriever.get_index_stats()
    print(f"  RAG Index  : {stats['total_documents']} documents in '{stats['collection']}'")

    from src.document_loader import DocumentLoader
    doc_loader = DocumentLoader()
    sources    = doc_loader.list_sources()
    print(f"  User Docs  : {len(sources)} source(s) — {', '.join(sources) if sources else 'none ingested yet'}")

    print()


def print_help():
    print("""
Interview Trainer Agent — Launcher
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  python run.py            Launch Streamlit UI
  python run.py --test     CLI smoke test (validates full pipeline)
  python run.py --index    Force-rebuild RAG vector index
  python run.py --health   Check IBM Granite + index connectivity
  python run.py --help     Show this message
""")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""

    if arg == "--test":
        run_test()
    elif arg == "--index":
        run_index()
    elif arg == "--health":
        run_health()
    elif arg in ("--help", "-h"):
        print_help()
    else:
        run_ui()
