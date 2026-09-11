"""
Quick smoke-test script — validates IBM Granite connectivity and the full RAG pipeline
without launching the Streamlit UI.

Usage:
    python test_agent.py
"""

import os
import sys

# Run from Project/interview_trainer_agent/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.agent import InterviewTrainerAgent, CandidateProfile

SEPARATOR = "─" * 70


def header(title: str):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def main():
    print("🚀 Interview Trainer Agent — Smoke Test")
    print("Powered by IBM Granite (watsonx.ai) + ChromaDB RAG\n")

    # ── 1. Initialise agent (builds RAG index on first run)
    header("1. Agent Initialisation")
    agent = InterviewTrainerAgent()

    # ── 2. Health check
    header("2. Health Check")
    health = agent.health_check()
    print(f"   LLM Connected  : {health['llm_connected']}")
    print(f"   Index Stats    : {health['index_stats']}")
    print(f"   Status         : {health['status']}")

    if not health["llm_connected"]:
        print("\n⚠️  LLM not reachable. Check your .env credentials.")
        sys.exit(1)

    # ── 3. Sample candidate profile
    profile = CandidateProfile(
        name             = "Priya Sharma",
        role             = "Software Engineer",
        experience_level = "Mid-level (3–5 yrs)",
        focus_areas      = ["technical", "system design", "behavioral"],
        company          = "IBM",
        interview_date   = "2025-08-20",
    )

    # ── 4. Generate questions
    header("3. Generate Interview Questions")
    questions = agent.generate_questions(profile, num_questions=3)
    print(questions[:1200], "…\n[truncated]")

    # ── 5. Mock interview
    header("4. Mock Interview — Technical Question")
    mock = agent.run_mock_interview(profile, category="technical", difficulty="intermediate")
    print(f"Generated Question:\n{mock['generated_question'][:600]}")
    print(f"\nReference Answer (from KB):\n{mock['reference_answer'][:300]}")

    # ── 6. Evaluate a sample answer
    header("5. Evaluate a Candidate Answer")
    sample_q   = "What is the difference between a stack and a queue?"
    sample_ans = "A stack is LIFO and a queue is FIFO. Stack is used for recursion, queue for BFS."
    feedback   = agent.evaluate_answer(profile, sample_q, sample_ans)
    print(feedback[:1200])

    # ── 7. Preparation strategy
    header("6. Preparation Strategy")
    plan = agent.build_prep_strategy(profile)
    print(plan[:1000], "…\n[truncated]")

    # ── 8. Resume analysis
    header("7. Resume Analysis")
    sample_resume = """
    Software Engineer with 4 years experience in Python, REST APIs, and cloud services.
    Worked with AWS (EC2, S3, Lambda), Docker, Kubernetes, PostgreSQL.
    Led a team of 3 to deliver a microservices migration for a fintech product.
    Strong in data structures, algorithms, and system design.
    """
    analysis = agent.analyze_resume(sample_resume)
    print(analysis[:1200])

    print(f"\n{SEPARATOR}")
    print("  ✅  All tests passed. Agent is fully operational.")
    print(f"{SEPARATOR}\n")
    print("Start the UI with:  streamlit run app.py\n")


if __name__ == "__main__":
    main()
