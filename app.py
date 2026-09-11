"""
Interview Trainer Agent — Streamlit Frontend

All interactive UI is built with native Streamlit widgets only.
No raw HTML is used for interactive or content elements.

Sidebar:
  - Candidate profile (name, role, level, company, date, focus areas)
  - st.file_uploader for PDF / DOCX / TXT resumes → ChromaDB RAG ingestion

Tab 1 — Generate Questions
  - Calls agent.generate_questions() via RAG + IBM Granite
  - Each question is parsed into its own st.expander
  - Inside each expander: st.text_area + st.button("Submit Answer")
  - Submitting calls Evaluator → shows rubric score inline via native widgets

Tab 2 — Evaluate Answer (free-form, any question)
Tab 3 — Mock Interview  (live Q&A + session tracking)
Tab 4 — Prep Strategy   (day-by-day study plan)
Tab 5 — Resume Analyser (skills, level, predicted questions)
Tab 6 — Session History (score trend + category breakdown)

Run:
    streamlit run app.py
"""
import hashlib   #✅ ADD
import json
import re
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load .env from the same directory as app.py — works regardless of cwd
_APP_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=_APP_DIR / ".env", override=False)
load_dotenv(override=False)   # fallback: cwd

sys.path.insert(0, str(_APP_DIR))

from src.agent import InterviewTrainerAgent, CandidateProfile
from src.document_loader import DocumentLoader
from src.session_manager import SessionManager
from src.evaluator import Evaluator, EvaluationResult

# ─────────────────────────────────────────────────────────────────────────────
# Page config — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Interview Trainer Agent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# One CSS block — only sidebar background colour (no native Streamlit equivalent)
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f3460 0%, #16213e 100%);
    }
    section[data-testid="stSidebar"] * { color: #ffffff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper — display a clear, actionable IBM credential error in the UI
# ─────────────────────────────────────────────────────────────────────────────
def _show_ibm_error(exc: Exception) -> None:
    """
    Render a user-friendly error panel for any IBM watsonx.ai connection failure.
    Covers InvalidCredentialsError, network errors, and unexpected SDK errors.
    """
    exc_str = str(exc)
    is_cred_error = any(k in exc_str for k in (
        "InvalidCredentials", "API key could not be found",
        "Provided API key", "401", "403", "authentication",
    ))

    st.error("### ⚠️ IBM watsonx.ai Connection Failed")

    if is_cred_error:
        st.markdown(
            "**Your API key was rejected by IBM Cloud.**\n\n"
            "This means the key in your `.env` file is either:\n"
            "- Still the placeholder `your_ibm_cloud_api_key_here`\n"
            "- Copied incorrectly (extra spaces, missing characters)\n"
            "- Expired or deleted\n\n"
            f"📄 `.env` file location: `{_APP_DIR / '.env'}`"
        )
        st.info(
            "**Steps to fix:**\n\n"
            "**1. Generate a new IBM Cloud API key**\n"
            "   → [cloud.ibm.com](https://cloud.ibm.com) → top-right menu → "
            "**Manage** → **Access (IAM)** → **API keys** → **Create**\n\n"
            "**2. Get your watsonx.ai Project ID**\n"
            "   → [dataplatform.cloud.ibm.com](https://dataplatform.cloud.ibm.com) "
            "→ open your project → **Manage** tab → **General** → copy the UUID\n\n"
            "**3. Edit your `.env` file** — paste your real values:\n"
            "```\n"
            "WATSONX_API_KEY=<paste your real API key here>\n"
            "WATSONX_PROJECT_ID=<paste your real project UUID here>\n"
            "WATSONX_URL=https://us-south.ml.cloud.ibm.com\n"
            "```\n\n"
            "**4. Restart Streamlit** — press `Ctrl+C` in the terminal, then run `streamlit run app.py` again."
        )
    else:
        st.markdown(f"**Error details:** `{exc_str[:500]}`")
        st.info(
            "Check that:\n"
            "- Your `.env` file exists and has real (non-placeholder) values\n"
            "- `WATSONX_URL` matches your IBM Cloud region (default: `https://us-south.ml.cloud.ibm.com`)\n"
            "- You have an active Watson Machine Learning service associated with your project\n"
            "- Your internet connection is working"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Cached resource singletons
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading IBM Granite + RAG index …")
def load_agent() -> InterviewTrainerAgent:
    return InterviewTrainerAgent()


@st.cache_resource(show_spinner=False)
def load_doc_loader() -> DocumentLoader:
    return DocumentLoader()


@st.cache_resource(show_spinner=False)
def load_session_manager() -> SessionManager:
    return SessionManager()


# ─────────────────────────────────────────────────────────────────────────────
# Helper — split Granite's numbered question output into a list of dicts
# ─────────────────────────────────────────────────────────────────────────────
def parse_questions(raw_text: str) -> List[Dict]:
    """
    Parse the LLM's free-text output into individual question objects.

    Recognises patterns:
        "1. ...", "1) ...", "## Question 1", "**Question 1**", "Q1:"
    Returns list of {"number": int, "title": str, "body": str}.
    Falls back to a single entry containing the full text if no pattern matches.
    """
    splits = re.split(
        r"(?:^|\n)\s*(?:#{1,3}\s*)?(?:\*{1,2})?"
        r"(?:Question\s+|Q\.?\s*)?(\d{1,2})[.):\s](?:\*{1,2})?",
        raw_text,
        flags=re.IGNORECASE,
    )

    questions: List[Dict] = []
    if len(splits) >= 3:
        for i in range(1, len(splits) - 1, 2):
            num  = int(splits[i]) if splits[i].isdigit() else len(questions) + 1
            body = splits[i + 1].strip() if (i + 1) < len(splits) else ""
            if not body:
                continue
            lines      = body.splitlines()
            first_line = lines[0].strip().strip("*#").strip()
            rest       = "\n".join(lines[1:]).strip()
            questions.append({
                "number": num,
                "title":  first_line or f"Question {num}",
                "body":   rest,
            })

    if not questions:
        questions = [{"number": 1, "title": "Generated Questions", "body": raw_text.strip()}]

    return questions


# ─────────────────────────────────────────────────────────────────────────────
# Helper — display an EvaluationResult using native Streamlit widgets
# ─────────────────────────────────────────────────────────────────────────────
def render_rubric_card(result: EvaluationResult) -> None:
    """
    Show an EvaluationResult with st.metric, st.progress, st.success/warning/info.
    No HTML is used.
    """
    score = result.overall_score
    grade_emoji = (
        "🏆" if score >= 9 else
        "✅" if score >= 7 else
        "⚠️" if score >= 5 else
        "❌"
    )

    col_grade, col_score = st.columns([4, 1])
    with col_grade:
        st.subheader(f"{grade_emoji}  {result.grade}")
    with col_score:
        st.metric("Overall", f"{score} / 10")

    st.divider()

    # Dimension scores — one metric + progress bar per column
    st.markdown("**📊 Dimension Scores**")
    if result.dimensions:
        dim_cols = st.columns(len(result.dimensions))
        for col, (dim_key, dim_score) in zip(dim_cols, result.dimensions.items()):
            label = dim_key.replace("_", " ").title()
            with col:
                st.metric(label, f"{dim_score}/10")
                # st.progress takes 0.0–1.0
                st.progress(min(float(dim_score) / 10.0, 1.0))

    st.divider()

    # Strengths / improvements
    col_s, col_i = st.columns(2)
    with col_s:
        st.markdown("**✅ Strengths**")
        for s in (result.strengths or ["—"]):
            st.markdown(f"- {s}")
    with col_i:
        st.markdown("**⚠️ Areas for Improvement**")
        for imp in (result.improvements or ["Keep it up!"]):
            st.markdown(f"- {imp}")

    if result.coach_tip:
        st.info(f"💡 **Coach Tip:** {result.coach_tip}")


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Candidate profile + PDF/DOCX/TXT resume upload
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar(doc_loader: DocumentLoader) -> CandidateProfile:
    """
    Render sidebar with:
      1. Candidate profile widgets
      2. st.file_uploader for resume → RAG ingestion via DocumentLoader
    Returns a CandidateProfile dataclass.
    """
    st.sidebar.title("🎯 Interview Trainer")
    st.sidebar.divider()
    st.sidebar.subheader("👤 Candidate Profile")

    name = st.sidebar.text_input("Full Name", value="Alex Johnson", key="sb_name")

    role = st.sidebar.selectbox(
        "Target Job Role",
        [
            "Software Engineer", "Data Scientist", "ML Engineer",
            "Product Manager", "DevOps Engineer", "Frontend Developer",
            "Backend Developer", "Full Stack Developer", "Cloud Architect",
        ],
        key="sb_role",
    )

    experience = st.sidebar.selectbox(
        "Experience Level",
        [
            "Fresher (0–1 yr)", "Junior (1–3 yrs)", "Mid-level (3–5 yrs)",
            "Senior (5–8 yrs)", "Lead / Principal (8+ yrs)",
        ],
        key="sb_exp",
    )

    company        = st.sidebar.text_input("Target Company (optional)", key="sb_company")
    interview_date = st.sidebar.text_input(
        "Interview Date (optional)", placeholder="e.g. 2025-08-15", key="sb_date"
    )

    st.sidebar.markdown("**Focus Areas**")
    areas: List[str] = []
    if st.sidebar.checkbox("Technical / DSA",  value=True, key="fa_tech"): areas.append("technical")
    if st.sidebar.checkbox("System Design",     value=True, key="fa_sys"):  areas.append("system design")
    if st.sidebar.checkbox("Behavioral (STAR)", value=True, key="fa_beh"):  areas.append("behavioral")
    if st.sidebar.checkbox("HR / Situational",  value=True, key="fa_hr"):   areas.append("HR")

    # ── Resume upload ──────────────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.subheader("📎 Upload Resume / JD")
    st.sidebar.caption(
        "PDF, DOCX, or TXT — file is chunked and embedded into ChromaDB "
        "so IBM Granite personalises questions to your profile."
    )

    uploaded = st.sidebar.file_uploader(
        "Choose a file (PDF / DOCX / TXT)",
        type=["pdf", "docx", "txt"],
        key="sb_upload",
    )

    if uploaded is not None:
        # Use a stable key so we only ingest once per unique file+name combo
        ingest_key = f"ingested_{uploaded.name}_{name}"
        if ingest_key not in st.session_state:
            with st.sidebar:
                with st.spinner(f"Ingesting {uploaded.name} into RAG …"):
                    n = doc_loader.ingest_bytes(
                        uploaded.read(),
                        uploaded.name,
                        source_label=f"{name}_{uploaded.name}",
                    )
            st.session_state[ingest_key] = n
            # Cache top-matching chunks for this role
            chunks = doc_loader.search(role, n_results=12)
            st.session_state["resume_context"]  = "\n\n".join(c["text"] for c in chunks)
            st.session_state["resume_filename"] = uploaded.name

        n = st.session_state[ingest_key]
        st.sidebar.success(f"✅ {uploaded.name}  —  {n} chunks indexed in RAG")
        st.sidebar.caption("Resume context is active. Question generation will use it.")

    elif "resume_context" in st.session_state:
        fname = st.session_state.get("resume_filename", "uploaded file")
        st.sidebar.success(f"📂 Resume in RAG: {fname}")

    # List ingested sources + clear button
    sources = doc_loader.list_sources()
    if sources:
        st.sidebar.caption(
            "RAG sources: " + ", ".join(s.split("_", 1)[-1] for s in sources)
        )
        if st.sidebar.button("🗑 Clear all RAG sources", key="sb_clear_rag"):
            for s in sources:
                doc_loader.clear_source(s)
            st.session_state.pop("resume_context",  None)
            st.session_state.pop("resume_filename", None)
            st.rerun()

    st.sidebar.divider()
    st.sidebar.caption("Powered by IBM Granite · ChromaDB RAG · IBM Cloud Lite")

    return CandidateProfile(
        name             = name,
        role             = role,
        experience_level = experience,
        focus_areas      = areas,
        company          = company or "Not specified",
        interview_date   = interview_date or "Not specified",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Question card — one native st.expander per parsed question
# ─────────────────────────────────────────────────────────────────────────────
import time

def render_question_card(q, profile, agent, sm, idx) -> None:
    """
    Render one question as a native st.expander containing:
      - Guidance text (model answer outline, what is assessed)
      - st.text_area  — candidate types their answer
      - st.button("Submit Answer") — calls Evaluator via IBM Granite
      - Rubric result shown inline with native widgets after evaluation
      - st.button("Reset") — clears a prior evaluation
    State is stored in st.session_state keyed by a content hash so keys
    are stable across reruns and unique across questions.
    """
    idx        = q["number"]
    title      = q["title"]
    body       = q["body"]

    # Stable unique ID derived from question content — never changes on rerun,
    # never collides even if two questions share the same index.
    q_id       = hashlib.md5(title.encode()).hexdigest()

    eval_key   = f"q_eval_{q_id}"
    answer_key = f"q_ans_{q_id}"

    already_answered = eval_key in st.session_state

    # Build expander label — includes score badge when answered
    if already_answered:
        score = st.session_state[eval_key].overall_score
        expander_label = f"Q{idx}. {title}   ✅  Score: {score}/10"
    else:
        expander_label = f"Q{idx}. {title}"

    with st.expander(expander_label, expanded=already_answered):

        # Guidance / detail from Granite (what the interviewer assesses, model key points)
        if body:
            st.caption("Interviewer guidance — from IBM Granite")
            st.markdown(body)
            st.divider()

        # Answer text area
        user_ans = st.text_area(
            "Your Answer",
            value       = st.session_state.get(answer_key, ""),
            height      = 180,
            key         = f"ta_{q_id}",
            placeholder = "Think carefully, then type your full answer here …",
        )

        # Submit / Reset buttons
        btn_col, reset_col = st.columns([4, 1])
        with btn_col:
            submit_clicked = st.button(
                label="Submit Answer",
                key = f"submit_{q_id}",
                use_container_width = True,
            )
        with reset_col:
            if already_answered:
                if st.button("Reset", key=f"reset_{q_id}", use_container_width=True):
                    st.session_state.pop(eval_key,   None)
                    st.session_state.pop(answer_key, None)
                    st.rerun()

        # ── Handle submission ──────────────────────────────────────────────
        if submit_clicked:
            if not user_ans.strip():
                st.warning("Please write your answer before submitting.")
            else:
                # Persist the answer so it survives the rerun
                st.session_state[answer_key] = user_ans

                # Retrieve closest reference answer from the RAG knowledge base
                docs    = agent.retriever.retrieve(title, role=profile.role)
                ref_ans = docs[0]["model_answer"] if docs else ""
                cat     = docs[0]["category"]     if docs else "technical"

                evaluator = Evaluator(agent.llm)
                try:
                    with st.spinner("IBM Granite is scoring your answer …"):
                        eval_result = evaluator.score(
                            role             = profile.role,
                            experience_level = profile.experience_level,
                            question         = title,
                            user_answer      = user_ans,
                            model_answer     = ref_ans,
                            category         = cat,
                        )
                except Exception as exc:
                    _show_ibm_error(exc)
                    return
                st.session_state[eval_key] = eval_result

                # Save to active mock session if one is running
                active_sid = st.session_state.get("active_session_id")
                if active_sid:
                    sm_ref: Optional[SessionManager] = st.session_state.get("_sm_ref")
                    if sm_ref is not None:
                        sm_ref.add_round(
                            session_id  = active_sid,
                            question    = title,
                            user_answer = user_ans,
                            feedback    = eval_result.raw_feedback,
                            score       = int(eval_result.overall_score),
                            category    = cat,
                        )
                        sm_ref.save()

                st.rerun()

        # ── Show evaluation results (persisted from prior rerun) ───────────
        if already_answered:
            st.divider()
            st.markdown("#### 📊 Your Evaluation")
            render_rubric_card(st.session_state[eval_key])

            # Reference answer from knowledge base
            docs = agent.retriever.retrieve(title, role=profile.role)
            if docs and docs[0].get("model_answer"):
                d = docs[0]
                with st.expander("📖 Reference Answer from Knowledge Base"):
                    st.markdown(
                        f"**Category:** {d.get('category','').title()}  |  "
                        f"**Difficulty:** {d.get('difficulty','').title()}"
                    )
                    st.markdown(d["model_answer"])
                    if d.get("improvement_tip"):
                        st.info(f"💡 Expert Tip: {d['improvement_tip']}")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Generate Questions
# ─────────────────────────────────────────────────────────────────────────────
def tab_generate_questions(
    agent: InterviewTrainerAgent,
    profile: CandidateProfile,
    doc_loader: DocumentLoader,
    sm: SessionManager,
) -> None:
    st.header("🎯 Generate Tailored Interview Questions")
    st.markdown(
        f"Role: **{profile.role}**  |  Level: **{profile.experience_level}**  |  "
        f"Candidate: **{profile.name}**"
    )

    # ── Controls ──────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        category = st.selectbox(
            "Question Category",
            ["All", "technical", "behavioral", "hr"],
            key="gcat",
        )
    with c2:
        num_q = st.slider("Number of Questions", min_value=3, max_value=10, value=6, key="gnq")
    with c3:
        resume_available = "resume_context" in st.session_state
        use_resume = st.checkbox(
            "Use my resume",
            value    = resume_available,
            key      = "guse_resume",
            help     = "Enriches questions with your uploaded resume via RAG.",
            disabled = not resume_available,
        )

    # Status banners
    if resume_available and use_resume:
        fname = st.session_state.get("resume_filename", "uploaded document")
        st.success(f"✅ Resume context active — using **{fname}** (RAG-retrieved).")
    elif not resume_available:
        st.info("Upload a PDF/DOCX/TXT resume in the sidebar to personalise questions.")

    # RAG knowledge base stats
    idx_stats = agent.retriever.get_index_stats()
    st.caption(
        f"Knowledge base: {idx_stats['total_documents']} Q&As in ChromaDB  "
        f"·  Embedding model: all-MiniLM-L6-v2  ·  LLM: IBM Granite"
    )

    st.divider()

    # ── Generate button ────────────────────────────────────────────────────
    if st.button("🚀 Generate Questions", key="btn_gen"):
        # Clear previous questions and all per-question state immediately
        st.session_state.pop("gen_questions_raw", None)
        st.session_state.pop("gen_questions_parsed", None)
        for key in list(st.session_state.keys()):
            if str(key).startswith("q_eval_") or str(key).startswith("q_ans_"):
                del st.session_state[key]

        # Build resume context string
        resume_ctx = ""
        if use_resume and resume_available:
            live_chunks = doc_loader.search(
                f"{profile.role} {profile.experience_level} {' '.join(profile.focus_areas)}",
                n_results=8,
            )
            live_text  = "\n\n".join(c["text"] for c in live_chunks)
            resume_ctx = (live_text or st.session_state.get("resume_context", ""))[:3000]

        profile_with_ctx = CandidateProfile(
            name             = profile.name,
            role             = profile.role,
            experience_level = profile.experience_level,
            focus_areas      = profile.focus_areas,
            company          = profile.company,
            interview_date   = profile.interview_date,
            resume_text      = resume_ctx,
        )

        try:
            with st.spinner("IBM Granite + RAG is generating your question set …"):
                raw = agent.generate_questions(
                    profile_with_ctx,
                    num_questions = num_q,
                    category      = None if category == "All" else category,
                )
        except Exception as exc:
            _show_ibm_error(exc)
            return

        parsed = parse_questions(raw)
        # Deduplicate by question title hash to prevent any key collisions
        seen_hashes: set = set()
        unique_parsed = []
        for q in parsed:
            q_hash = hashlib.md5(q["title"].encode()).hexdigest()
            if q_hash not in seen_hashes:
                seen_hashes.add(q_hash)
                unique_parsed.append(q)

        st.session_state["gen_questions_raw"]    = raw
        st.session_state["gen_questions_parsed"] = unique_parsed
        # Rerun so the render block sees the new state with a clean widget tree
        st.rerun()

    # ── Render question expanders ──────────────────────────────────────────
    if "gen_questions_parsed" in st.session_state:
        questions = st.session_state["gen_questions_parsed"]
        answered  = sum(
            1 for q in questions
            if f"q_eval_{hashlib.md5(q['title'].encode()).hexdigest()}" in st.session_state
        )

        hdr_col, dl_col = st.columns([5, 1])
        with hdr_col:
            st.subheader(
                f"📋 {len(questions)} Questions Generated"
                + (f"  —  {answered} answered" if answered else "")
            )
        with dl_col:
            st.download_button(
                "📥 Download",
                data      = st.session_state["gen_questions_raw"],
                file_name = f"questions_{profile.role.lower().replace(' ', '_')}.txt",
                mime      = "text/plain",
                key       = "dl_gen",
            )

        st.caption(
            "Expand a question below, type your answer, "
            "then click **Submit Answer** for instant IBM Granite rubric feedback."
        )

        for idx, q in enumerate(questions):
            render_question_card(q, profile, agent, sm, idx)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Evaluate Answer (free-form entry)
# ─────────────────────────────────────────────────────────────────────────────
def tab_evaluate_answer(
    agent: InterviewTrainerAgent,
    profile: CandidateProfile,
) -> None:
    st.header("📝 Evaluate Your Answer")
    st.markdown(
        "Paste any interview question and your answer. "
        "IBM Granite scores it across **5 dimensions** and gives "
        "strengths, improvement areas, and a concrete coaching tip."
    )

    question = st.text_area(
        "Interview Question",
        height      = 80,
        placeholder = "e.g. Explain the CAP theorem in distributed systems.",
        key         = "eval_q",
    )
    user_answer = st.text_area(
        "Your Answer",
        height      = 200,
        placeholder = "Type your full answer. Take your time — quality matters.",
        key         = "eval_a",
    )

    c1, c2 = st.columns([3, 1])
    with c1:
        use_rubric = st.checkbox(
            "Use structured 5-dimension rubric scoring (recommended)",
            value = True,
            key   = "use_rubric",
        )
    with c2:
        cat_override = st.selectbox(
            "Category override",
            ["auto-detect", "technical", "behavioral", "hr"],
            key = "eval_cat",
        )

    if st.button("🔍 Evaluate My Answer", key="btn_eval"):
        if not question.strip():
            st.warning("Please enter the interview question.")
            return
        if not user_answer.strip():
            st.warning("Please enter your answer.")
            return

        # Retrieve reference from RAG knowledge base
        docs    = agent.retriever.retrieve(question, role=profile.role)
        ref_ans = docs[0]["model_answer"] if docs else ""
        cat     = docs[0]["category"]     if docs else "technical"
        if cat_override != "auto-detect":
            cat = cat_override

        if use_rubric:
            evaluator = Evaluator(agent.llm)
            try:
                with st.spinner("IBM Granite is scoring across 5 dimensions …"):
                    result = evaluator.score(
                        role             = profile.role,
                        experience_level = profile.experience_level,
                        question         = question,
                        user_answer      = user_answer,
                        model_answer     = ref_ans,
                        category         = cat,
                    )
            except Exception as exc:
                _show_ibm_error(exc)
                return
            st.divider()
            render_rubric_card(result)

            if ref_ans:
                with st.expander("📖 Reference Answer from Knowledge Base"):
                    st.markdown(
                        f"**Category:** {cat.title()}  |  "
                        f"**Difficulty:** {docs[0].get('difficulty','').title()}"
                    )
                    st.markdown(ref_ans)
                    if docs[0].get("improvement_tip"):
                        st.info(f"💡 Expert Tip: {docs[0]['improvement_tip']}")

            dl_text = result.summary
        else:
            try:
                with st.spinner("IBM Granite is evaluating …"):
                    feedback = agent.evaluate_answer(profile, question, user_answer, ref_ans)
            except Exception as exc:
                _show_ibm_error(exc)
                return
            st.divider()
            st.markdown(feedback)
            dl_text = feedback

        st.download_button(
            "📥 Download Feedback",
            data      = dl_text,
            file_name = "answer_feedback.txt",
            mime      = "text/plain",
            key       = "dl_eval",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Mock Interview
# ─────────────────────────────────────────────────────────────────────────────
def tab_mock_interview(
    agent: InterviewTrainerAgent,
    profile: CandidateProfile,
    sm: SessionManager,
) -> None:
    st.header("🎤 Mock Interview Session")
    st.markdown(
        "Get a fresh IBM Granite-generated question, type your answer, "
        "and receive instant rubric feedback saved to your session history."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        category   = st.selectbox("Category",   ["technical", "behavioral", "hr"],         key="mcat")
    with c2:
        difficulty = st.selectbox("Difficulty", ["beginner", "intermediate", "advanced"],   key="mdiff")
    with c3:
        st.write("")   # vertical alignment spacer
        if st.button("🆕 Start New Session", key="btn_new_sess"):
            sid = sm.new_session(SessionManager.profile_to_dict(profile))
            st.session_state["active_session_id"] = sid
            st.session_state.pop("mock_result", None)
            st.session_state.pop("mock_eval",   None)
            st.success(f"Session **{sid}** started — good luck! 🍀")

    # Active session status
    active_sid = st.session_state.get("active_session_id")
    if active_sid:
        report = sm.get_session_report(active_sid)
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Session ID",  active_sid)
        mc2.metric("Rounds Done", report["total_rounds"])
        mc3.metric(
            "Avg Score",
            f"{report['average_score']} / 10" if report["total_rounds"] else "—",
        )
    else:
        st.info("Click **Start New Session** above to begin tracking your scores.")

    st.divider()

    if st.button("🎲 Give Me a Question", key="btn_mock"):
        with st.spinner("IBM Granite is generating a question via RAG …"):
            mock_data = agent.run_mock_interview(profile, category, difficulty)
        st.session_state["mock_result"] = mock_data
        st.session_state.pop("mock_eval", None)   # clear previous evaluation

    if "mock_result" in st.session_state:
        r = st.session_state["mock_result"]

        st.subheader("❓ Your Question")
        st.info(f"**{r['category'].upper()}  ·  {r['difficulty'].title()}**")
        st.markdown(r["generated_question"])

        st.divider()

        user_ans = st.text_area(
            "Your Answer",
            height      = 180,
            key         = "mock_ans",
            placeholder = "Take a moment to think, then write your answer in full …",
        )

        if st.button("✅ Submit & Score", key="btn_mock_submit"):
            if not user_ans.strip():
                st.warning("Please write your answer before submitting.")
                return

            evaluator = Evaluator(agent.llm)
            with st.spinner("Scoring with IBM Granite rubric …"):
                result_eval = evaluator.score(
                    role             = profile.role,
                    experience_level = profile.experience_level,
                    question         = r["generated_question"],
                    user_answer      = user_ans,
                    model_answer     = r["reference_answer"],
                    category         = r["category"],
                )
            st.session_state["mock_eval"] = (user_ans, result_eval)

            if active_sid:
                sm.add_round(
                    session_id  = active_sid,
                    question    = r["generated_question"],
                    user_answer = user_ans,
                    feedback    = result_eval.raw_feedback,
                    score       = int(result_eval.overall_score),
                    category    = r["category"],
                    difficulty  = r["difficulty"],
                )
                sm.save()

            st.rerun()

        # Show evaluation from state (survives rerun)
        if "mock_eval" in st.session_state:
            _ans, result_eval = st.session_state["mock_eval"]
            st.divider()
            render_rubric_card(result_eval)

            if r.get("improvement_tip"):
                st.info(f"💡 Expert Tip from Knowledge Base: {r['improvement_tip']}")

            if r.get("reference_answer"):
                with st.expander("📖 View Reference Answer"):
                    st.markdown(r["reference_answer"])

            if active_sid:
                st.caption(f"✅ Round saved to session **{active_sid}**.")
            else:
                st.caption("Start a session above to save your scores.")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — Preparation Strategy
# ─────────────────────────────────────────────────────────────────────────────
def tab_prep_strategy(
    agent: InterviewTrainerAgent,
    profile: CandidateProfile,
) -> None:
    st.header("📅 Personalised Preparation Plan")
    st.markdown(
        f"**Role:** {profile.role}  |  **Company:** {profile.company}  |  "
        f"**Interview date:** {profile.interview_date}  |  "
        f"**Level:** {profile.experience_level}"
    )

    if st.button("📋 Build My Prep Plan", key="btn_prep"):
        with st.spinner("IBM Granite is building your study plan …"):
            plan = agent.build_prep_strategy(profile)
        st.session_state["prep_plan"] = plan

    if "prep_plan" in st.session_state:
        st.divider()
        st.markdown(st.session_state["prep_plan"])
        st.download_button(
            "📥 Download Plan",
            data      = st.session_state["prep_plan"],
            file_name = f"prep_plan_{profile.name.lower().replace(' ', '_')}.txt",
            mime      = "text/plain",
            key       = "dl_prep",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 5 — Resume Analyser
# ─────────────────────────────────────────────────────────────────────────────
def tab_resume_analyzer(
    agent: InterviewTrainerAgent,
    profile: CandidateProfile,
    doc_loader: DocumentLoader,
) -> None:
    st.header("📄 Resume & Job Description Analyser")
    st.markdown(
        "IBM Granite analyses your resume or a job description to extract skills, "
        "estimate your level, predict tough questions, and recommend roles."
    )

    upload_tab, text_tab = st.tabs(["📎 Upload File", "✏️ Paste Text"])

    resume_text = ""

    with upload_tab:
        st.markdown("Upload a **PDF**, **DOCX**, or **TXT** file.")

        col_up, col_sources = st.columns([3, 2])
        with col_up:
            tab_file = st.file_uploader(
                "Choose file",
                type=["pdf", "docx", "txt"],
                key="tab_upload",
            )
        with col_sources:
            sources = doc_loader.list_sources()
            if sources:
                st.markdown("**Already in RAG:**")
                for s in sources:
                    st.caption(f"• {s.split('_', 1)[-1]}")

        if tab_file is not None:
            ingest_key = f"tab_ingested_{tab_file.name}"
            if ingest_key not in st.session_state:
                with st.spinner(f"Ingesting {tab_file.name} into RAG …"):
                    n = doc_loader.ingest_bytes(
                        tab_file.read(),
                        tab_file.name,
                        source_label=f"{profile.name}_{tab_file.name}",
                    )
                st.session_state[ingest_key] = n
                chunks = doc_loader.search(profile.role, n_results=10)
                st.session_state["resume_context"]  = "\n\n".join(c["text"] for c in chunks)
                st.session_state["resume_filename"] = tab_file.name

            st.success(
                f"✅ **{tab_file.name}** — "
                f"{st.session_state[ingest_key]} chunks in vector store."
            )
            resume_text = st.session_state.get("resume_context", "")

    with text_tab:
        pasted = st.text_area(
            "Paste resume or job description text",
            height      = 280,
            placeholder = "Paste plain text from your CV or a JD here …",
            key         = "rtext",
        )
        if pasted.strip():
            resume_text = pasted
            if st.button("📥 Ingest into RAG", key="btn_ingest_text"):
                n = doc_loader.ingest_text(pasted, source_label="pasted_text")
                st.session_state["resume_context"]  = pasted
                st.session_state["resume_filename"] = "pasted_text"
                st.success(f"✅ Pasted text ingested ({n} chunks).")

    st.divider()

    if st.button("🔎 Analyse with IBM Granite", key="btn_analyse"):
        if not resume_text.strip():
            st.warning("Please upload a file or paste text first.")
            return
        with st.spinner("IBM Granite is analysing your profile …"):
            analysis = agent.analyze_resume(resume_text)
        st.session_state["resume_analysis"]     = analysis
        st.session_state["resume_analysis_ctx"] = resume_text

    if "resume_analysis" in st.session_state:
        st.markdown(st.session_state["resume_analysis"])

        st.divider()
        st.subheader("🎯 Auto-Generate Questions from this Resume")
        if st.button("Generate tailored questions from this resume", key="btn_gen_from_resume"):
            with st.spinner("Generating questions from resume context …"):
                ctx = st.session_state.get("resume_analysis_ctx", "")
                profile_with_ctx = CandidateProfile(
                    name             = profile.name,
                    role             = profile.role,
                    experience_level = profile.experience_level,
                    focus_areas      = profile.focus_areas,
                    company          = profile.company,
                    interview_date   = profile.interview_date,
                    resume_text      = ctx,
                )
                qs = agent.generate_questions(profile_with_ctx, num_questions=5)
            st.session_state["resume_gen_questions"] = qs

        if "resume_gen_questions" in st.session_state:
            st.markdown(st.session_state["resume_gen_questions"])
            st.download_button(
                "📥 Download Questions",
                data      = st.session_state["resume_gen_questions"],
                file_name = "resume_based_questions.txt",
                mime      = "text/plain",
                key       = "dl_res_q",
            )

        st.download_button(
            "📥 Download Analysis",
            data      = st.session_state["resume_analysis"],
            file_name = "resume_analysis.txt",
            mime      = "text/plain",
            key       = "dl_resume",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 6 — Session History & Analytics
# ─────────────────────────────────────────────────────────────────────────────
def tab_session_history(sm: SessionManager) -> None:
    st.header("📊 Session History & Analytics")

    sessions = sm.get_recent_sessions(10)
    if not sessions:
        st.info(
            "No sessions recorded yet. "
            "Go to **🎤 Mock Interview**, click **Start New Session**, "
            "then answer questions to start tracking your progress."
        )
        return

    session_ids = [s["id"] for s in sessions]
    labels = [
        f"{s['id']}  ·  {s['profile'].get('role', 'Unknown')}  ·  "
        f"{s['created_at'][:10]}  ·  {len(s['rounds'])} rounds"
        for s in sessions
    ]
    chosen_label = st.selectbox("Select session to review", labels, key="sess_picker")
    chosen_id    = session_ids[labels.index(chosen_label)]

    report = sm.get_session_report(chosen_id)

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Total Rounds",  report["total_rounds"])
    mc2.metric("Scored Rounds", report["scored_rounds"])
    mc3.metric("Avg Score",     f"{report['average_score']} / 10")
    mc4.metric(
        "Weakest Area",
        report["weakest_category"].replace("_", " ").title()
        if report["weakest_category"] != "N/A" else "N/A",
    )

    chart_c1, chart_c2 = st.columns(2)
    with chart_c1:
        if report["score_trend"]:
            st.markdown("**📈 Score Trend**")
            st.line_chart(
                pd.DataFrame(
                    {"Score": report["score_trend"]},
                    index=range(1, len(report["score_trend"]) + 1),
                ),
                height=220,
            )
    with chart_c2:
        if report["category_averages"]:
            st.markdown("**📋 Category Averages**")
            st.bar_chart(
                pd.DataFrame(
                    [(k.title(), v) for k, v in report["category_averages"].items()],
                    columns=["Category", "Avg Score"],
                ).set_index("Category"),
                height=220,
            )

    st.subheader("🗂️ Round Details")
    if not report["rounds"]:
        st.info("No rounds in this session yet.")
    else:
        for r in report["rounds"]:
            score_val = r["score"] or 0
            with st.expander(
                f"Round {r['round_no']}  ·  {r['category'].title()}  ·  "
                f"{r['difficulty'].title()}  ·  Score: {score_val}/10"
            ):
                st.markdown(f"**Question:**\n\n{r['question']}")
                st.divider()
                ans = r["user_answer"]
                st.markdown(
                    f"**Your Answer:**\n\n"
                    + (ans[:600] + " …" if len(ans) > 600 else ans)
                )
                st.divider()
                st.markdown("**Feedback:**")
                st.markdown(r["feedback"][:1000])

    st.divider()
    col_j, col_m = st.columns(2)
    with col_j:
        st.download_button(
            "📥 Export Session JSON",
            data      = json.dumps(report, indent=2),
            file_name = f"session_{chosen_id}.json",
            mime      = "application/json",
            key       = "dl_sess_json",
        )
    with col_m:
        lines = [
            f"# Session Report — {chosen_id}",
            f"Role: {report['profile'].get('role','')}",
            f"Date: {report['created_at'][:10]}",
            f"Average Score: {report['average_score']}/10",
            f"Rounds: {report['total_rounds']}",
            "",
        ]
        for r in report["rounds"]:
            lines += [
                f"## Round {r['round_no']} ({r['category'].title()})",
                f"**Q:** {r['question']}",
                f"**Score:** {r['score']}/10",
                "",
            ]
        st.download_button(
            "📥 Export Markdown Summary",
            data      = "\n".join(lines),
            file_name = f"session_{chosen_id}_summary.md",
            mime      = "text/markdown",
            key       = "dl_sess_md",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    # App header
    st.title("🎯 Interview Trainer Agent")
    st.markdown(
        "Powered by **IBM Granite** (watsonx.ai)  ·  "
        "RAG Pipeline (ChromaDB)  ·  IBM Cloud Lite"
    )

    # Top-level stats row
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Knowledge Base Q&As", "34+")
    mc2.metric("Interview Modes",      "6")
    mc3.metric("Rubric Dimensions",    "5")
    mc4.metric("LLM Engine",           "IBM Granite")

    st.divider()

    # Singletons
    doc_loader = load_doc_loader()
    sm         = load_session_manager()

    # Sidebar — returns candidate profile
    profile = render_sidebar(doc_loader)

    # Store session manager so question cards can save rounds
    st.session_state["_sm_ref"] = sm

    # Agent — load with full credential error handling
    try:
        agent = load_agent()
    except EnvironmentError as exc:
        # Missing or placeholder credentials — actionable setup guide
        st.error("### ⚠️ Credentials Not Configured")
        st.markdown(str(exc))
        st.info(
            "**How to fix:**\n\n"
            f"1. Open `{_APP_DIR / '.env'}` in Notepad\n"
            "2. Replace the placeholder values with your real credentials:\n\n"
            "```\n"
            "WATSONX_API_KEY=<your IBM Cloud API key>\n"
            "WATSONX_PROJECT_ID=<your watsonx.ai project UUID>\n"
            "WATSONX_URL=https://us-south.ml.cloud.ibm.com\n"
            "```\n\n"
            "Get your API key → [cloud.ibm.com](https://cloud.ibm.com) → Manage → Access (IAM) → API keys\n\n"
            "Get your Project ID → [dataplatform.cloud.ibm.com](https://dataplatform.cloud.ibm.com) → your project → Manage → General"
        )
        st.stop()
    except Exception as exc:
        # InvalidCredentialsError or any other IBM SDK error at import time
        _show_ibm_error(exc)
        st.stop()

    # Tabs
    tabs = st.tabs([
        "🎯 Generate Questions",
        "📝 Evaluate Answer",
        "🎤 Mock Interview",
        "📅 Prep Strategy",
        "📄 Resume Analyser",
        "📊 Session History",
    ])

    with tabs[0]: tab_generate_questions(agent, profile, doc_loader, sm)
    with tabs[1]: tab_evaluate_answer(agent, profile)
    with tabs[2]: tab_mock_interview(agent, profile, sm)
    with tabs[3]: tab_prep_strategy(agent, profile)
    with tabs[4]: tab_resume_analyzer(agent, profile, doc_loader)
    with tabs[5]: tab_session_history(sm)


if __name__ == "__main__":
    main()
