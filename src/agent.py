"""
Interview Trainer Agent — Core Orchestration Layer

Ties together:
  1. InterviewRetriever (RAG vector store)
  2. GraniteLLM (IBM watsonx.ai inference)
  3. Prompt templates (role-specific prompting)

Public API:
  agent = InterviewTrainerAgent()
  agent.generate_questions(profile)
  agent.evaluate_answer(profile, question, user_answer, model_answer)
  agent.build_prep_strategy(profile)
  agent.analyze_resume(text)
  agent.run_mock_interview(profile, category, difficulty)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .granite_llm import GraniteLLM
from .rag_pipeline import InterviewRetriever
from .prompt_templates import (
    SYSTEM_PERSONA,
    QUESTION_GENERATION_TEMPLATE,
    ANSWER_EVALUATION_TEMPLATE,
    PREP_STRATEGY_TEMPLATE,
    RESUME_ANALYSIS_TEMPLATE,
    SINGLE_QUESTION_TEMPLATE,
    format_retrieval_context,
)


# ---------------------------------------------------------------------------
# Candidate profile dataclass
# ---------------------------------------------------------------------------
@dataclass
class CandidateProfile:
    name:             str
    role:             str
    experience_level: str                    # Fresher | Junior | Mid | Senior | Lead
    focus_areas:      List[str] = field(default_factory=list)
    company:          str = "Not specified"
    interview_date:   str = "Not specified"
    resume_text:      str = ""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class InterviewTrainerAgent:
    """
    End-to-end Interview Trainer orchestrating RAG retrieval + IBM Granite LLM.
    """

    def __init__(self):
        print("[Agent] Initialising IBM Granite LLM …")
        self.llm       = GraniteLLM()

        print("[Agent] Initialising RAG retriever …")
        self.retriever = InterviewRetriever()
        self.retriever.build_index()

        print("[Agent] Ready ✓")

    # ------------------------------------------------------------------
    # 1. Generate tailored question set
    # ------------------------------------------------------------------
    def generate_questions(
        self,
        profile: CandidateProfile,
        num_questions: int = 6,
        category: Optional[str] = None,
    ) -> str:
        """
        Retrieve relevant questions from the knowledge base, then ask
        Granite to generate a tailored question set for the candidate.
        """
        query  = f"{profile.role} {profile.experience_level} interview questions "
        query += " ".join(profile.focus_areas)

        docs     = self.retriever.retrieve(query, role=profile.role, category=category)
        context  = format_retrieval_context(docs)

        # Append uploaded resume / JD text so Granite personalises the questions
        if profile.resume_text:
            context += (
                "\n\n## Candidate Resume / Profile Context\n"
                + profile.resume_text[:2000].strip()
            )

        prompt = QUESTION_GENERATION_TEMPLATE.format(
            system           = SYSTEM_PERSONA,
            name             = profile.name,
            role             = profile.role,
            experience_level = profile.experience_level,
            focus_areas      = ", ".join(profile.focus_areas) if profile.focus_areas else "General",
            context          = context,
            num_questions    = num_questions,
        )

        return self.llm.generate(prompt)

    # ------------------------------------------------------------------
    # 2. Evaluate a candidate's answer
    # ------------------------------------------------------------------
    def evaluate_answer(
        self,
        profile: CandidateProfile,
        question: str,
        user_answer: str,
        model_answer: str = "",
    ) -> str:
        """
        Retrieve the expert model answer (if not provided), then evaluate
        the candidate's answer using IBM Granite.
        """
        if not model_answer:
            docs = self.retriever.retrieve(question, role=profile.role)
            model_answer = docs[0]["model_answer"] if docs else "No reference answer found."

        prompt = ANSWER_EVALUATION_TEMPLATE.format(
            system           = SYSTEM_PERSONA,
            role             = profile.role,
            experience_level = profile.experience_level,
            question         = question,
            user_answer      = user_answer,
            model_answer     = model_answer,
        )

        return self.llm.generate(prompt)

    # ------------------------------------------------------------------
    # 3. Build a personalised preparation strategy
    # ------------------------------------------------------------------
    def build_prep_strategy(self, profile: CandidateProfile) -> str:
        """Generate a day-by-day preparation plan for the candidate."""
        prompt = PREP_STRATEGY_TEMPLATE.format(
            system           = SYSTEM_PERSONA,
            name             = profile.name,
            role             = profile.role,
            experience_level = profile.experience_level,
            interview_date   = profile.interview_date,
            company          = profile.company,
        )
        return self.llm.generate(prompt)

    # ------------------------------------------------------------------
    # 4. Analyse resume / job description text
    # ------------------------------------------------------------------
    def analyze_resume(self, resume_text: str) -> str:
        """Extract skills, assess experience level, and predict interview topics."""
        prompt = RESUME_ANALYSIS_TEMPLATE.format(
            system      = SYSTEM_PERSONA,
            resume_text = resume_text,
        )
        return self.llm.generate(prompt)

    # ------------------------------------------------------------------
    # 5. Single mock interview question
    # ------------------------------------------------------------------
    def run_mock_interview(
        self,
        profile: CandidateProfile,
        category: str = "technical",
        difficulty: str = "intermediate",
    ) -> dict:
        """
        Retrieve one contextually relevant question and ask Granite to
        generate a fresh version of it. Returns both the question and
        the reference material for post-answer evaluation.
        """
        query = f"{profile.role} {category} {difficulty} interview question"
        docs  = self.retriever.retrieve(query, role=profile.role, category=category)

        retrieved_q = docs[0]["question"] if docs else "N/A"
        ref_answer  = docs[0]["model_answer"] if docs else ""
        tip         = docs[0]["improvement_tip"] if docs else ""

        prompt = SINGLE_QUESTION_TEMPLATE.format(
            system             = SYSTEM_PERSONA,
            role               = profile.role,
            category           = category,
            difficulty         = difficulty,
            retrieved_question = retrieved_q,
        )

        generated = self.llm.generate(prompt)

        return {
            "generated_question": generated,
            "reference_answer":   ref_answer,
            "improvement_tip":    tip,
            "category":           category,
            "difficulty":         difficulty,
        }

    # ------------------------------------------------------------------
    # 6. Health check
    # ------------------------------------------------------------------
    def health_check(self) -> dict:
        llm_ok  = self.llm.health_check()
        idx_ok  = self.retriever.get_index_stats()
        return {
            "llm_connected":   llm_ok,
            "index_stats":     idx_ok,
            "status":          "healthy" if llm_ok else "degraded",
        }
