"""
Session Manager — tracks mock interview sessions, scores each answer,
computes aggregate analytics, and persists history to a local JSON file.

Each session contains:
  - candidate profile snapshot
  - list of Q&A rounds (question, user_answer, feedback, score)
  - aggregate statistics (avg score, category breakdown)

Usage:
    sm = SessionManager()
    session_id = sm.new_session(profile)
    sm.add_round(session_id, question, user_answer, feedback, score=7)
    report = sm.get_session_report(session_id)
    sm.save()
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

SESSION_FILE = os.getenv("SESSION_FILE", "./data/sessions.json")


class SessionManager:
    """
    Lightweight JSON-backed session store.
    Keeps interview history in memory during the Streamlit session
    and persists to disk on save().
    """

    def __init__(self, session_file: str = SESSION_FILE):
        self.session_file = Path(session_file)
        self.sessions: Dict[str, dict] = {}
        self._load()

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------
    def new_session(self, profile_dict: dict) -> str:
        """Create a new blank session and return its ID."""
        session_id = str(uuid.uuid4())[:8]
        self.sessions[session_id] = {
            "id":         session_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "profile":    profile_dict,
            "rounds":     [],
            "status":     "active",
        }
        return session_id

    def add_round(
        self,
        session_id: str,
        question:    str,
        user_answer: str,
        feedback:    str,
        score:       Optional[int] = None,
        category:    str = "general",
        difficulty:  str = "intermediate",
    ) -> None:
        """Append one Q&A round to an existing session."""
        if session_id not in self.sessions:
            raise KeyError(f"Session '{session_id}' not found.")

        round_data = {
            "round_no":   len(self.sessions[session_id]["rounds"]) + 1,
            "timestamp":  datetime.now().isoformat(timespec="seconds"),
            "question":   question,
            "user_answer": user_answer,
            "feedback":   feedback,
            "score":      score,
            "category":   category,
            "difficulty": difficulty,
        }
        self.sessions[session_id]["rounds"].append(round_data)

    def close_session(self, session_id: str) -> None:
        """Mark session as completed."""
        if session_id in self.sessions:
            self.sessions[session_id]["status"]   = "completed"
            self.sessions[session_id]["closed_at"] = datetime.now().isoformat(timespec="seconds")

    # ------------------------------------------------------------------
    # Reporting & analytics
    # ------------------------------------------------------------------
    def get_session_report(self, session_id: str) -> dict:
        """
        Build a summary report for a completed/active session including:
        - Total rounds attempted
        - Average score
        - Scores by category
        - Weakest category (lowest avg score)
        - Progress trend (scores over rounds)
        """
        if session_id not in self.sessions:
            raise KeyError(f"Session '{session_id}' not found.")

        session = self.sessions[session_id]
        rounds  = session["rounds"]

        if not rounds:
            return {"session_id": session_id, "rounds": 0, "message": "No rounds recorded yet."}

        scores = [r["score"] for r in rounds if r["score"] is not None]
        avg    = round(sum(scores) / len(scores), 1) if scores else 0.0

        # Per-category breakdown
        cat_scores: Dict[str, List[int]] = {}
        for r in rounds:
            if r["score"] is not None:
                cat_scores.setdefault(r["category"], []).append(r["score"])

        cat_avg = {cat: round(sum(v) / len(v), 1) for cat, v in cat_scores.items()}
        weakest = min(cat_avg, key=cat_avg.get) if cat_avg else "N/A"

        return {
            "session_id":        session_id,
            "profile":           session["profile"],
            "created_at":        session["created_at"],
            "status":            session["status"],
            "total_rounds":      len(rounds),
            "scored_rounds":     len(scores),
            "average_score":     avg,
            "category_averages": cat_avg,
            "weakest_category":  weakest,
            "score_trend":       scores,
            "rounds":            rounds,
        }

    def get_all_sessions(self) -> List[dict]:
        """Return all sessions sorted by creation date (newest first)."""
        return sorted(
            self.sessions.values(),
            key=lambda s: s["created_at"],
            reverse=True,
        )

    def get_recent_sessions(self, n: int = 5) -> List[dict]:
        return self.get_all_sessions()[:n]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self) -> None:
        """Write all sessions to disk."""
        self.session_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.session_file, "w", encoding="utf-8") as f:
            json.dump(self.sessions, f, indent=2, ensure_ascii=False)

    def _load(self) -> None:
        """Load sessions from disk if the file exists."""
        if self.session_file.exists():
            try:
                with open(self.session_file, "r", encoding="utf-8") as f:
                    self.sessions = json.load(f)
            except (json.JSONDecodeError, OSError):
                self.sessions = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def profile_to_dict(profile) -> dict:
        """Convert a CandidateProfile dataclass to a plain dict."""
        return {
            "name":             profile.name,
            "role":             profile.role,
            "experience_level": profile.experience_level,
            "focus_areas":      profile.focus_areas,
            "company":          profile.company,
            "interview_date":   profile.interview_date,
        }
