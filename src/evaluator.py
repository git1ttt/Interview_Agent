"""
Structured Evaluator — rubric-based answer scoring engine.

Parses Granite's free-text feedback to extract a numeric score,
then builds a structured EvaluationResult with per-dimension ratings.

Dimensions scored (each 0–10):
  - Content Coverage   : did the answer address key points?
  - Clarity            : was it well-structured and clear?
  - Depth              : was the level of detail appropriate?
  - Examples           : were concrete examples or evidence provided?
  - STAR Compliance    : (behavioral only) did it follow STAR format?

Usage:
    ev = Evaluator(llm)
    result = ev.score(profile, question, user_answer, model_answer, category)
    print(result.overall_score)   # float 0–10
    print(result.summary)         # markdown string
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Optional

from .granite_llm import GraniteLLM
from .prompt_templates import SYSTEM_PERSONA


# ---------------------------------------------------------------------------
# Rubric prompt
# ---------------------------------------------------------------------------
RUBRIC_PROMPT = """\
{system}

## Evaluation Task
You are scoring a candidate's interview answer using a strict rubric.

### Interview Details
- Role            : {role}
- Experience Level: {experience_level}
- Category        : {category}
- Question        : {question}

### Candidate's Answer
{user_answer}

### Reference / Model Answer
{model_answer}

### Scoring Rubric (each dimension 0–10)
Score each dimension. Be strict but fair. Senior roles should be held to higher standards.

| Dimension | Criteria |
|---|---|
| Content Coverage | Did the answer address the key points of the question? |
| Clarity | Was the answer well-structured, easy to follow, free of confusion? |
| Depth | Was the level of technical or situational detail appropriate for the experience level? |
| Examples | Were concrete examples, numbers, or evidence provided? |
| STAR Compliance | (Behavioral/HR only) Did the answer follow Situation-Task-Action-Result? For technical questions, score this 5/10 as N/A. |

### Output Format (use EXACTLY this format)
CONTENT_COVERAGE: <score>/10
CLARITY: <score>/10
DEPTH: <score>/10
EXAMPLES: <score>/10
STAR_COMPLIANCE: <score>/10
OVERALL: <score>/10

STRENGTHS:
- <strength 1>
- <strength 2>

IMPROVEMENTS:
- <improvement 1>
- <improvement 2>

COACH_TIP: <one sentence actionable tip>
"""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class EvaluationResult:
    question:          str
    user_answer:       str
    category:          str
    overall_score:     float = 0.0
    dimensions: Dict[str, float] = field(default_factory=dict)
    strengths:         list  = field(default_factory=list)
    improvements:      list  = field(default_factory=list)
    coach_tip:         str   = ""
    raw_feedback:      str   = ""

    @property
    def grade(self) -> str:
        s = self.overall_score
        if s >= 9:   return "🏆 Excellent"
        if s >= 7:   return "✅ Good"
        if s >= 5:   return "⚠️ Needs Work"
        return "❌ Needs Significant Improvement"

    @property
    def summary(self) -> str:
        """Render a clean markdown summary card."""
        dims = "\n".join(
            f"| {k.replace('_', ' ').title()} | {'█' * int(v)} {'░' * (10 - int(v))} | {v}/10 |"
            for k, v in self.dimensions.items()
        )
        strengths   = "\n".join(f"- {s}" for s in self.strengths)
        improvements = "\n".join(f"- {i}" for i in self.improvements)

        return f"""### {self.grade} — Overall Score: **{self.overall_score}/10**

#### 📊 Dimension Scores
| Dimension | Progress | Score |
|---|---|---|
{dims}

#### ✅ Strengths
{strengths or '- None noted.'}

#### ⚠️ Areas for Improvement
{improvements or '- Keep it up!'}

#### 💡 Coach Tip
{self.coach_tip}
"""


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------
class Evaluator:
    """
    Uses IBM Granite to score answers against the rubric, then parses
    the structured output into an EvaluationResult.
    """

    def __init__(self, llm: GraniteLLM):
        self.llm = llm

    def score(
        self,
        role:             str,
        experience_level: str,
        question:         str,
        user_answer:      str,
        model_answer:     str = "",
        category:         str = "technical",
    ) -> EvaluationResult:
        """
        Run the rubric evaluation and return a structured EvaluationResult.
        """
        prompt = RUBRIC_PROMPT.format(
            system           = SYSTEM_PERSONA,
            role             = role,
            experience_level = experience_level,
            category         = category,
            question         = question,
            user_answer      = user_answer,
            model_answer     = model_answer or "Not provided — evaluate on general best practices.",
        )

        raw = self.llm.generate(prompt)
        return self._parse(question, user_answer, category, raw)

    # ------------------------------------------------------------------
    # Parser
    # ------------------------------------------------------------------
    def _parse(self, question: str, user_answer: str, category: str, raw: str) -> EvaluationResult:
        """Extract structured fields from Granite's free-text rubric output."""

        def extract_score(label: str) -> float:
            pattern = rf"{label}:\s*(\d+(?:\.\d+)?)\s*/\s*10"
            m = re.search(pattern, raw, re.IGNORECASE)
            return float(m.group(1)) if m else 5.0

        dimensions = {
            "content_coverage": extract_score("CONTENT_COVERAGE"),
            "clarity":          extract_score("CLARITY"),
            "depth":            extract_score("DEPTH"),
            "examples":         extract_score("EXAMPLES"),
            "star_compliance":  extract_score("STAR_COMPLIANCE"),
        }
        overall = extract_score("OVERALL")
        if overall == 5.0 and dimensions:
            # Fall back to average of dimensions if OVERALL wasn't parsed
            overall = round(sum(dimensions.values()) / len(dimensions), 1)

        # Extract bulleted lists
        strengths    = self._extract_bullets(raw, "STRENGTHS")
        improvements = self._extract_bullets(raw, "IMPROVEMENTS")

        # Coach tip
        tip_m = re.search(r"COACH_TIP:\s*(.+)", raw)
        tip   = tip_m.group(1).strip() if tip_m else ""

        return EvaluationResult(
            question      = question,
            user_answer   = user_answer,
            category      = category,
            overall_score = overall,
            dimensions    = dimensions,
            strengths     = strengths,
            improvements  = improvements,
            coach_tip     = tip,
            raw_feedback  = raw,
        )

    @staticmethod
    def _extract_bullets(text: str, section: str) -> list:
        """Extract '-' or '*' bullet lines following a section header."""
        pattern = rf"{section}:\s*\n((?:\s*[-*]\s*.+\n?)+)"
        m       = re.search(pattern, text, re.IGNORECASE)
        if not m:
            return []
        lines = re.findall(r"[-*]\s*(.+)", m.group(1))
        return [l.strip() for l in lines if l.strip()]
