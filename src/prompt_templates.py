"""
Prompt Templates for the Interview Trainer Agent
All prompts follow IBM Granite's recommended instruction-following format.
"""

# ---------------------------------------------------------------------------
# System persona injected at the top of every prompt
# ---------------------------------------------------------------------------
SYSTEM_PERSONA = """\
You are an expert Interview Trainer Agent powered by IBM Granite. \
You help candidates prepare for job interviews by generating tailored \
questions, model answers, and actionable improvement tips. \
Always be encouraging, specific, and professional.\
"""


# ---------------------------------------------------------------------------
# Template 1 – Generate a custom interview question set
# ---------------------------------------------------------------------------
QUESTION_GENERATION_TEMPLATE = """\
{system}

## Candidate Profile
- **Name**: {name}
- **Job Role**: {role}
- **Experience Level**: {experience_level}
- **Focus Areas**: {focus_areas}

## Retrieved Context (similar questions from knowledge base)
{context}

## Task
Based on the candidate profile and context above, generate a set of \
{num_questions} tailored interview questions. Include a mix of:
- Technical questions specific to the role
- Behavioral questions using the STAR framework  
- HR / situational questions

For each question, provide:
1. The question text
2. What the interviewer is really assessing
3. A model answer outline (3-5 key points)
4. One concrete improvement tip

Format each question clearly with headers.
"""


# ---------------------------------------------------------------------------
# Template 2 – Evaluate a user's answer and provide feedback
# ---------------------------------------------------------------------------
ANSWER_EVALUATION_TEMPLATE = """\
{system}

## Interview Context
- **Role**: {role}
- **Experience Level**: {experience_level}
- **Question**: {question}

## Candidate's Answer
{user_answer}

## Expert Reference Answer
{model_answer}

## Task
Evaluate the candidate's answer as a senior interviewer would. Provide:

### ✅ Strengths
List 2-3 things the candidate did well.

### ⚠️ Areas for Improvement
List 2-3 specific gaps compared to the ideal answer.

### 📈 Improvement Tips
Give 2-3 concrete, actionable tips to improve this answer.

### 🏅 Score
Rate the answer out of 10 with a brief justification.

Be constructive and specific. Reference the candidate's actual words.
"""


# ---------------------------------------------------------------------------
# Template 3 – Generate a full preparation strategy
# ---------------------------------------------------------------------------
PREP_STRATEGY_TEMPLATE = """\
{system}

## Candidate Profile
- **Name**: {name}
- **Target Role**: {role}
- **Experience Level**: {experience_level}
- **Interview Date**: {interview_date}
- **Company / Industry**: {company}

## Task
Create a structured, day-by-day interview preparation plan for this candidate. Include:

### 📋 Key Topics to Study
List the most important technical and soft-skill topics for this role.

### 📅 Preparation Timeline
Break down a preparation schedule (up to 7 days).

### 🔑 Top 5 Questions to Master
The must-know questions for this specific role and experience level.

### 💡 Confidence Tips
3 practical tips to build interview confidence.

### 🚀 Day-of-Interview Checklist
A quick checklist for the day of the interview.

Keep the tone motivating and the plan realistic.
"""


# ---------------------------------------------------------------------------
# Template 4 – Extract topics from a resume / job description
# ---------------------------------------------------------------------------
RESUME_ANALYSIS_TEMPLATE = """\
{system}

## Input Text (Resume or Job Description)
{resume_text}

## Task
Analyze the text above and extract:

### 🛠️ Technical Skills Identified
List all technical skills, tools, and technologies mentioned.

### 📊 Experience Level Assessment
Estimate the experience level (Fresher / Junior / Mid / Senior / Lead) with reasoning.

### 🎯 Likely Interview Topics
List the top 5-7 topics the interviewer will likely probe based on this profile.

### ❓ Predicted Tough Questions
Generate 3 challenging questions this specific candidate might face, based on their profile.

### 💼 Role Recommendations
Suggest 2-3 job roles this candidate is best suited for.

Be precise and use the actual content from the input text.
"""


# ---------------------------------------------------------------------------
# Template 5 – Mock interview question (single, with context)
# ---------------------------------------------------------------------------
SINGLE_QUESTION_TEMPLATE = """\
{system}

## Context
- **Role**: {role}
- **Category**: {category}
- **Difficulty**: {difficulty}
- **Retrieved Similar Question**: {retrieved_question}

## Task
Generate ONE original interview question for this candidate.  
Then provide:
- **What is being assessed**: (1 sentence)
- **Model Answer Key Points**: (3-5 bullet points)
- **Red Flags**: Things that would signal a weak answer (2-3 points)
"""


# ---------------------------------------------------------------------------
# Helper – build a formatted context block from retrieved documents
# ---------------------------------------------------------------------------
def format_retrieval_context(retrieved_docs: list) -> str:
    """Convert retrieved documents into a formatted context string for prompts."""
    if not retrieved_docs:
        return "No similar questions retrieved from knowledge base."

    lines = []
    for i, doc in enumerate(retrieved_docs, 1):
        lines.append(f"**[{i}] {doc['category'].upper()} | {doc['difficulty'].capitalize()}**")
        lines.append(f"Q: {doc['question']}")
        lines.append(f"Key Points: {doc['model_answer'][:200]}…")
        lines.append("")

    return "\n".join(lines)
