# Interview Trainer Agent 🎯

> An AI-powered interview preparation assistant built with **IBM Granite** foundation models on **IBM Cloud Lite** and a **RAG (Retrieval-Augmented Generation)** pipeline.

---

## Architecture

```
User Input  (Streamlit UI)
      │
      ▼
CandidateProfile  ─────────────────────────────────────────────────┐
      │                                                             │
      ▼                                                             │
InterviewTrainerAgent (src/agent.py)                               │
      │                                                             │
      ├──► InterviewRetriever (src/rag_pipeline.py)                │
      │         │                                                   │
      │         ├── sentence-transformers  (local embeddings)      │
      │         └── ChromaDB               (vector store)          │
      │                                                             │
      └──► GraniteLLM (src/granite_llm.py)                        │
                │                                                   │
                └── IBM watsonx.ai  ←  ibm/granite-3-8b-instruct  ◄┘
                        │
                        ▼
          Response  (Questions / Feedback / Strategy)
```

---

## Features

| Feature | Description |
|---|---|
| 🎯 Generate Questions | Role-specific technical + HR question sets |
| 📝 Evaluate Answer | STAR-based scoring with actionable feedback |
| 🎤 Mock Interview | Live Q&A session with instant AI feedback |
| 📅 Prep Strategy | Day-by-day preparation plan |
| 📄 Resume Analyser | Extract skills, estimate level, predict questions |

---

## Prerequisites

- Python 3.9+
- IBM Cloud account (free Lite tier)
- watsonx.ai project

---

## IBM Cloud Setup (Step-by-Step)

### Step 1 — Create IBM Cloud Account
1. Go to [cloud.ibm.com](https://cloud.ibm.com) → **Create a free account**
2. Select **Lite** plan (no credit card required)

### Step 2 — Create a watsonx.ai Project
1. Navigate to **watsonx.ai** → [dataplatform.cloud.ibm.com](https://dataplatform.cloud.ibm.com)
2. Click **New Project** → select **Create an empty project**
3. Give it a name (e.g., `interview-trainer`) → **Create**
4. Copy the **Project ID** from Project → Manage → General

### Step 3 — Associate watsonx.ai Runtime
1. Inside your project → **Manage** tab → **Services & Integrations**
2. Click **Associate Service** → search for **Watson Machine Learning**
3. Create a **Lite** plan instance → **Associate**

### Step 4 — Generate IBM Cloud API Key
1. IBM Cloud top-right → **Manage** → **Access (IAM)**
2. Left sidebar → **API Keys** → **Create an IBM Cloud API key**
3. Name it `interview-trainer-key` → **Create** → **Copy** (shown only once!)

### Step 5 — Get your watsonx.ai URL
| Region | URL |
|---|---|
| Dallas (us-south) | `https://us-south.ml.cloud.ibm.com` |
| Frankfurt (eu-de) | `https://eu-de.ml.cloud.ibm.com` |
| Tokyo (jp-tok) | `https://jp-tok.ml.cloud.ibm.com` |

---

## Local Setup

```bash
# 1. Clone or extract the project
cd Project/interview_trainer_agent

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure credentials
cp .env.example .env
# Edit .env — fill in WATSONX_API_KEY and WATSONX_PROJECT_ID

# 5. Run smoke test (validates everything without UI)
python test_agent.py

# 6. Launch Streamlit UI
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## Environment Variables

| Variable | Description | Required |
|---|---|---|
| `WATSONX_API_KEY` | IBM Cloud API key | ✅ |
| `WATSONX_PROJECT_ID` | watsonx.ai project ID | ✅ |
| `WATSONX_URL` | Regional endpoint URL | ✅ |
| `GRANITE_MODEL_ID` | Granite model to use | Optional |
| `CHROMA_PERSIST_DIR` | ChromaDB storage path | Optional |
| `MAX_RETRIEVED_DOCS` | RAG top-k documents | Optional |
| `MAX_NEW_TOKENS` | LLM max output tokens | Optional |
| `TEMPERATURE` | LLM creativity (0–1) | Optional |

---

## IBM Granite Models (Cloud Lite)

| Model ID | Description | Best For |
|---|---|---|
| `ibm/granite-3-8b-instruct` | ⭐ Recommended — fast & capable | All tasks |
| `ibm/granite-13b-instruct-v2` | Larger, richer responses | Complex analysis |
| `ibm/granite-20b-multilingual` | 20B, multilingual | Non-English resumes |
| `ibm/granite-3b-code-instruct` | Code-focused | Technical interviews |

---

## Project Structure

```
interview_trainer_agent/
│
├── app.py                          ← Streamlit frontend (5 tabs)
├── test_agent.py                   ← Smoke test / CLI demo
├── requirements.txt
├── .env.example                    ← Credentials template
│
├── src/
│   ├── __init__.py
│   ├── agent.py                    ← Orchestration (core agent)
│   ├── granite_llm.py              ← IBM Granite LLM wrapper
│   ├── rag_pipeline.py             ← ChromaDB vector store + retriever
│   └── prompt_templates.py         ← All Granite prompt templates
│
├── data/
│   └── knowledge_base/
│       ├── software_engineer_questions.json
│       ├── data_scientist_questions.json
│       └── hr_behavioral_questions.json
│
└── vector_store/
    └── chroma_db/                  ← Auto-created on first run
```

---

## Sample Prompt Template (Granite Instruction Format)

```
You are an expert Interview Trainer Agent powered by IBM Granite.

## Candidate Profile
- Name: Priya Sharma
- Job Role: Software Engineer
- Experience Level: Mid-level (3–5 yrs)
- Focus Areas: technical, system design, behavioral

## Retrieved Context (RAG)
[1] TECHNICAL | Intermediate
Q: Explain the CAP theorem in distributed systems.
Key Points: Consistency, Availability, Partition Tolerance …

## Task
Generate 6 tailored interview questions with model answers …
```

---

## RAG Pipeline Detail

1. **Ingestion** — JSON knowledge base files loaded at startup
2. **Embedding** — `all-MiniLM-L6-v2` (sentence-transformers, runs locally)
3. **Storage** — ChromaDB persistent vector store (cosine similarity)
4. **Retrieval** — Top-k semantic search filtered by role + category
5. **Generation** — Retrieved context injected into Granite prompt

---

## Technologies Used

| Component | Technology |
|---|---|
| LLM | IBM Granite (`ibm/granite-3-8b-instruct`) |
| LLM Runtime | IBM watsonx.ai (Cloud Lite) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | ChromaDB (persistent, local) |
| Frontend | Streamlit |
| Auth | IBM Cloud API Key |
| Language | Python 3.9+ |

---

## License
MIT — Built for IBM Cloud Lite × IBM Granite demo.
