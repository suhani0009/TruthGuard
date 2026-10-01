# TruthGuard

**Evidence-Based Correction for Trustworthy LLMs**

TruthGuard is a final-year project prototype that checks factual claims made by Large Language Models (LLMs) using external web evidence.

## Current Prototype

The current backend implements:

1. Claim extraction
2. Web evidence search using Tavily
3. BGE embeddings for semantic search
4. Qdrant evidence retrieval
5. Evidence focusing
6. DeBERTa-v3-MNLI verification
7. Claim classification:
   - Supported
   - Contradicted
   - Uncertain
8. Overall trust score calculation
9. Correction for contradicted claims
10. FastAPI backend
11. Automated backend tests

A React/Vite frontend is also included for displaying the verification results.

## System Flow

LLM Response  
↓  
Claim Extraction  
↓  
Tavily Web Search  
↓  
BGE Embeddings + Qdrant  
↓  
Relevant Evidence  
↓  
Evidence Focusing  
↓  
DeBERTa NLI Verification  
↓  
Supported / Contradicted / Uncertain  
↓  
Trust Score + Correction  
↓  
Frontend

## Technologies Used

### Backend

- Python
- FastAPI
- Tavily
- Sentence Transformers
- BAAI/bge-small-en-v1.5
- Qdrant
- DeBERTa-v3-MNLI
- Pytest

### Frontend

- React
- Vite
- JavaScript

## Project Structure

```text
TruthGuard/
├── backend/
│   ├── app/
│   ├── tests/
│   ├── .env.example
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   ├── package.json
│   └── package-lock.json
│
├── .gitignore
└── README.md

Backend Setup
Create a Python virtual environment:
python -m venv .venv

Activate it on Windows:
.venv\Scripts\activate

Install dependencies:
pip install -r backend/requirements.txt

Create:
backend/.env

Add the required API keys using backend/.env.example as a reference.
Do not upload .env or API keys to GitHub.
Start the Backend
From the project root:
uvicorn backend.app.main:app --reload

Backend:
http://localhost:8000

API Endpoints
GET  /api/health
POST /api/analyze

Example request:
{
  "response": "Google was founded in 1999."
}

Start the Frontend
Open a second terminal:
cd frontend
npm install
npm run dev

Frontend:
http://localhost:5173

Example
Input
Google was founded in 1999.
The Earth revolves around the Sun.

Verification
Google was founded in 1999.
→ CONTRADICTED

The Earth revolves around the Sun.
→ SUPPORTED

For contradicted claims, TruthGuard attempts to provide an evidence-based correction.
Trust Score
The current trust score mapping is:
SUPPORTED     → 1.0
UNCERTAIN     → 0.5
CONTRADICTED  → 0.0

The average score is converted into a percentage.
Testing
Run the backend tests:
pytest backend/tests -q

Current test status:
83 tests passed
Known Limitations
- Web search can sometimes return irrelevant evidence.
- Irrelevant evidence can occasionally produce an incorrect high-confidence NLI result.
- Evidence quality and source reliability need further improvement.
- Correction generation can fail if the LLM service is unavailable.
- Frontend integration and user experience are still being developed.
Project Goal
The goal of TruthGuard is to identify and correct factual errors in LLM-generated responses using external evidence and verification rather than relying only on another LLM to judge the response.