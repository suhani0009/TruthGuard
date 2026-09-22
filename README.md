# TruthLens

TruthLens is a final-year-project prototype for inspecting factual claims in an AI-generated response and retrieving related evidence.

## Current prototype

The seminar build implements only:

1. Claim extraction (Stage 1)
2. BGE embeddings and Qdrant evidence retrieval (Stage 2)
3. FastAPI and a React interface for displaying claims and evidence

Verification, trust scores, correction, Tavily, and MongoDB are intentionally not implemented yet.

## Start the backend

From the project root, activate the Python environment and run:

```powershell
cd "C:\Users\SUHANI\.cagent\Desktop\Final Year Project\New folder"
.\.venv\Scripts\Activate.ps1
cd backend
python -m uvicorn app.main:app --reload


pytest tests/test_verification_service.py -q -s
curl -X POST "http://127.0.0.1:8000/api/analyze" -H "Content-Type: application/json" -d "{\"response\":\"Google was founded in 1999.\"}"
```

The API is available at `http://localhost:8000`.

- `GET /api/health`
- `POST /api/analyze` with `{ "response": "..." }`

## Start the frontend

In a second terminal:

```powershell
cd frontend
npm run dev


npm install
```

Open `http://localhost:5173`.

## Demo input

```text
Google was founded in 1999. The Earth revolves around the Sun.
```

The Google claim retrieves the local evidence stating that Google was founded in 1998; the Earth claim retrieves the matching Earth-orbit evidence.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests -q
```
