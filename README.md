# Patient Care Backend

Backend "brain" for the patient-care project. It follows `pland/PLAN.md` and uses the LangGraph patterns from `pland/langgraph.md`:

1. Retrieve patient care context from Pinecone/Gemini embeddings.
2. Summarize symptoms and vitals with Gemini structured output for care-coordination review.
3. Route `warning` and `critical` records to a human-in-the-loop pause.
4. Resume the graph after the family note and record the selected coordination action.

## Run

```powershell
uv sync
uv run uvicorn patient_care_backend.main:app --reload
```

Copy `.env.example` to `.env` and fill in real keys before enabling Gemini/Pinecone.

## API

- `POST /brain/assess` starts the graph from a caregiver observation report.
- `POST /brain/{thread_id}/decision` writes the family/caregiver coordination note and resumes the graph.
- `GET /brain/{thread_id}/state` returns the current checkpoint state.

When keys are missing, the app uses deterministic local fallbacks so the backend can be tested without external services.
