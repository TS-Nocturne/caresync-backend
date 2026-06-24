# Patient Care Backend

Backend "brain" for the patient-care project. It follows `pland/PLAN.md` and uses the LangGraph patterns from `pland/langgraph.md`:

1. Retrieve patient medical context from Pinecone/Gemini embeddings.
2. Evaluate symptoms and vitals with Gemini structured output.
3. Route `warning` and `critical` cases to a human-in-the-loop pause.
4. Resume the graph after the family decision and execute the selected action.

## Run

```powershell
uv sync
uv run uvicorn patient_care_backend.main:app --reload
```

Copy `.env.example` to `.env` and fill in real keys before enabling Gemini/Pinecone.

## API

- `POST /brain/assess` starts the graph from a nurse symptom report.
- `POST /brain/{thread_id}/decision` writes `family_decision` and resumes the graph.
- `GET /brain/{thread_id}/state` returns the current checkpoint state.

When keys are missing, the app uses deterministic local fallbacks so the backend can be tested without external services.
