from typing import Protocol

from patient_care_backend.config import Settings
from patient_care_backend.schemas import PatientState


class PatientContextRetriever(Protocol):
    def retrieve(self, state: PatientState) -> str:
        """Return relevant clinical context for the current patient state."""


class FallbackPatientContextRetriever:
    def retrieve(self, state: PatientState) -> str:
        medications = ", ".join(state.get("current_medications", [])) or "none reported"
        return (
            "No external knowledge base is configured. "
            f"Review medication list locally before acting: {medications}."
        )


class PineconePatientContextRetriever:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._index = None
        self._embeddings = None

    def _connect(self) -> None:
        if self._index is not None and self._embeddings is not None:
            return

        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from pinecone import Pinecone

        pc = Pinecone(api_key=self._settings.pinecone_api_key)
        self._index = pc.Index(self._settings.pinecone_index_name)
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=self._settings.gemini_embedding_model,
            google_api_key=self._settings.google_api_key,
        )

    def retrieve(self, state: PatientState) -> str:
        self._connect()
        assert self._index is not None
        assert self._embeddings is not None

        query = (
            f"Symptoms: {', '.join(state.get('symptoms', []))}. "
            f"Medications: {', '.join(state.get('current_medications', []))}. "
            f"Patient context lookup for care-coordination review."
        )
        vector = self._embeddings.embed_query(query)
        patient_results = self._index.query(
            vector=vector,
            top_k=3,
            include_metadata=True,
            filter={"scope": "patient_context", "patient_id": state["patient_id"]},
        )

        knowledge_results = self._index.query(
            vector=vector,
            top_k=5,
            include_metadata=True,
            filter={"scope": "knowledge_base"},
        )

        patient_context = "\n".join(
            match.get("metadata", {}).get("text", "")
            for match in patient_results.get("matches", [])
            if match.get("metadata", {}).get("text")
        )
        knowledge_context = "\n".join(
            (
                f"[{match.get('metadata', {}).get('source')} "
                f"p.{match.get('metadata', {}).get('page')}] "
                f"{match.get('metadata', {}).get('text')}"
            )
            for match in knowledge_results.get("matches", [])
            if match.get("metadata", {}).get("text")
        )

        sections = []
        if patient_context:
            sections.append(f"Patient-specific context:\n{patient_context}")
        if knowledge_context:
            sections.append(f"Clinical knowledge base:\n{knowledge_context}")
        return "\n\n".join(sections)


def build_context_retriever(settings: Settings) -> PatientContextRetriever:
    if settings.enable_external_ai and settings.pinecone_api_key and settings.google_api_key:
        return PineconePatientContextRetriever(settings)
    return FallbackPatientContextRetriever()
