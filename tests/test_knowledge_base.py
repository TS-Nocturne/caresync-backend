from patient_care_backend.services.knowledge_base import _chunk_text


def test_chunk_text_keeps_overlap_for_long_documents():
    text = " ".join(f"sentence {i}." for i in range(500))

    chunks = _chunk_text(text, max_chars=500)

    assert len(chunks) > 1
    assert all(len(chunk) <= 500 for chunk in chunks)
    assert "sentence 0" in chunks[0]
