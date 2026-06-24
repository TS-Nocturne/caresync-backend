from patient_care_backend.services.context_indexer import build_patient_context_chunks


def test_build_chunks_includes_allergies_and_baseline():
    chunks = build_patient_context_chunks({
        "patient_id": "p-1",
        "first_name": "Test",
        "nickname": "ลุงทดสอบ",
        "underlying_diseases": ["เบาหวาน"],
        "allergies": ["Penicillin"],
        "mobility_status": "WHEELCHAIR",
        "baseline_systolic": 140,
        "baseline_diastolic": 90,
        "medications": [{"name": "Amlodipine", "dosage": "5mg", "time_of_day": ["MORNING"], "instruction": "after meal"}],
    })

    texts = " ".join(c["text"] for c in chunks)
    assert "Penicillin" in texts
    assert "140/90" in texts
    assert "Amlodipine" in texts
    assert "wheelchair" in texts.lower() or "รถเข็น" in texts
