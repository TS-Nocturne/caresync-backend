from patient_care_backend.services.vital_validation import validate_vitals, validation_needs_confirmation


def test_temperature_typo_375():
    issues = validate_vitals({"temperature_c": 375})
    assert validation_needs_confirmation(issues)
    assert any(i.field == "temperature_c" for i in issues)


def test_systolic_typo_12():
    issues = validate_vitals({"systolic": 12, "diastolic": 80})
    assert validation_needs_confirmation(issues)
    assert any(i.suggested_value == 120 for i in issues if i.field == "systolic")


def test_normal_vitals_pass():
    issues = validate_vitals({"systolic": 120, "diastolic": 80, "temperature_c": 36.5, "spo2": 98})
    assert not validation_needs_confirmation(issues)
