import re
import numpy as np

def run_rules(rec: dict, schema: dict, form_type: str):
    cfg = schema["form_types"].get(form_type, {})
    req = cfg.get("required_fields", [])
    url_fields = cfg.get("url_fields", [])
    bool_expected_true = set(cfg.get("bool_expected_true", []))
    numeric_min = cfg.get("numeric_min", {})
    min_length = cfg.get("min_length", {})

    findings = []

    # Required fields
    for f in req:
        v = rec.get(f, None)
        if v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and not v.strip()):
            findings.append({
                "category": "Attributes",
                "issue": f"Required field '{f}' missing",
                "action_request": f"Fill in the '{f}' field completely."
            })

    # String min length
    for f, m in min_length.items():
        v = rec.get(f, "")
        if not isinstance(v, str) or len(v.strip()) < m:
            findings.append({
                "category": "Attributes",
                "issue": f"Text for '{f}' too short",
                "action_request": f"Provide a more descriptive '{f}' (min {m} characters)."
            })

    # Numeric min
    for f, m in numeric_min.items():
        v = rec.get(f, None)
        try:
            bad = (v is None) or (float(v) < float(m))
        except Exception:
            bad = True
        if bad:
            findings.append({
                "category": "Attributes",
                "issue": f"Numeric value '{f}' below minimum ({m}) or missing",
                "action_request": f"Enter a valid number for '{f}' (>= {m})."
            })

    # URL contains rule
    for spec in url_fields:
        f = spec.get("field")
        substrs = spec.get("contains_any", [])
        v = rec.get(f, "")
        if not isinstance(v, str) or not any(s.lower() in v.lower() for s in substrs):
            findings.append({
                "category": "Evidence",
                "issue": f"Missing/invalid link in '{f}'",
                "action_request": f"Attach a valid URL containing one of: {', '.join(substrs)}."
            })

    # Domain-specific: materials booked
    # If we see material_code/quantity/specification missing for a quote_stub row, raise a materials flag
    if form_type == "quote_stub":
        # If any row is missing the trio, we raise the issue
        needed = ["material_code", "specification", "quantity"]
        if any(rec.get(k) in (None, "", float('nan')) for k in needed):
            findings.append({
                "category": 'Materials',
                "issue": "Materials not fully specified for invoicing",
                "action_request": "Provide material code, specification and quantity; ensure the booking order is created."
            })

    return findings