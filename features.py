import re
import numpy as np
import pandas as pd

def _is_missing(v):
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    if isinstance(v, str) and len(v.strip()) == 0:
        return True
    return False

def featureize_record(rec: dict, schema: dict, form_type: str) -> dict:
    ft = {}

    cfg = schema["form_types"].get(form_type, {})
    req = cfg.get("required_fields", [])
    min_length = cfg.get("min_length", {})
    numeric_min = cfg.get("numeric_min", {})
    url_fields = cfg.get("url_fields", [])
    bool_expected_true = set(cfg.get("bool_expected_true", []))

    # basic missingness on required fields
    for f in req:
        v = rec.get(f, None)
        ft[f"missing__{f}"] = 1.0 if _is_missing(v) else 0.0

    # string length checks
    for f, m in min_length.items():
        v = rec.get(f, "")
        vlen = len(v.strip()) if isinstance(v, str) else 0
        ft[f"short__{f}"] = 1.0 if vlen < m else 0.0

    # numeric min checks
    for f, m in numeric_min.items():
        v = rec.get(f, None)
        bad = False
        try:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                bad = True
            else:
                bad = float(v) < float(m)
        except Exception:
            bad = True
        ft[f"ltmin__{f}"] = 1.0 if bad else 0.0

    # URL content checks
    for spec in url_fields:
        f = spec.get("field")
        substrs = spec.get("contains_any", [])
        v = rec.get(f, "")
        present = False
        if isinstance(v, str):
            s = v.lower()
            for sub in substrs:
                if sub.lower() in s:
                    present = True
                    break
        ft[f"url_contains__{f}"] = 0.0 if present else 1.0  # flag 1.0 if missing or doesn't contain expected

    # booleans expected True
    for f in bool_expected_true:
        v = rec.get(f, False)
        bad = not bool(v)
        ft[f"false__{f}"] = 1.0 if bad else 0.0

    # aggregate
    req_flags = [ft[k] for k in ft if k.startswith("missing__")]
    ft["agg_missing_required"] = float(np.mean(req_flags)) if req_flags else 0.0
    ft["agg_flags_count"] = float(sum(ft.values()))

    return ft