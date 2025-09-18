import argparse, json, os, pandas as pd
from joblib import load
from .features import featureize_record
from .rules import run_rules

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--schema_json", required=True)
    ap.add_argument("--form_type", required=True, choices=["quote_stub","service_report"])
    ap.add_argument("--out_findings_csv", required=True)
    ap.add_argument("--model_path", default=None)  # optional isolation forest
    args = ap.parse_args()

    schema = json.load(open(args.schema_json))
    df = pd.read_csv(args.input_csv)

    findings_rows = []
    model = None
    if args.model_path and os.path.exists(args.model_path):
        try:
            model = load(args.model_path)
        except Exception:
            model = None

    # Build features and findings for each row independently
    for idx, row in df.iterrows():
        rec = row.to_dict()

        # Rule-based findings (deterministic)
        rb_findings = run_rules(rec, schema, args.form_type)

        # Optional anomaly score
        feat = featureize_record(rec, schema, args.form_type)
        anom_score = None
        if model is not None:
            import pandas as pd
            X = pd.DataFrame([feat]).fillna(0.0)
            # IsolationForest decision_function: the lower, the more abnormal. We'll convert to a 0..1 "sloppiness" score.
            raw = -model.decision_function(X)[0]  # higher => more sloppy
            anom_score = max(0.0, float(raw))

        for f in rb_findings:
            findings_rows.append({
                "row_index": idx,
                "category": f.get("category"),
                "issue": f.get("issue"),
                "action_request": f.get("action_request"),
                "sloppiness_score": anom_score
            })

        # If there were no rule-based findings but the model flags it as abnormal, still log a generic note
        if not rb_findings and anom_score is not None and anom_score > 0.5:
            findings_rows.append({
                "row_index": idx,
                "category": "Anomaly",
                "issue": "Potentially incomplete/inconsistent row",
                "action_request": "Review this row; model flagged it as unusual compared to good history.",
                "sloppiness_score": anom_score
            })

    out = pd.DataFrame(findings_rows)
    out.to_csv(args.out_findings_csv, index=False)
    print(f"Wrote findings to {args.out_findings_csv}")

if __name__ == "__main__":
    main()