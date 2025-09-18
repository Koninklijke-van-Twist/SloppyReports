# predict_multilabel.py
import argparse, json, glob, joblib, yaml, re
from pathlib import Path
import pandas as pd
import numpy as np


def load_taxonomy(tax_path):
    with open(tax_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    return pd.DataFrame(y["labels"])

def load_structured(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def rec_to_df(s, num_cols):
    flags = s.get("flags", {})
    comments = s.get("excerpts", {}).get("comments", "") or ""
    attrs    = s.get("excerpts", {}).get("attributes", "") or ""
    row = {
        "report_id": s.get("report_id"),
        "attributes_filled": bool(s.get("attributes_filled", False)),
        "comments": comments,
        "attributes_excerpt": attrs,
        "__text__": (comments + " " + attrs).strip(),
    }
    # ensure all numeric feature columns exist
    for c in num_cols:
        row[c] = bool(flags.get(c.replace("flag__",""), False)) if c.startswith("flag__") else row.get(c, False)
    return pd.DataFrame([row])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structured_glob", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--taxonomy_yaml", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tax = load_taxonomy(args.taxonomy_yaml)
    mdl = joblib.load(args.model_path)
    pipe = mdl["pipe"]; labels = mdl["label_cols"]; num_cols = mdl["num_cols"]

    for path in glob.glob(args.structured_glob):
        s = load_structured(path)
        df = rec_to_df(s, num_cols)

        # Fill any missing expected numeric columns with False/0
        for c in num_cols:
            if c not in df.columns:
                df[c] = False

        # Predict probabilities if available
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(df)[0]
        else:
            dec = pipe.decision_function(df)[0]
            proba = 1/(1+np.exp(-dec))

        chosen = []
        for i, code in enumerate(labels):
            p = float(proba[i]) if hasattr(proba, "__len__") else float(proba)
            if p >= args.threshold:
                chosen.append((code, p))

        rid = s.get("report_id")
        rows = []
        for code, p in chosen:
            row = {"report_id": rid, "label_code": code, "confidence": round(p,3)}
            hit = tax[tax["code"]==code].head(1)
            if not hit.empty:
                row["issue"] = hit["issue"].iloc[0]
                row["action_request"] = hit["action"].iloc[0]
            rows.append(row)

        out_csv = out / f"predicted_findings_{rid}.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print("Wrote", out_csv)

if __name__ == "__main__":
    main()
