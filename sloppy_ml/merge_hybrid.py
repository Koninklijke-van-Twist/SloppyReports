import argparse, pandas as pd
from pathlib import Path
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule_findings_csv", required=True)
    ap.add_argument("--ml_findings_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()
    rule = pd.read_csv(args.rule_findings_csv) if Path(args.rule_findings_csv).exists() else pd.DataFrame()
    ml = pd.read_csv(args.ml_findings_csv) if Path(args.ml_findings_csv).exists() else pd.DataFrame()
    if not rule.empty:
        rule["source"] = "rules"
        rule["label_code"] = rule["issue"].str.upper().str.replace(r"[^A-Z0-9]+","_", regex=True)
        rule["confidence"] = 1.0
    if not ml.empty:
        ml["source"] = "ml"
    combined = pd.concat([rule, ml], ignore_index=True, sort=False)
    if not combined.empty:
        combined["key"] = combined["label_code"].fillna("") + "|" + combined["issue"].fillna("")
        combined = combined.sort_values(["source","confidence"], ascending=[True, False]).drop_duplicates("key", keep="first")
        combined.drop(columns=["key"], inplace=True)
    combined.to_csv(args.out_csv, index=False)
    print("Wrote", args.out_csv)
if __name__ == "__main__":
    main()