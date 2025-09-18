import argparse, json, os
import pandas as pd
from sklearn.ensemble import IsolationForest
from joblib import dump
from .features import featureize_record

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records_csv", required=True, help="CSV with flattened records (one row per item)")
    ap.add_argument("--schema_json", required=True, help="Schema JSON")
    ap.add_argument("--form_type", required=True, choices=["quote_stub","service_report"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--contamination", type=float, default=0.15)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    schema = json.load(open(args.schema_json))

    df = pd.read_csv(args.records_csv)
    feats = []
    for _, row in df.iterrows():
        rec = row.to_dict()
        feats.append(featureize_record(rec, schema, args.form_type))
    X = pd.DataFrame(feats).fillna(0.0)

    model = IsolationForest(n_estimators=200, contamination=args.contamination, random_state=42)
    model.fit(X)

    dump(model, os.path.join(args.out_dir, f"model_{args.form_type}.joblib"))
    X.to_csv(os.path.join(args.out_dir, f"feats_{args.form_type}.csv"), index=False)
    print("Saved model and features.")

if __name__ == "__main__":
    main()