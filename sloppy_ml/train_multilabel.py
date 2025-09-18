# train_multilabel.py
import argparse, pandas as pd, numpy as np, joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, classification_report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--min_positives", type=int, default=1,
                    help="Drop labels with fewer than this many positives.")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.dataset_csv)

    # Build single text column to avoid any custom function in the pipeline
    for col in ["comments", "attributes_excerpt"]:
        if col not in df.columns:
            df[col] = ""
    df["__text__"] = (df["comments"].fillna("") + " " + df["attributes_excerpt"].fillna("")).str.strip()

    # Infer label columns (UPPERCASE 0/1)
    candidate_labels = [c for c in df.columns if c.isupper() and set(df[c].dropna().unique()) <= {0,1}]
    # Keep labels with enough positives
    keep = []
    pos_counts = {}
    for c in candidate_labels:
        pc = int(df[c].sum())
        pos_counts[c] = pc
        if pc >= args.min_positives:
            keep.append(c)
    print("Label positives in full dataset:", pos_counts)
    if not keep:
        raise SystemExit("No labels meet min_positives threshold. Lower --min_positives or add data.")
    y = df[keep].values

    # Numeric flags
    num_cols = [c for c in df.columns if c.startswith("flag__")]
    if "attributes_filled" in df.columns:
        num_cols.append("attributes_filled")

    # Preprocess: TF-IDF on __text__, scale numeric flags
    pre = ColumnTransformer(
        transformers=[
            ("txt", TfidfVectorizer(max_features=20000, ngram_range=(1,2)), "__text__"),
            ("num", StandardScaler(with_mean=False), num_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3
    )

    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0, solver="liblinear")
    )
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=args.test_size, random_state=42)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    micro_f1 = f1_score(y_test, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    print("Micro-F1:", micro_f1, " Macro-F1:", macro_f1)
    print(classification_report(y_test, y_pred, target_names=keep, zero_division=0))

    joblib.dump({"pipe": pipe,
                 "label_cols": keep,
                 "num_cols": num_cols},
                out/"multilabel_model.joblib")
    print("Saved model to", out/"multilabel_model.joblib")

if __name__ == "__main__":
    main()
