import argparse, json, glob, os, re
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import yaml

def load_taxonomy(tax_path):
    with open(tax_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    table = pd.DataFrame(y["labels"])
    return table

def sniff_read_csv(path):
    """Read CSV with , or ; and tolerate BOM."""
    # quick sniff
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        sample = f.read(2048)
    sep = ";" if sample.count(";") > sample.count(",") else ","
    return pd.read_csv(path, sep=sep, encoding="utf-8-sig")

def clean_report_id_series(s):
    """Keep only digits; return pandas nullable Int64."""
    return (
        s.astype(str)
         .str.strip()
         .str.extract(r"(\d+)", expand=False)
         .astype("Int64")
    )

def load_structured_jsons(folder_glob):
    rows = []
    paths = glob.glob(folder_glob)
    if not paths:
        print(f"[WARN] No structured JSONs matched: {folder_glob}")
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                s = json.load(f)
            # normalize report_id
            rid = s.get("report_id")
            if rid is None or str(rid).strip() == "":
                # fallback: last number in filename
                m = re.findall(r"(\d+)", Path(path).stem)
                rid = int(m[-1]) if m else None
            else:
                rid = int(re.findall(r"(\d+)", str(rid))[0])
            row = {
                "report_id": rid,
                "arrival": s.get("arrival"),
                "departure": s.get("departure"),
                "total_time_spent": s.get("total_time_spent"),
                "attributes_filled": bool(s.get("attributes_filled", False)),
                "comments": (s.get("excerpts", {}) or {}).get("comments", ""),
                "attributes_excerpt": (s.get("excerpts", {}) or {}).get("attributes", ""),
            }
            # boolean flags
            for k, v in (s.get("flags", {}) or {}).items():
                row[f"flag__{k}"] = bool(v)
            rows.append(row)
        except Exception as e:
            print(f"[SKIP] {path}: {e}")
    df = pd.DataFrame(rows)
    if "report_id" in df.columns:
        df["report_id"] = pd.to_numeric(df["report_id"], errors="coerce").astype("Int64")
    return df

def attach_labels(df, labels_csv, taxonomy):
    lab = sniff_read_csv(labels_csv)

    # validate columns
    if "report_id" not in lab.columns:
        raise ValueError(f"labels CSV missing 'report_id'. Found: {list(lab.columns)}")
    label_col = None
    for cand in ["label_codes", "labels", "codes"]:
        if cand in lab.columns:
            label_col = cand
            break
    if label_col is None:
        raise ValueError(f"labels CSV missing label column. Expected one of: label_codes, labels, codes. Found: {list(lab.columns)}")

    # clean/align types
    lab["report_id"] = clean_report_id_series(lab["report_id"])
    df["report_id"] = pd.to_numeric(df["report_id"], errors="coerce").astype("Int64")

    # parse semicolon separated codes -> list
    lab["label_list"] = lab[label_col].fillna("").apply(
        lambda s: [c.strip() for c in str(s).split(";") if c and str(c).strip()]
    )

    # make sure we use the taxonomy order
    classes = taxonomy["code"].tolist()
    mlb = MultiLabelBinarizer(classes=classes)
    _ = mlb.fit([[]])  # lock classes; we set 0/1 manually below

    # merge labels onto features
    merged = df.merge(lab[["report_id", "label_list"]], on="report_id", how="left")

    # initialize label columns to 0
    for c in classes:
        merged[c] = 0

    # set 1s per labeled report
    lab_map = dict(zip(lab["report_id"], lab["label_list"]))
    for rid, codes in lab_map.items():
        if pd.isna(rid):
            continue
        codes = set(codes)
        for c in codes:
            if c in merged.columns:
                merged.loc[merged["report_id"] == rid, c] = 1

    # summary/debug
    set_struct = set(merged["report_id"].dropna().astype(int).tolist())
    set_labels = set(lab["report_id"].dropna().astype(int).tolist())
    inter = set_struct & set_labels
    only_labels = set_labels - set_struct
    only_struct = set_struct - set_labels
    print(f"[INFO] structured rows: {len(set_struct)} unique IDs")
    print(f"[INFO] labeled rows   : {len(set_labels)} unique IDs")
    print(f"[INFO] intersection   : {len(inter)} IDs")
    if only_labels:
        print(f"[WARN] Labels with no structured JSON: {sorted(list(only_labels))[:10]}{' ...' if len(only_labels)>10 else ''}")
    if only_struct:
        print(f"[WARN] Structured JSONs with no labels: {sorted(list(only_struct))[:10]}{' ...' if len(only_struct)>10 else ''}")

    # show positive counts
    pos = {c: int(merged[c].sum()) for c in classes}
    print(f"[INFO] Label positives in merged dataset: {pos}")

    return merged, classes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structured_glob", required=True, help='Glob to structured_*.json')
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--taxonomy_yaml", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    taxonomy = load_taxonomy(args.taxonomy_yaml)
    X = load_structured_jsons(args.structured_glob)
    if X.empty:
        raise SystemExit(f"[ERROR] No structured data found for: {args.structured_glob}")

    Xy, label_order = attach_labels(X, args.labels_csv, taxonomy)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    Xy.to_csv(args.out_csv, index=False)
    print(f"[OK] Wrote dataset with {len(Xy)} rows and {len(label_order)} labels to {args.out_csv}")

if __name__ == "__main__":
    main()
