import pandas as pd, re

CSV = r"C:\Users\sokade\Downloads\sloppy_ml\dataset.csv"
df = pd.read_csv(CSV)

# infer label columns: UPPERCASE and 0/1
def is_bin(c):
    s = set(df[c].dropna().unique())
    return s.issubset({0,1})

label_cols = [c for c in df.columns if c.isupper() and is_bin(c)]
print("Label columns:", label_cols)

# count positives
pos = {c:int(df[c].sum()) for c in label_cols}
print("Positives per label:", pos)

# show a few rows that have any positive label
mask = df[label_cols].sum(axis=1) > 0
print("Rows with any label:", int(mask.sum()))
print(df.loc[mask].head(10))