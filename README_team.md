### SRI - Sloppy Report Identifier
## - By Samanyu Okade, with MKB Data Studios (Dario Turelli), for and thanks to KVT (Milan Scheenloop, Yannick van der Borght, and Wouter Meijer)

# Sloppy Report Auto‑Flagging: Rules + ML Pipeline

Here I explains how to train the complaint classifier over sloppy service reports and how to detect issues on new reports. It combines:
- Schema/rule checks (deterministic, explainable)
- Multi‑label ML predictions (learns patterns)

All commands below are for Windows CMD.

also import pyPDF2reader

-------------------------------------------------------------------------------------------------------

## 1) Folder organisation for neat segregation and data collection

Codes can be found in:
```
C:\Users\sokade\Downloads\sloppy_ml\
  aggregate_dataset.py
  train_multilabel.py        ← uses only built‑ins (safe to pickle)
  predict_multilabel.py      ← uses only built‑ins (safe to pickle)
  merge_hybrid.py
  labels_taxonomy.yaml       ← list of complaint codes → text + action
  labels_template.csv        ← team fills label_codes per report_id
```

Data (produced by the PDF extractor `sloppy_reports_reader.ipynb`) can be found in:
```
C:\Users\sokade\Downloads\sloppy_reports\
  findings\        (findings_<ID>.csv)          ← optional (rule outputs)
  structured\      (structured_<ID>.json)       ← ML input (required)
  quotes_stub\     (quote_stub_<ID>.csv)        ← optional per‑line checks
```
- all pdfs in the folder are converted to the associated findings, structured and quotes_stub by the extractor jupyter notebook.

> Minimum required for ML: the pdf extractor should output at least `structured_<ID>.json` files for the report where ID = report ID.

## Optional phases:
### A) Rule pass on quote stubs

For consistent, explainable checks on materials/spec/qty/hours:

Prepare schema.json (from your edited schema.yaml):

```
py -c "import yaml, json; s=yaml.safe_load(open(r'C:\path\to\schema.yaml','r',encoding='utf-8')); json.dump(s, open(r'C:\path\to\schema.json','w',encoding='utf-8'))"
```

Score every quote stub CSV with the schema rules (batch loop):

```
cmd /v:on
mkdir "C:\Users\sokade\Downloads\sloppy_reports\findings_rules"
for %F in ("C:\Users\sokade\Downloads\sloppy_reports\quotes_stub\quote_stub_*.csv") do (
  set "n=%~nF"
  set "id=!n:quote_stub_=!"
  py C:\path\to\run_score.py ^
    --input_csv "%F" ^
    --schema_json C:\path\to\schema.json ^
    --form_type quote_stub ^
    --out_findings_csv "C:\Users\sokade\Downloads\sloppy_reports\findings_rules\findings_!id!.csv"
)
```

What this does under the hood:
score.py reads schema.json and runs the required/length/numeric/url/cross-checks defined there, using rules.py and features.py.

These are the results to be combined if needed in the Optional point 8 all the way below. Still, not compulsory, just adds more information to the normal predictions.

### B) Train the ML model (for auto-updating complaints in `labels_template.csv` but can also be done manually and ignored here)

For auto-labeling that mirrors the agent’s complaints:

Fill labels: open labels_template.csv and fill label_codes for the listed report_ids.

Build dataset:
```
py C:\path\to\aggregate_dataset.py ^
  --structured_glob "C:/Users/sokade/Downloads/sloppy_reports/structured/structured_*.json" ^
  --labels_csv C:/path/to/labels_template.csv ^
  --taxonomy_yaml C:/path/to/labels_taxonomy.yaml ^
  --out_csv C:/path/to/dataset.csv
```

Train:
```
py C:\path\to\train_multilabel.py ^
  --dataset_csv C:/path/to/dataset.csv ^
  --out_dir C:/path/to/model_out ^
  --test_size 0.2 ^
  --min_positives 1
```

------------------------------------------------------------------------------------------------

## 2) For a One‑Time Setup, perform in cmd, the following command (already done on this system):

```
py -m pip install pandas scikit-learn joblib pyyaml
```

-------------------------------------------------------------------------------------------------

## 3) Data Formats

### 3.1 `structured_<ID>.json` (input to ML)
Each JSON should contain (at minimum), example fillers in a form below:
```json
{
  "report_id": 4083505,
  "arrival": "08:10",
  "departure": "11:45",
  "total_time_spent": "03:35",
  "attributes_filled": true,
  "flags": {
    "fuel_polisher_pump_leak": false,
    "fuel_level_indicator_issue": true,
    "repair_advice_present": true,
    "run_log_incomplete": true
  },
  "excerpts": {
    "comments": "Short text snippet from Comments/Notes field ...",
    "attributes": "Short snippet from Attributes area ..."
  }
}
```
- `flags` are booleans the extractor infers from text cues.
- `excerpts` are short strings; the model uses them (TF‑IDF).

### 3.2 Labels: overall taxonomy to choose from can be found in `labels_taxonomy.yaml`
This file is for remembering and keeping track of what taxonomy associates to which abbreviation. Will help as the bank of problems and taxonomy increases.
Defines allowed label codes + texts + actions. Example (snippet):
```yaml
labels:
  - code: HOURS_MISSING
    issue: "Working hours missing/zero"
    action: "Enter arrival, departure, and total working time."
  - code: ATTRIBUTES_PARTIAL
    issue: "Attributes not filled / partially filled"
    action: "Fill power/battery/capacity/spec fields."
  - code: MATERIALS_NOT_BOOKED
    issue: "Materials used not booked to order"
    action: "Book materials to a new order to enable invoicing."
  # ... add more as needed
```

### 3.3 Labeling: `labels_template.csv`
This is the tiny manual file the team fills. Columns:
- `report_id` (e.g., 4083507)
- `label_codes` (semicolon‑separated list of codes from the taxonomy)
Here, as new reports and new issues pop up that need to be trained against, they can be filled. It is used later by the predictor to classify properly. 

Example:
```csv
report_id,label_codes
4083505,HOURS_MISSING;RUN_LOG_INCOMPLETE;FUEL_LEVEL_DECISION_MISSING;MATERIALS_DETAILS_MISSING
4083506,ATTRIBUTES_PARTIAL;MATERIALS_NOT_BOOKED
4083517,VIDEO_MISSING;TECH_SUPPORT_NOT_CONSULTED
```

Cheat‑sheet mapping from agent phrases → codes:
- “Attributes: PARTIALLY filled” → `ATTRIBUTES_PARTIAL`
- “Materials used not booked to new order” → `MATERIALS_NOT_BOOKED`
- “See video on SharePoint, no video” → `VIDEO_MISSING`
- “Should have been discussed with technical support” → `TECH_SUPPORT_NOT_CONSULTED`
- “Run log not completed” → `RUN_LOG_INCOMPLETE`
- “Fuel level indicator decision missing” → `FUEL_LEVEL_DECISION_MISSING`
- “Materials/spec/quantity/hours missing in quote lines” → `MATERIALS_DETAILS_MISSING`
- “Working hours missing/zero” → `HOURS_MISSING`

> Even if some labels have 0 positives initially; the trainer will just skip them until data exists, so both rules, and sloppy data reports to be trained with can continue to be updated overtime. Naturally, more true test reports, better the performance of the model over time.

--------------------------------------------------------------------------------------------------------------------

## 4) Step 0 — Preflight (CMD)

```
:: Check Python
py --version

:: Verify folders exist
dir "C:\Users\sokade\Downloads\sloppy_ml"
dir "C:\Users\sokade\Downloads\sloppy_reports\structured"

:: Count your structured JSONs (this scans across all thew IDs under structured_<ID>.json)
dir /b "C:\Users\sokade\Downloads\sloppy_reports\structured\structured_*.json" | find /c /v ""

:: Verify labels + taxonomy
dir "C:\Users\sokade\Downloads\sloppy_ml\labels_template.csv"
dir "C:\Users\sokade\Downloads\sloppy_ml\labels_taxonomy.yaml"

:: Make output folders if needed
mkdir "C:\Users\sokade\Downloads\sloppy_ml\model_out"
mkdir "C:\Users\sokade\Downloads\sloppy_ml\predicted"
```

-------------------------------------------------------------------------------------------------------------

## 5) Build Training Dataset (Aggregate)

Combine all `structured_*.json` with your labeled `labels_template.csv`, run the following command in cmd:

```
python "C:\Users\sokade\Downloads\sloppy_ml\aggregate_dataset.py" ^
  --structured_glob "C:/Users/sokade/Downloads/sloppy_reports/structured/structured_*.json" ^
  --labels_csv "C:/Users/sokade/Downloads/sloppy_ml/labels_template.csv" ^
  --taxonomy_yaml "C:/Users/sokade/Downloads/sloppy_ml/labels_taxonomy.yaml" ^
  --out_csv "C:/Users/sokade/Downloads/sloppy_ml/dataset.csv"
```

This creates `dataset.csv` for training.

------------------------------------------------------------------------------------------------------------------------------------------

## 6) Train the Multi‑Label Model, in cmd run:

```
python "C:\Users\sokade\Downloads\sloppy_ml\train_multilabel.py" ^
  --dataset_csv "C:/Users/sokade/Downloads/sloppy_ml/dataset.csv" ^
  --out_dir "C:/Users/sokade/Downloads/sloppy_ml/model_out" ^
  --test_size 0.2 ^
  --min_positives 1
```
- Here, the test size can be set as needed. 0.1 or 0.2 is recommended as the number of training reports are already quite low.
- Prints positives per label and Micro/Macro‑F1.
- Skips labels with fewer than `--min_positives` positives.
- Saves the model to:
  `C:\Users\sokade\Downloads\sloppy_ml\model_out\multilabel_model.joblib`

> For better generalization, aim for ≥3–5 positives per label and set `--min_positives 3` later.

-----------------------------------------------------------------------------------------------------------------------------------------

## 7) Predict on Any Reports (Including New Test Forms)

Ensure your extractor has saved `structured_<ID>.json` for the test report, then run the following command in cmd:

```
python "C:\Users\sokade\Downloads\sloppy_ml\predict_multilabel.py" ^
  --structured_glob "C:/Users/sokade/Downloads/sloppy_reports/structured/structured_*.json" ^
  --model_path "C:/Users/sokade/Downloads/sloppy_ml/model_out/multilabel_model.joblib" ^
  --taxonomy_yaml "C:/Users/sokade/Downloads/sloppy_ml/labels_taxonomy.yaml" ^
  --threshold 0.5 ^
  --out_dir "C:/Users/sokade/Downloads/sloppy_ml/predicted"
```

Outputs (per report):
```
C:\Users\sokade\Downloads\sloppy_ml\predicted\predicted_findings_<ID>.csv
```
Columns: `label_code, confidence, issue, action_request`.
- confidence => how sure the model is of those particular fields being errored in the report. With more training and more data for training, the confidences increase.

Threshold tuning:  
- Too many false positives → raise to `0.6–0.7`.  
- Missing issues → lower to `0.4–0.45`.

To predict only one report (e.g., `structured_4099999.json`, can be used for every new singular report once the team has trained the ML model against enough data), in cmd run:
```
python "C:\Users\sokade\Downloads\sloppy_ml\predict_multilabel.py" ^
  --structured_glob "C:/Users/sokade/Downloads/sloppy_reports/structured/structured_4099999.json" ^
  --model_path "C:/Users/sokade/Downloads/sloppy_ml/model_out/multilabel_model.joblib" ^
  --taxonomy_yaml "C:/Users/sokade/Downloads/sloppy_ml/labels_taxonomy.yaml" ^
  --threshold 0.5 ^
  --out_dir "C:/Users/sokade/Downloads/sloppy_ml/predicted"
```

--------------------------------------------------------------------------------------------------------------------------------------------------------------

## 8) (Optional) Merge ML Predictions with Rule Findings

If you also have rule findings (`findings\findings_<ID>.csv`) from your extractor, merge them with ML predictions into a single final CSV per report.

Enable delayed expansion for the for‑loop, in cmd:
```
cmd /v:on
```

Then run:
```
0213789+/*-
for %P in ("C:\Users\sokade\Downloads\sloppy_ml\predicted\predicted_findings_*.csv") do (
  set "n=%~nP"
  set "id=!n:predicted_findings_=!"
  python "C:\Users\sokade\Downloads\sloppy_ml\merge_hybrid.py" ^
    --rule_findings_csv "C:\Users\sokade\Downloads\sloppy_reports\findings\findings_!id!.csv" ^
    --ml_findings_csv   "C:\Users\sokade\Downloads\sloppy_ml\predicted\predicted_findings_!id!.csv" ^
    --out_csv           "C:\Users\sokade\Downloads\sloppy_ml\predicted\final_findings_!id!.csv"
)
```

Deliverables:
```
C:\Users\sokade\Downloads\sloppy_ml\predicted\final_findings_<ID>.csv
```

---------------------------------------------------------------------------------------------------------------------------------

## 9) Maintenance & Retraining

- Add labels for new complaint types in `labels_taxonomy.yaml` and re‑label a few examples.
- Update `labels_template.csv` as new feedback arrives.
- Re‑run aggregate → train → predict periodically (weekly/bi‑weekly).
- Keep `--threshold` under review; adjust according to the quality assurance agent feedback.
- Eventually, an ML model (a simple and light pattern finder or pattern recognition tool) can be deployed on this csv to append more possible issues in future reports. 
- Training with at least 50 reports (from 3.3) will enhance results, and show better performance with the test reports the team might have to use eventually. With the pattern finder in the csv, and ample training data, my current multilabel learning tool and the pattern finder will feed into each other to automate the entire process.
- 
----------------------------------------------------------------------------------------------------------------------------------------

## 10) Troubleshooting

- Some labels always 0 during training  
  Not enough positives; trainer will skip them. Add some labeled examples and retrain.

- Pickle error (can’t load function)  
  Use the provided `train_multilabel.py` / `predict_multilabel.py` that avoid custom functions inside pipelines (already included here).

- No predictions  
  Threshold too high or no trained labels. Lower `--threshold` or add labels & retrain.

- CMD line wrapping  
  End each continued line with `^`. Avoid trailing spaces after `^`. Always quote paths.

--------------------------------------------------------------------------------------------------------------------------------------------

## 11) FAQ

Q: How many samples would you need?  
Starting with ~50 labeled reports would greatly impact and show improvements in performance. For each label needed, it would be nice to aim for at least 3–5 positives for the best results.

Q: What does “multi‑label” mean?  
A single report can have multiple issues at once; the model predicts all applicable labels, not just one.

---------------------------------------------------------------------------------------------------------------------------------------------

Contact / Handover Notes:  
- If adding new fields, keep keys stable and update labeling/feature logic only if needed.

Happy shipping! 🚀
