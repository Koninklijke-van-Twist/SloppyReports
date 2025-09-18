@echo off
setlocal EnableDelayedExpansion

:: Check Python
py --version

:: Verify folders exist
dir "sloppy_ml"
dir "sloppy_reports\structured"

:: Count your structured JSONs (this scans across all thew IDs under structured_<ID>.json)
dir /b "sloppy_reports\structured\structured_*.json" | find /c /v ""

:: Verify labels + taxonomy
dir "sloppy_ml\labels_template.csv"
dir "sloppy_ml\labels_taxonomy.yaml"

:: Make output folders if needed
mkdir "sloppy_ml\model_out"
mkdir "sloppy_ml\predicted"

python "sloppy_ml\predict_multilabel.py" --structured_glob "sloppy_reports/structured/structured_*.json" --model_path "sloppy_ml/model_out/multilabel_model.joblib" --taxonomy_yaml "sloppy_ml/labels_taxonomy.yaml" --threshold 0.5 --out_dir "sloppy_ml/predicted"

for %%P in ("sloppy_ml\predicted\predicted_findings_*.csv") do (
  set "n=%%~nP"
  set "id=!n:predicted_findings_=!"
  python "sloppy_ml\merge_hybrid.py" --rule_findings_csv "sloppy_reports\findings\findings_!id!.csv" --ml_findings_csv   "sloppy_ml\predicted\predicted_findings_!id!.csv" --out_csv "sloppy_ml\predicted\final_findings_!id!.csv"
)

echo Success!
pause
endlocal