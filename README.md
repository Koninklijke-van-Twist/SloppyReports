# ML Add-On: Multi-Label Complaint Prediction

Fill `sloppy_detector/ml/labels_template.csv` (report_id, label_codes).
Codes are in `sloppy_detector/ml/labels_taxonomy.yaml`.

Aggregate:
```bash
python /mnt/data/sloppy_detector/ml/aggregate_dataset.py   --structured_glob "/path/to/structured_*.json"   --labels_csv /mnt/data/sloppy_detector/ml/labels_template.csv   --taxonomy_yaml /mnt/data/sloppy_detector/ml/labels_taxonomy.yaml   --out_csv /mnt/data/sloppy_detector/ml/dataset.csv
```

Train:
```bash
python /mnt/data/sloppy_detector/ml/train_multilabel.py   --dataset_csv /mnt/data/sloppy_detector/ml/dataset.csv   --out_dir /mnt/data/sloppy_detector/ml/model_out   --test_size 0.2
```

Predict:
```bash
python /mnt/data/sloppy_detector/ml/predict_multilabel.py   --structured_glob "/path/to/new/structured_*.json"   --model_path /mnt/data/sloppy_detector/ml/model_out/multilabel_model.joblib   --taxonomy_yaml /mnt/data/sloppy_detector/ml/labels_taxonomy.yaml   --threshold 0.5   --out_dir /mnt/data/sloppy_detector/ml/predicted
```

Merge rules + ML:
```bash
python /mnt/data/sloppy_detector/ml/merge_hybrid.py   --rule_findings_csv /path/to/findings_<ID>.csv   --ml_findings_csv /mnt/data/sloppy_detector/ml/predicted/predicted_findings_<ID>.csv   --out_csv /mnt/data/sloppy_detector/ml/predicted/final_findings_<ID>.csv
```