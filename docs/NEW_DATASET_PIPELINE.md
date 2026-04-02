# New Dataset Pipeline (Static)

This file describes the end-to-end process to train a new static model when you add or change a dataset.

---

## 1) Place the Dataset

**Option A: Folder-per-class (recommended)**
```
data/raw/static/
  A/
  B/
  ...
  space/
  full-stop/
```

**Option B: Pre-split train/test**
```
data/processed/static/
  train/
    A/
    B/
    ...
  test/
    A/
    B/
    ...
```

**Label rules**
- Use consistent class names (e.g., `full-stop` not `fullstop`).
- Avoid duplicates that differ only by case.

---

## 2) Extract Hand Vectors (Images -> CSV)

**From folder-per-class:**
```bash
python -m ml.pipeline.data_collection.images_to_vectors \
  --input-dir data/raw/static \
  --output data/processed/static/static_data.csv
```

**From train/test split:**
```bash
python -m ml.pipeline.data_collection.images_to_vectors \
  --input-dir data/processed/static \
  --output-dir data/processed/static/hand_vectors
```

**Helpful options**
```bash
--min-det-conf 0.3 --min-pres-conf 0.3
--normalize-rotation --mirror-left
--retry-min-side 384
--log-every 500
```

---

## 3) Train the Model

**Single CSV:**
```bash
python -m ml.pipeline.training.train_static --data data/processed/static/static_data.csv
```

**Train/test CSV folder:**
```bash
python -m ml.pipeline.training.train_static --data data/processed/static/hand_vectors
```

**Faster training (skip XGBoost):**
```bash
python -m ml.pipeline.training.train_static --data data/processed/static/static_data.csv --skip-xgb
```

---

## 4) Evaluate on Images

```bash
python -m ml.pipeline.inference.eval_static_images --input-dir data/raw/static
```

---

## 5) Run Live Inference

All-in-one live view (prediction + sentence):
```bash
python -m ml.pipeline.inference.run_sentence_builder
```

If your dataset requires both hands:
```bash
python -m ml.pipeline.inference.run_sentence_builder --require-two-hands
```

---

## 6) Sanity Checks (Recommended)

**Check if vectors are valid:**
```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv('data/processed/static/static_data.csv')
print("rows:", len(df), "cols:", len(df.columns))
print("labels:", df['label'].nunique())
print("missing:", df.isna().sum().sum())
PY
```

**Check skipped images during extraction**  
When `images_to_vectors` runs, it prints:
```
Wrote X rows (skipped Y images).
```
If skipped is high, adjust detection settings.

---

## Output Files

- `ml/models/static_model.pkl`
- `ml/models/static_labels.txt`
- `data/processed/static/static_data.csv` or `data/processed/static/hand_vectors/*.csv`

