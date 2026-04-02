"""
Train static sign classifier (letters + numbers) from CSV of landmark vectors.
Compares Random Forest and XGBoost; saves best model and label list.
Supports explicit train/test CSVs or a single dataset with auto split.
"""
import argparse
import glob
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
import xgboost as xgb

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

DEFAULT_DATA_DIR = os.path.join(ROOT, "data", "processed", "static")
FALLBACK_DATA_DIR = os.path.join(ROOT, "data", "preprocessed", "static")
if not os.path.isdir(DEFAULT_DATA_DIR) and os.path.isdir(FALLBACK_DATA_DIR):
    DEFAULT_DATA_DIR = FALLBACK_DATA_DIR
DEFAULT_SPLIT_DIR = os.path.join(DEFAULT_DATA_DIR, "hand_vectors")
DEFAULT_SINGLE_CSV = os.path.join(DEFAULT_DATA_DIR, "static_data.csv")
DEFAULT_DATA_PATH = DEFAULT_SPLIT_DIR if os.path.isdir(DEFAULT_SPLIT_DIR) else DEFAULT_SINGLE_CSV

NO_DATA_MSG = (
    "No data at {path}. Provide an existing dataset: a CSV (or directory of CSVs) "
    "with columns: label, f0..f125 (126 features for both hands). "
    "You can also pass --train-data and --test-data explicitly."
)


def load_static_data(csv_path: str):
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError("CSV must have 'label' column")
    y = df["label"].astype(str)
    X = df.drop(columns=["label"]).astype(np.float32)
    return X.values, y.values


def load_static_data_from_path(data_path: str):
    """Load from a single CSV file or a directory of CSV files (one level)."""
    if os.path.isfile(data_path):
        return load_static_data(data_path)
    if os.path.isdir(data_path):
        pattern = os.path.join(data_path, "*.csv")
        csv_files = sorted(glob.glob(pattern))
        if not csv_files:
            return None, None
        X_list, y_list = [], []
        for f in csv_files:
            X, y = load_static_data(f)
            X_list.append(X)
            y_list.append(y)
        return np.vstack(X_list), np.concatenate(y_list)
    return None, None


def load_split_from_dir(data_dir: str):
    """Load train.csv and (optional) test.csv from a directory."""
    train_csv = os.path.join(data_dir, "train.csv")
    test_csv = os.path.join(data_dir, "test.csv")
    if not os.path.isfile(train_csv):
        return None, None, None, None
    X_train, y_train = load_static_data(train_csv)
    X_test, y_test = (load_static_data(test_csv) if os.path.isfile(test_csv) else (None, None))
    return X_train, y_train, X_test, y_test


def load_train_test_from_paths(train_path: str, test_path: str | None):
    """Load train data and optional test data from paths (file or directory)."""
    X_train, y_train = load_static_data_from_path(train_path)
    if X_train is None:
        return None, None, None, None
    if test_path:
        X_test, y_test = load_static_data_from_path(test_path)
    else:
        X_test, y_test = None, None
    return X_train, y_train, X_test, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA_PATH,
                        help="Path to CSV or directory of CSVs with label + f0..f125 (both hands)")
    parser.add_argument("--train-data", default=None,
                        help="Optional explicit train CSV or directory of CSVs")
    parser.add_argument("--test-data", default=None,
                        help="Optional explicit test CSV or directory of CSVs")
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "ml", "models"),
                        help="Directory to save model and labels")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true", help="Reduce console output")
    parser.add_argument("--skip-xgb", action="store_true", help="Skip XGBoost training")
    parser.add_argument("--rf-n-estimators", type=int, default=300)
    parser.add_argument("--rf-max-depth", type=int, default=25)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=1)
    parser.add_argument("--rf-n-jobs", type=int, default=-1)
    parser.add_argument("--rf-class-weight", default="balanced",
                        help="RandomForest class_weight (balanced or none)")
    parser.add_argument("--use-scaler", action=argparse.BooleanOptionalAction, default=True,
                        help="Apply StandardScaler before model training")
    parser.add_argument("--xgb-n-estimators", type=int, default=300)
    parser.add_argument("--xgb-max-depth", type=int, default=8)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.1)
    parser.add_argument("--xgb-subsample", type=float, default=0.9)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.9)
    parser.add_argument("--xgb-n-jobs", type=int, default=-1)
    args = parser.parse_args()

    X_train = y_train = X_test = y_test = None

    if args.train_data:
        X_train, y_train, X_test, y_test = load_train_test_from_paths(args.train_data, args.test_data)
        if X_train is None or len(X_train) == 0:
            print(NO_DATA_MSG.format(path=args.train_data))
            return
    else:
        if os.path.isdir(args.data):
            X_train, y_train, X_test, y_test = load_split_from_dir(args.data)
            if X_train is None:
                X_all, y_all = load_static_data_from_path(args.data)
            else:
                X_all, y_all = None, None
        else:
            X_all, y_all = load_static_data_from_path(args.data)

        if X_train is None:
            if X_all is None or len(X_all) == 0:
                print(NO_DATA_MSG.format(path=args.data))
                return

    # Encode string labels to integers (XGBoost requires integer classes)
    le = LabelEncoder()

    if X_train is None:
        y_all_enc = le.fit_transform(y_all)
        labels = [str(x) for x in le.classes_]
        if not args.quiet:
            print("Classes:", len(labels), labels[:10], "...")
            print("Samples:", len(X_all))
        X_train, X_test, y_train_enc, y_test_enc = train_test_split(
            X_all, y_all_enc, test_size=args.test_size, stratify=y_all_enc, random_state=args.seed
        )
        y_test_str = le.inverse_transform(y_test_enc)
    else:
        y_train_enc = le.fit_transform(y_train)
        labels = [str(x) for x in le.classes_]
        if not args.quiet:
            print("Classes:", len(labels), labels[:10], "...")
            print("Train samples:", len(X_train))

        if X_test is None:
            X_train, X_test, y_train_enc, y_test_enc = train_test_split(
                X_train, y_train_enc, test_size=args.test_size, stratify=y_train_enc, random_state=args.seed
            )
            y_test_str = le.inverse_transform(y_test_enc)
        else:
            try:
                y_test_enc = le.transform(y_test)
            except ValueError as exc:
                print("Test set has labels not seen in train:", exc)
                return
            y_test_str = y_test
        if not args.quiet:
            print("Test samples:", len(X_test))

    os.makedirs(args.out_dir, exist_ok=True)
    labels_path = os.path.join(args.out_dir, "static_labels.txt")
    with open(labels_path, "w") as f:
        f.write("\n".join(labels))
    if not args.quiet:
        print("Saved labels to", labels_path)

    # Random Forest
    class_weight = None
    if isinstance(args.rf_class_weight, str) and args.rf_class_weight.lower() != "none":
        class_weight = args.rf_class_weight
    rf_est = RandomForestClassifier(
        n_estimators=args.rf_n_estimators,
        max_depth=args.rf_max_depth,
        min_samples_leaf=args.rf_min_samples_leaf,
        class_weight=class_weight,
        n_jobs=args.rf_n_jobs,
        random_state=args.seed,
    )
    rf = make_pipeline(StandardScaler(), rf_est) if args.use_scaler else rf_est
    rf.fit(X_train, y_train_enc)
    rf_acc = accuracy_score(y_test_enc, rf.predict(X_test))
    if not args.quiet:
        print("Random Forest test accuracy:", rf_acc)

    # XGBoost (expects integer labels 0..n_classes-1)
    xgb_acc = -1.0
    xgb_model = None
    if not args.skip_xgb:
        xgb_est = xgb.XGBClassifier(
            n_estimators=args.xgb_n_estimators,
            max_depth=args.xgb_max_depth,
            learning_rate=args.xgb_learning_rate,
            subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample_bytree,
            n_jobs=args.xgb_n_jobs,
            random_state=args.seed,
            eval_metric="mlogloss",
            tree_method="hist",
            verbosity=0,
        )
        xgb_model = make_pipeline(StandardScaler(), xgb_est) if args.use_scaler else xgb_est
        xgb_model.fit(X_train, y_train_enc)
        xgb_acc = accuracy_score(y_test_enc, xgb_model.predict(X_test))
        if not args.quiet:
            print("XGBoost test accuracy:", xgb_acc)

    best = rf if rf_acc >= xgb_acc else xgb_model
    best_name = "rf" if rf_acc >= xgb_acc else "xgb"
    model_path = os.path.join(args.out_dir, "static_model.pkl")
    joblib.dump(best, model_path)
    if not args.quiet:
        print("Saved best model (", best_name, ") to", model_path)
        pred_str = le.inverse_transform(best.predict(X_test))
        print(classification_report(y_test_str, pred_str))


if __name__ == "__main__":
    main()
