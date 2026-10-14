"""Reproducible ML pipeline for vehicle failure-risk classification.

Run from the project root:
    python train_model.py

Workflow (all model selection uses the TRAINING partition only):
  1. Stratified 80/20 train/test split.
  2. Untuned baselines for Logistic Regression, Random Forest, XGBoost
     (stratified 5-fold CV).
  3. Hyperparameter tuning of ALL THREE models with the same CV folds and the
     same metric (average precision, suited to the ~4.5% positive class).
  4. Final-model choice from the tuned CV results (rule explained below).
  5. Decision threshold chosen from out-of-fold training predictions.
  6. One-time evaluation of the final model on the untouched test split.

The held-out test set is used only for final reporting.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from backend.reference import build_reference, risk_band, risk_index

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "sensor_data.csv"
MODEL_PATH = ROOT / "backend" / "models" / "vehicle_failure_model.joblib"
REPORT_PATH = ROOT / "reports" / "model_metrics.json"
COMPARISON_CSV = ROOT / "reports" / "baseline_cv_comparison.csv"
SEED = 42
TARGET = "failure_risk"
SELECTION_METRIC = "average_precision"
SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
}
XGB_N_ITER = 40  # random configurations tried for XGBoost


def as_json_number(value):
    """Convert NumPy numbers to ordinary JSON-compatible values."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def imputer():
    return ("imputer", SimpleImputer(strategy="median"))


def build_models(scale_pos_weight: float) -> dict[str, dict]:
    """Return {name: {baseline, tuned_pipeline, search_space, search_type}}.

    Every pipeline keeps imputation/scaling inside the pipeline, so CV folds
    never see statistics computed from their own validation data.
    """
    lr_base = Pipeline(
        [
            imputer(),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=SEED
                ),
            ),
        ]
    )
    rf_base = Pipeline(
        [
            imputer(),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced_subsample",
                    min_samples_leaf=2,
                    random_state=SEED,
                    n_jobs=1,
                ),
            ),
        ]
    )
    xgb_base = Pipeline(
        [
            imputer(),
            (
                "model",
                XGBClassifier(
                    n_estimators=250,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    min_child_weight=3,
                    reg_lambda=2,
                    scale_pos_weight=scale_pos_weight,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=SEED,
                    n_jobs=1,
                ),
            ),
        ]
    )
    # Tuning starts from the same pipeline but with a search space on top.
    xgb_tune = Pipeline(
        [
            imputer(),
            (
                "model",
                XGBClassifier(
                    scale_pos_weight=scale_pos_weight,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=SEED,
                    n_jobs=1,
                ),
            ),
        ]
    )
    return {
        "Logistic Regression": {
            "baseline": lr_base,
            "tune": lr_base,
            "search": "grid",
            "space": {
                "model__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
            },
        },
        "Random Forest": {
            "baseline": rf_base,
            "tune": rf_base,
            "search": "grid",
            "space": {
                "model__n_estimators": [300],
                "model__max_depth": [4, 8, 12],
                "model__min_samples_leaf": [2, 5, 10],
                "model__max_features": ["sqrt", 0.5],
            },
        },
        "XGBoost": {
            "baseline": xgb_base,
            "tune": xgb_tune,
            "search": "random",
            "space": {
                "model__n_estimators": [100, 200, 300, 500],
                "model__max_depth": [2, 3, 4, 5],
                "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "model__min_child_weight": [1, 3, 5, 10],
                "model__subsample": [0.7, 0.8, 0.9, 1.0],
                "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                "model__reg_lambda": [1, 2, 5, 10],
            },
        },
    }


def cv_summary(scores: dict, prefix: str = "test_") -> dict:
    return {m: float(np.mean(scores[f"{prefix}{m}"])) for m in SCORING}


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Put sensor_data.csv in the data/ folder."
        )

    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise ValueError(f"Required target column {TARGET!r} is missing.")
    if df[TARGET].isna().any():
        raise ValueError(f"Target column {TARGET!r} contains missing values.")
    if set(df[TARGET].unique()) - {0, 1}:
        raise ValueError(
            f"Expected binary target values 0 and 1, got {sorted(df[TARGET].unique())}."
        )

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    feature_names = list(X.columns)

    # 1. Stratified split keeps the rare positive class in both partitions.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    scale_pos_weight = negatives / max(positives, 1)

    models = build_models(scale_pos_weight)

    # 2. Untuned baselines.
    baseline_results: dict[str, dict] = {}
    for name, spec in models.items():
        t0 = time.time()
        scores = cross_validate(
            spec["baseline"], X_train, y_train, cv=cv, scoring=SCORING, n_jobs=2
        )
        baseline_results[name] = cv_summary(scores)
        print(f"[baseline] {name}: AP={baseline_results[name]['average_precision']:.4f} "
              f"ROC-AUC={baseline_results[name]['roc_auc']:.4f} ({time.time()-t0:.0f}s)",
              flush=True)

    # 3. Tune ALL models: same folds, same selection metric.
    tuned: dict[str, dict] = {}
    for name, spec in models.items():
        t0 = time.time()
        common = dict(
            scoring=SCORING,
            refit=SELECTION_METRIC,
            cv=cv,
            n_jobs=2,
            return_train_score=False,
        )
        if spec["search"] == "grid":
            search = GridSearchCV(spec["tune"], param_grid=spec["space"], **common)
        else:
            search = RandomizedSearchCV(
                spec["tune"],
                param_distributions=spec["space"],
                n_iter=XGB_N_ITER,
                random_state=SEED,
                **common,
            )
        search.fit(X_train, y_train)
        i = search.best_index_
        res = search.cv_results_
        summary = {m: float(res[f"mean_test_{m}"][i]) for m in SCORING}
        ap_std = float(res["std_test_average_precision"][i])
        tuned[name] = {
            "search": search,
            "best_params": {k: as_json_number(v) for k, v in search.best_params_.items()},
            "cv": summary,
            "cv_ap_std": ap_std,
            "cv_ap_stderr": ap_std / np.sqrt(cv.get_n_splits()),
            "n_candidates": int(len(res["params"])),
        }
        print(f"[tuned]    {name}: AP={summary['average_precision']:.4f} "
              f"±{ap_std:.4f} ROC-AUC={summary['roc_auc']:.4f} "
              f"({tuned[name]['n_candidates']} configs, {time.time()-t0:.0f}s)", flush=True)

    # 4. Final-model choice.
    # Rule: compare tuned CV average precision. The API explains predictions with
    # SHAP TreeExplainer on XGBoost, so XGBoost is deployed. We record honestly
    # whether it is the top scorer, tied with the top scorer (within ONE standard
    # error), or clearly behind it.
    final_name = "XGBoost"
    best_name = max(tuned, key=lambda n: tuned[n]["cv"][SELECTION_METRIC])
    best_ap = tuned[best_name]["cv"][SELECTION_METRIC]
    best_se = tuned[best_name]["cv_ap_stderr"]
    xgb_ap = tuned["XGBoost"]["cv"][SELECTION_METRIC]
    if best_name == "XGBoost":
        selection_reason = f"XGBoost had the highest tuned CV average precision ({xgb_ap:.4f})."
    elif (best_ap - xgb_ap) <= best_se:
        selection_reason = (
            f"{best_name} had the highest CV average precision ({best_ap:.4f}), but XGBoost "
            f"({xgb_ap:.4f}) is within one standard error ({best_se:.4f}), so the models are "
            "statistically indistinguishable; XGBoost was kept because the API uses SHAP "
            "TreeExplainer and XGBoost captures non-linear interactions."
        )
    else:
        selection_reason = (
            f"WARNING: {best_name} scored higher on CV average precision ({best_ap:.4f} vs "
            f"{xgb_ap:.4f}, more than one standard error of {best_se:.4f}). XGBoost is still "
            "deployed only because the API's SHAP explainer needs a tree model; report this honestly."
        )
        print(selection_reason, flush=True)
    print(f"Final model: {final_name}. {selection_reason}", flush=True)

    best_pipeline = tuned[final_name]["search"].best_estimator_  # already refit on X_train

    # 5. Threshold from out-of-fold predictions on the training partition only.
    oof_scores = cross_val_predict(
        best_pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=2
    )[:, 1]
    threshold_candidates = []
    for threshold in np.linspace(0.05, 0.95, 181):
        preds = (oof_scores >= threshold).astype(int)
        threshold_candidates.append(
            {
                "threshold": float(threshold),
                "f1": float(f1_score(y_train, preds, zero_division=0)),
                "precision": float(precision_score(y_train, preds, zero_division=0)),
                "recall": float(recall_score(y_train, preds, zero_division=0)),
            }
        )
    selected_threshold = max(threshold_candidates, key=lambda row: row["f1"])
    threshold = selected_threshold["threshold"]

    # 6. Evaluate exactly once on the untouched test partition.
    test_scores = best_pipeline.predict_proba(X_test)[:, 1]
    test_metrics = {}
    for threshold_value in sorted({threshold, 0.5}):
        preds = (test_scores >= threshold_value).astype(int)
        test_metrics[f"{threshold_value:g}"] = {
            "threshold": float(threshold_value),
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1": float(f1_score(y_test, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, test_scores)),
            "average_precision": float(average_precision_score(y_test, test_scores)),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
            "confusion_matrix_layout": "[[TN, FP], [FN, TP]]",
        }

    # Reference data for the API (feature ranges, SHAP grading, risk bands, rule
    # evidence, demo vehicles). Built from training data; demo rows come from the
    # held-out split and are used only as examples, never for fitting or selection.
    reference = build_reference(
        best_pipeline, feature_names, X_train, y_train, oof_scores, threshold, X_test, y_test
    )
    # Band-level failure rates on the test split are REPORTING ONLY.
    test_band = risk_band(risk_index(test_scores, threshold))
    y_test_arr = np.asarray(y_test)
    risk_bands_report = {
        "definition": {
            "index": "0-100, monotone in model score, decision threshold = 50",
            "LOW": "index < 30",
            "MODERATE": "30 <= index < 50",
            "HIGH": "index >= 50 (same as predicted class 1)",
        },
        "out_of_fold_train": reference["band_evidence"],
        "test": {
            band: {
                "n_vehicles": int((test_band == band).sum()),
                "dataset_failure_rate": (
                    float(y_test_arr[test_band == band].mean()) if (test_band == band).any() else None
                ),
            }
            for band in ("LOW", "MODERATE", "HIGH")
        },
    }
    print("Risk bands (test):", risk_bands_report["test"], flush=True)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "reference": reference,
        "pipeline": best_pipeline,
        "threshold": threshold,
        "model_name": f"{final_name}Classifier".replace(" ", ""),
        "features": feature_names,
        "target": TARGET,
        "score_note": (
            "The returned score is an uncalibrated model score learned from the "
            "dataset's failure_risk labels; it is not a verified real-world failure probability."
        ),
        "threshold_selection": "Threshold maximizing out-of-fold F1 on the training partition.",
    }
    joblib.dump(bundle, MODEL_PATH)

    # Comparison table: untuned vs tuned, one row per model/stage.
    rows = []
    for name in models:
        rows.append({"model": name, "stage": "baseline", **baseline_results[name]})
        rows.append({"model": name, "stage": "tuned", **tuned[name]["cv"]})
    pd.DataFrame(rows).to_csv(COMPARISON_CSV, index=False, float_format="%.4f")

    report = {
        "dataset": {
            "file": "data/sensor_data.csv",
            "n_rows": int(len(df)),
            "n_features": int(len(feature_names)),
            "features": feature_names,
            "target": TARGET,
            "positive_count": int((y == 1).sum()),
            "negative_count": int((y == 0).sum()),
            "positive_rate": float(y.mean()),
            "missing_total": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
        },
        "split": {
            "random_state": SEED,
            "test_size": 0.2,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "stratified": True,
        },
        "cross_validation": {"type": "StratifiedKFold", "n_splits": 5, "shuffle": True, "random_state": SEED},
        "baseline_5_fold_cv": baseline_results,
        "tuned_5_fold_cv": {
            name: {
                "search_type": models[name]["search"],
                "configurations_tried": tuned[name]["n_candidates"],
                "search_space": {k: [as_json_number(v) for v in vals] for k, vals in models[name]["space"].items()},
                "best_params": tuned[name]["best_params"],
                "cv_metrics": tuned[name]["cv"],
                "cv_average_precision_std": tuned[name]["cv_ap_std"],
            }
            for name in models
        },
        "model_selection": {
            "selection_metric": SELECTION_METRIC,
            "selected_model": final_name,
            "reason": selection_reason,
            "best_params": tuned[final_name]["best_params"],
            "five_fold_cv_average_precision": tuned[final_name]["cv"][SELECTION_METRIC],
        },
        "threshold": {
            **selected_threshold,
            "selected_using": "5-fold out-of-fold predictions on training partition, maximum F1",
        },
        "test_metrics": test_metrics,
        "risk_bands": risk_bands_report,
        "maintenance_rule_evidence": {
            "basis": reference["rule_evidence_basis"],
            "rules": reference["rule_evidence"],
        },
        "caveats": [
            "failure_risk is the dataset's synthetic binary label (generated from a rule-based risk score); label semantics come from the dataset source.",
            "Model scores are not calibrated real-world failure probabilities.",
            "The dataset has no sequence/time-to-failure target; this model does not predict when a failure will occur.",
            "CV scores of the tuned models are the best of many candidates, so they are slightly optimistic; the test split is the unbiased estimate.",
        ],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=as_json_number))
    print(f"Saved model: {MODEL_PATH}", flush=True)
    print(f"Saved metrics: {REPORT_PATH}", flush=True)
    print(f"Selected threshold: {threshold:.3f}", flush=True)
    print(f"Held-out test metrics: {test_metrics[f'{threshold:g}']}", flush=True)


if __name__ == "__main__":
    main()
