"""
Train a supervised fault classifier on the pseudo-labelled meter-days.

Input : Data/Processed_Data/meter_days_labeled.pkl   (from Pseudo_Labeling.py)
Output: Models/fault_classifier.pkl
        Outputs/supervised_cv_report.txt
        Outputs/supervised_confusion_matrix.png
        Outputs/supervised_feature_importance.csv
        Outputs/supervised_predictions_on_ingen_data_days.csv

Evaluation is done with GroupKFold over MeterID: a meter is never in both the
training and the validation fold, so the score says something about NEW
installations (which is what the radiators are), not about meters the model
has already seen.

The model is compared with the simple temperature rule "daily ΔT < 10 °C"
(the trigger used by Leiria et al. 2023). If the model cannot beat that rule,
that is a result worth reporting.

Remember what the cross-validation score means: how well the model reproduces
the pseudo-labels from temperatures alone. It is NOT the accuracy on real
faults. That is only measured on the radiator data (Validate_On_Radiator_Data.py).
"""
import os
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Pseudo_Labeling"))
from heat_features import FEATURES  # noqa: E402

INPUT_FILE = "Data/Processed_Data/meter_days_labeled.pkl"
MODEL_FILE = "Models/fault_classifier.pkl"
OUT_DIR = "Outputs"

# The group's six rules (Pseudo_Labeling.py). Days labelled "ingen_data" are
# never used. Training uses heating days only (outdoor < 12 °C): almost all
# rules need it, and the radiator test data is collected in the heating season.
TRAIN_CLASSES = ["normal", "fejlinstallation", "radiator_slukket", "blokeret_radiator",
                 "radiator_helt_aaben", "aabent_vindue"]
HEATING_MAX_TOUT = 12.0
MAX_PER_CLASS = 300_000     # cap on rows per class (speed); minority classes are kept whole
N_FOLDS = 5
RANDOM_STATE = 42
RULE_DT_THRESHOLD = 10.0    # baseline: day is a fault if mean hourly ΔT < this


def make_model():
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.1,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        class_weight="balanced",
        early_stopping=True,
        validation_fraction=0.1,
        random_state=RANDOM_STATE,
    )


def load_training_data():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"{INPUT_FILE} not found. Run Pseudo_Labeling.py first.")
    df = pd.read_pickle(INPUT_FILE)
    lab = df[df["label"].isin(TRAIN_CLASSES) & (df["T_outdoor"] < HEATING_MAX_TOUT)].copy()
    lab = lab.dropna(subset=["rel_dT"])   # need at least the core feature

    parts = []
    for c, g in lab.groupby("label"):
        parts.append(g.sample(n=MAX_PER_CLASS, random_state=RANDOM_STATE) if len(g) > MAX_PER_CLASS else g)
    lab = pd.concat(parts).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return df, lab


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    df, lab = load_training_data()
    X, y, groups = lab[FEATURES], lab["label"].to_numpy(), lab["MeterID"].to_numpy()

    print(f"[1/4] Training rows: {len(lab):,} from {lab.MeterID.nunique()} meters")
    print(lab["label"].value_counts().to_string())

    print(f"\n[2/4] {N_FOLDS}-fold cross-validation grouped by MeterID...")
    oof_pred = np.empty(len(lab), dtype=object)
    oof_proba = np.zeros((len(lab), len(TRAIN_CLASSES)))
    for k, (tr, va) in enumerate(GroupKFold(n_splits=N_FOLDS).split(X, y, groups)):
        m = make_model().fit(X.iloc[tr], y[tr])
        oof_pred[va] = m.predict(X.iloc[va])
        proba = m.predict_proba(X.iloc[va])
        oof_proba[va] = proba[:, [list(m.classes_).index(c) for c in TRAIN_CLASSES]]
        print(f"   fold {k + 1}: macro-F1 = {f1_score(y[va], oof_pred[va], average='macro'):.3f}")

    report = []
    report.append("SUPERVISED FAULT CLASSIFIER - grouped cross-validation (unseen meters)\n")
    report.append("Scores measure agreement with the PSEUDO-LABELS, not with real faults.\n")
    report.append(f"Rows: {len(lab):,}   Meters: {lab.MeterID.nunique()}   Features: {len(FEATURES)}\n")
    report.append(classification_report(y, oof_pred, labels=TRAIN_CLASSES, digits=3))
    report.append(f"Balanced accuracy: {balanced_accuracy_score(y, oof_pred):.3f}")
    report.append(f"Macro F1:          {f1_score(y, oof_pred, average='macro'):.3f}\n")

    # Binary fault vs normal, model vs simple temperature rule
    is_fault = y != "normal"
    p_fault = 1 - oof_proba[:, TRAIN_CLASSES.index("normal")]
    model_fault = oof_pred != "normal"
    rule_fault = (lab["dT_mean"] < RULE_DT_THRESHOLD).to_numpy()
    report.append("FAULT vs NORMAL (any fault class counts as fault)")
    report.append(f"  PR-AUC model:                  {average_precision_score(is_fault, p_fault):.3f}"
                  f"   (chance level = {is_fault.mean():.3f})")
    for name, pred in [("model", model_fault), (f"rule dT_mean < {RULE_DT_THRESHOLD:g} C", rule_fault)]:
        report.append(f"  {name:30s} precision {precision_score(is_fault, pred, zero_division=0):.3f}"
                      f"  recall {recall_score(is_fault, pred):.3f}  F1 {f1_score(is_fault, pred):.3f}")
    report.append("\nPer-class recall of the rule (which faults a ΔT rule can and cannot see):")
    for c in TRAIN_CLASSES[1:]:
        m = y == c
        report.append(f"  {c:18s} rule {rule_fault[m].mean():.3f}   model {model_fault[m].mean():.3f}")
    text = "\n".join(report)
    print("\n" + text)
    with open(os.path.join(OUT_DIR, "supervised_cv_report.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")

    cm = confusion_matrix(y, oof_pred, labels=TRAIN_CLASSES, normalize="true")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(TRAIN_CLASSES)), TRAIN_CLASSES, rotation=30, ha="right")
    ax.set_yticks(range(len(TRAIN_CLASSES)), TRAIN_CLASSES)
    for i in range(len(TRAIN_CLASSES)):
        for j in range(len(TRAIN_CLASSES)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Pseudo-label")
    ax.set_title("Cross-validated confusion matrix (row-normalised)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "supervised_confusion_matrix.png"), dpi=200)
    plt.close(fig)

    print("\n[3/4] Training final model on all labelled rows...")
    model = make_model().fit(X, y)

    sub = lab.sample(n=min(50_000, len(lab)), random_state=RANDOM_STATE)
    imp = permutation_importance(model, sub[FEATURES], sub["label"], scoring="f1_macro",
                                 n_repeats=3, random_state=RANDOM_STATE, n_jobs=1)
    imp_df = (pd.DataFrame({"feature": FEATURES, "importance": imp.importances_mean, "std": imp.importances_std})
                .sort_values("importance", ascending=False))
    imp_df.to_csv(os.path.join(OUT_DIR, "supervised_feature_importance.csv"), index=False)
    print(imp_df.to_string(index=False))

    print("\n[4/4] Saving model and predicting days the rules could not be applied to...")
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "classes": list(model.classes_),
                 "train_classes": TRAIN_CLASSES}, MODEL_FILE)

    unc = df[(df["label"] == "ingen_data") & (df["T_outdoor"] < HEATING_MAX_TOUT)].dropna(subset=["rel_dT"])
    if len(unc):
        unc_pred = pd.Series(model.predict(unc[FEATURES])).value_counts(normalize=True).rename("share")
        unc_pred.to_csv(os.path.join(OUT_DIR, "supervised_predictions_on_ingen_data_days.csv"))
        print("Predicted classes for heating days the rules could not be applied to:\n", unc_pred.to_string())

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Model saved to {MODEL_FILE}")


if __name__ == "__main__":
    main()
