"""
NSL-KDD binary intrusion detection (normal vs attack) - evaluation script.

Put KDDTrain+.txt and KDDTest+.txt next to this file, then run:
    python nsl_kdd_evaluation.py

Outputs go to ./results/ (metrics table, confusion matrices, PR curves,
feature importance).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score,
                             ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

SEEDS = [0, 1, 2, 3, 4]
MAIN_SEED = 42
OUT = "results"
os.makedirs(OUT, exist_ok=True)

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]
CAT = ["protocol_type", "service", "flag"]

# ----------------------------------------------------------------- data
train = pd.read_csv("KDDTrain+.txt", names=COLUMNS).drop(columns="difficulty")
test = pd.read_csv("KDDTest+.txt", names=COLUMNS).drop(columns="difficulty")

# 1 = attack, 0 = normal  (attack is the positive class)
y_train = (train["label"] != "normal").astype(int).values
y_test = (test["label"] != "normal").astype(int).values
X_train = train.drop(columns="label")
X_test = test.drop(columns="label")
NUM = [c for c in X_train.columns if c not in CAT]

# ------------------------------------------------- dataset sanity checks
train_attacks = set(train["label"]) - {"normal"}
test_attacks = set(test["label"]) - {"normal"}
unseen = test_attacks - train_attacks
is_unseen = test["label"].isin(unseen).values  # unseen attack rows in test
print(f"Train: {len(train)} rows | attack types: {len(train_attacks)}")
print(f"Test : {len(test)} rows | attack types: {len(test_attacks)}")
print(f"Attack types only in test: {len(unseen)} "
      f"({is_unseen.sum()} rows = {is_unseen.mean():.1%} of test, "
      f"{is_unseen.sum() / y_test.sum():.1%} of test attacks)")
for c in CAT:
    only_test = set(test[c]) - set(train[c])
    print(f"{c}: categories only in test = {sorted(only_test)}")


# --------------------------------------------------------------- models
def preprocessor(scale: bool):
    """One-hot encoder is FIT ON TRAIN ONLY (inside the pipeline);
    unknown test categories are ignored instead of crashing."""
    num_step = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("num", num_step, NUM),
    ])


def build(name, seed):
    if name == "Majority baseline":
        clf = DummyClassifier(strategy="most_frequent")
        scale = False
    elif name == "Logistic regression":
        clf = LogisticRegression(max_iter=2000, random_state=seed)
        scale = True  # scale matters for linear models
    elif name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=100, random_state=seed,
                                     n_jobs=-1)
        scale = False  # trees are scale-invariant
    elif name == "XGBoost":
        clf = XGBClassifier(random_state=seed, eval_metric="logloss",
                            n_jobs=-1)
        scale = False
    return Pipeline([("prep", preprocessor(scale)), ("clf", clf)])


def metrics(model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, pred),
        "attack_precision": precision_score(y, pred, zero_division=0),
        "attack_recall": recall_score(y, pred),
        "attack_f1": f1_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "pr_auc": average_precision_score(y, proba),
    }


NAMES = ["Majority baseline", "Logistic regression", "Random Forest", "XGBoost"]
rows, fitted = [], {}
for name in NAMES:
    seeds = SEEDS if name in ("Random Forest", "XGBoost") else [MAIN_SEED]
    per_seed = []
    for s in seeds:
        m = build(name, s).fit(X_train, y_train)
        per_seed.append(metrics(m, X_test, y_test))
        if s == seeds[0]:
            fitted[name] = m
    df = pd.DataFrame(per_seed)
    row = {"model": name, "n_seeds": len(seeds)}
    for c in df.columns:
        row[c] = f"{df[c].mean():.3f}" + (f" ± {df[c].std():.3f}"
                                           if len(seeds) > 1 else "")
    rows.append(row)
    print(f"done: {name}")

results = pd.DataFrame(rows)
results.to_csv(f"{OUT}/metrics.csv", index=False)
print("\n=== KDDTest+ results (mean ± std over seeds where applicable) ===")
print(results.to_string(index=False))

# --------------------------- recall on SEEN vs UNSEEN attack types
print("\n=== Attack recall: attack types seen in training vs unseen ===")
atk = y_test == 1
for name in ("Random Forest", "XGBoost"):
    pred = fitted[name].predict(X_test)
    seen_r = recall_score(y_test[atk & ~is_unseen], pred[atk & ~is_unseen])
    unseen_r = recall_score(y_test[atk & is_unseen], pred[atk & is_unseen])
    print(f"{name:15s} seen-attack recall = {seen_r:.3f} | "
          f"unseen-attack recall = {unseen_r:.3f}")

# ------------------------------------------------------------- figures
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, name in zip(axes, ("Random Forest", "XGBoost")):
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, fitted[name].predict(X_test)),
        display_labels=["normal", "attack"]).plot(ax=ax, colorbar=False)
    ax.set_title(name)
plt.tight_layout()
plt.savefig(f"{OUT}/confusion_matrices.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 4.5))
for name in ("Logistic regression", "Random Forest", "XGBoost"):
    p, r, _ = precision_recall_curve(
        y_test, fitted[name].predict_proba(X_test)[:, 1])
    ap = average_precision_score(y_test, fitted[name].predict_proba(X_test)[:, 1])
    plt.plot(r, p, label=f"{name} (AP={ap:.2f})")
plt.xlabel("Attack recall")
plt.ylabel("Attack precision")
plt.title("Precision-recall on KDDTest+")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/pr_curves.png", dpi=150)
plt.close()

rf = fitted["Random Forest"]
names = rf.named_steps["prep"].get_feature_names_out()
imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=names)
imp = imp.sort_values().tail(15)
imp.index = [i.split("__", 1)[-1] for i in imp.index]
plt.figure(figsize=(6, 5))
imp.plot.barh()
plt.title("Random Forest - top 15 features")
plt.tight_layout()
plt.savefig(f"{OUT}/feature_importance_rf.png", dpi=150)
plt.close()
print(f"\nSaved metrics and figures to ./{OUT}/")