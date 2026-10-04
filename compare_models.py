import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("network_traffic.csv")
X = df[["duration", "packet_size", "num_connections"]]
y = df["label"].values

# Encode labels for XGBoost (needs numbers not strings)
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # attack=0, normal=1

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Model 1: Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

# Model 2: XGBoost
xgb = XGBClassifier(random_state=42, eval_metric="logloss")
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)

# Compare results
print("=" * 50)
print("RANDOM FOREST")
print("=" * 50)
print(classification_report(y_test, rf_preds, target_names=le.classes_))
print("Attack Recall:", recall_score(y_test, rf_preds, pos_label=0))

print("=" * 50)
print("XGBOOST")
print("=" * 50)
print(classification_report(y_test, xgb_preds, target_names=le.classes_))
print("Attack Recall:", recall_score(y_test, xgb_preds, pos_label=0))

# Side by side summary
print("\n--- COMPARISON SUMMARY ---")
print(f"{'Metric':<20} {'Random Forest':<20} {'XGBoost':<20}")
print("-" * 60)

rf_recall = recall_score(y_test, rf_preds, pos_label=0)
xgb_recall = recall_score(y_test, xgb_preds, pos_label=0)

from sklearn.metrics import accuracy_score, f1_score
print(f"{'Accuracy':<20} {accuracy_score(y_test, rf_preds):<20.4f} {accuracy_score(y_test, xgb_preds):<20.4f}")
print(f"{'Attack Recall':<20} {rf_recall:<20.4f} {xgb_recall:<20.4f}")
print(f"{'Macro F1':<20} {f1_score(y_test, rf_preds, average='macro'):<20.4f} {f1_score(y_test, xgb_preds, average='macro'):<20.4f}")