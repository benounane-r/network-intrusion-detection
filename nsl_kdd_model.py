import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# NSL-KDD has 41 features + label + difficulty score
columns = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty"
]

# Load data
print("Loading data...")
train_df = pd.read_csv("KDDTrain+.txt", names=columns)
test_df  = pd.read_csv("KDDTest+.txt",  names=columns)

# Drop difficulty column
train_df = train_df.drop("difficulty", axis=1)
test_df  = test_df.drop("difficulty",  axis=1)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("\nLabel distribution (train):")
print(train_df["label"].value_counts())

# Convert to binary: normal vs attack
train_df["binary_label"] = train_df["label"].apply(
    lambda x: "normal" if x == "normal" else "attack"
)
test_df["binary_label"] = test_df["label"].apply(
    lambda x: "normal" if x == "normal" else "attack"
)

print("\nBinary label distribution (train):")
print(train_df["binary_label"].value_counts())

# Encode categorical features
# Three columns are text: protocol_type, service, flag
cat_cols = ["protocol_type", "service", "flag"]
train_df = pd.get_dummies(train_df, columns=cat_cols)
test_df  = pd.get_dummies(test_df,  columns=cat_cols)

# Align columns — test might have different categories
train_df, test_df = train_df.align(test_df, join="left", axis=1, fill_value=0)

print("\nFeatures after encoding:", train_df.shape[1])

# Prepare X and y
feature_cols = [c for c in train_df.columns
                if c not in ["label", "binary_label"]]

X_train = train_df[feature_cols]
y_train = train_df["binary_label"].values
X_test  = test_df[feature_cols]
y_test  = test_df["binary_label"].values

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))
print("\nAttack ratio (train):", 
      round(sum(y_train == "attack") / len(y_train) * 100, 1), "%")
print("Attack ratio (test):", 
      round(sum(y_test == "attack") / len(y_test) * 100, 1), "%")



print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

print("Training XGBoost...")
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)

xgb = XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1)
xgb.fit(X_train, y_train_enc)
xgb_preds = xgb.predict(X_test)
xgb_preds_labels = le.inverse_transform(xgb_preds)

print("\n" + "="*50)
print("RANDOM FOREST RESULTS")
print("="*50)
print(classification_report(y_test, rf_preds))
print("Attack Recall:", round(recall_score(y_test, rf_preds, pos_label="attack"), 4))

print("\n" + "="*50)
print("XGBOOST RESULTS")
print("="*50)
print(classification_report(y_test, xgb_preds_labels))
print("Attack Recall:", round(recall_score(y_test, xgb_preds_labels, pos_label="attack"), 4))

print("\n--- COMPARISON SUMMARY ---")
from sklearn.metrics import accuracy_score, f1_score
print(f"{'Metric':<20} {'Random Forest':<20} {'XGBoost'}")
print("-"*55)
print(f"{'Accuracy':<20} {accuracy_score(y_test, rf_preds):<20.4f} {accuracy_score(y_test, xgb_preds_labels):.4f}")
print(f"{'Attack Recall':<20} {recall_score(y_test, rf_preds, pos_label='attack'):<20.4f} {recall_score(y_test, xgb_preds_labels, pos_label='attack'):.4f}")
print(f"{'Macro F1':<20} {f1_score(y_test, rf_preds, average='macro'):<20.4f} {f1_score(y_test, xgb_preds_labels, average='macro'):.4f}")