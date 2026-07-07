import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 

# load data 
df = pd.read_csv("network_traffic.csv")
x = df[["duration", "packet_size", "num_connections"]]
y = df["label"]

# Train model 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Plot Feature Importance
importances = model.feature_importances_
features = x.columns

plt.figure(figsize=(8, 5))
plt.bar(features, importances, color=["#68875A", "#E3B43C", "#304f27"])
plt.title("Feature Importance — Network Intrusion Detection")
plt.xlabel("Feature")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

print("Chart saved as feature_importance.png")