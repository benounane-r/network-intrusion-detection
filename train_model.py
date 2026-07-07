import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# load the dataset
df=pd.read_csv("network_traffic.csv")

print("dataset loaded")
print(df.head(10)) # Display the first 10 rows of the dataset


# split the dataset into features and labels
# where the features are the columns duration, packet_size and num_connections and the label is the column label
# The label is used to indicate whether the connection is normal or an attack
X = df[["duration", "packet_size", "num_connections"]]
Y = df[["label"]]

# split the dataset into training and testing sets
# we are using 80% of the data for training and 20% for testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# now we are creating and training a model using Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train,Y_train)

# now we are making predictions on the test set
predictions = model.predict(X_test)

# now we are evaluating the model
# what it got correct
print("\nClassification Report:")
print(classification_report(Y_test, predictions))
# what it got wrong
print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, predictions))

#adding the features importance - which metric matters the most in making the prediction
importances = model.feature_importances_
print("\n----Feature Importance -----")
for feature, importance in zip(X.columns, importances):
    print(f"Feature : {feature:20} Importance: {importance:.4f}")
