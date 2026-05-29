import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Dataset
data = pd.read_csv("dataset.csv")

# Features and Labels
X = data["email"]
y = data["label"]

# Convert text into numbers
vectorizer = CountVectorizer()

X_vectorized = vectorizer.fit_transform(X)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.3,
    random_state=42
)

# Train Model
model = MultinomialNB()

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n===================================")
print(" PHISHING EMAIL DETECTION MODEL")
print("===================================")

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# User Input
print("\n===================================")

sample = input("Enter Email Text: ")

sample_data = vectorizer.transform([sample])

prediction = model.predict(sample_data)

print("\nPrediction Result:")

if prediction[0] == "phishing":
    print("⚠️ This Email is PHISHING")
else:
    print("✅ This Email is SAFE")

# Save Report
report = f"""
========== PHISHING EMAIL REPORT ==========

Accuracy: {accuracy * 100:.2f}%

Prediction:
{prediction[0]}
"""

with open("report.txt", "w") as file:
    file.write(report)

print("\nReport saved as report.txt")