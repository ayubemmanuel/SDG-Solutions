# SDG 3: Diabetes Risk Prediction
# ML Approach: Supervised Learning (Classification)
# Dataset: Pima Indians Diabetes Dataset (Kaggle)
# Author: Your Name
# Date: 2025-10-02

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Step 2: Load Dataset
# Dataset link: https://www.kaggle.com/uciml/pima-indians-diabetes-database
df = pd.read_csv("diabetes.csv")  # Upload diabetes.csv to Colab or your environment

# Step 3: Explore Dataset
print("Dataset Shape:", df.shape)
print(df.head())
print(df.info())
print(df.describe())

# Check missing values
print("Missing Values:\n", df.isnull().sum())

# Step 4: Data Visualization
sns.countplot(x='Outcome', data=df)
plt.title("Diabetes Outcome Distribution (0=No, 1=Yes)")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()

# Step 5: Data Preprocessing
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 6: Train Model
# Option 1: Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

# Option 2: Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)

# Step 7: Evaluate Models
def evaluate_model(y_test, y_pred, model_name):
    print(f"--- {model_name} ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Evaluate Logistic Regression
evaluate_model(y_test, y_pred_log, "Logistic Regression")

# Evaluate Random Forest
evaluate_model(y_test, y_pred_rf, "Random Forest Classifier")

# Step 8: Feature Importance (Random Forest)
feat_importances = pd.Series(rf_clf.feature_importances_, index=X.columns)
feat_importances.sort_values().plot(kind='barh', figsize=(8,6))
plt.title("Feature Importance - Random Forest")
plt.show()

# Step 9: Ethical Reflection (Print Summary)
print("""
Ethical Considerations:
1. Data Bias: The dataset represents Pima Indian females, so predictions may not generalize to other populations.
2. Privacy: Patient health data must be anonymized before use.
3. Fairness: Ensure model predictions are interpreted responsibly and do not misclassify certain groups.
4. Sustainability: This ML solution can help early detection and better allocation of healthcare resources, supporting SDG 3.
""")
