# SDG-Solutions
🩺 SDG 3: Diabetes Risk Prediction using Machine Learning
🌍 Project Overview

This project addresses UN Sustainable Development Goal 3: Good Health and Well-being by building a machine learning model to predict diabetes risk based on health indicators.
The model uses patient data (e.g., BMI, glucose level, blood pressure) to classify whether a person is likely to develop diabetes.

By enabling early detection, this project helps healthcare professionals and policymakers better allocate resources and support healthier communities.

📊 Dataset

Name: Pima Indians Diabetes Database

Source: Kaggle

Features:

Pregnancies

Glucose

BloodPressure

SkinThickness

Insulin

BMI

DiabetesPedigreeFunction

Age

Target: Outcome (0 = No Diabetes, 1 = Diabetes)

🛠️ Tools & Libraries

Language: Python

Environment: Google Colab / Jupyter Notebook

Libraries:

pandas, numpy (data processing)

matplotlib, seaborn (visualization)

scikit-learn (ML models & evaluation)

🔬 Workflow

Data Preprocessing

Cleaned and standardized features.

Split dataset into training (80%) and testing (20%).

Model Training

Logistic Regression (baseline).

Random Forest Classifier (final).

Evaluation Metrics

Accuracy

Precision, Recall, F1-score

Confusion Matrix

Results

Random Forest achieved higher accuracy than Logistic Regression.

Feature importance showed Glucose, BMI, and Age were the strongest predictors.

📈 Sample Results
Confusion Matrix

Feature Importance

⚖️ Ethical Reflection

Bias: Dataset represents only Pima Indian women → may not generalize globally.

Privacy: Patient health data must always be anonymized.

Fairness: Predictions must be used to support doctors, not replace them.

Impact: Early prediction can save lives and reduce healthcare burden
