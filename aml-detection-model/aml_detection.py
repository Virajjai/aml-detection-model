import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve

df = pd.read_csv('PS_20174392719_1491204439457_log.csv')
print("Dataset Loaded Successfully")

# STEP 3: Initial Exploration
print("\nDataset Shape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())
print("\nData Types:")
print(df.dtypes)

# Display first few rows
df.head()

# STEP 4: Data Cleaning and Preprocessing
# Drop identifier columns that don't help in prediction
df.drop(['nameOrig', 'nameDest'], axis=1, inplace=True)

# Encode 'type' categorical feature using one-hot encoding
df = pd.get_dummies(df, columns=['type'], drop_first=True)


# STEP 5: Exploratory Data Analysis
sns.countplot(x='isFraud', data=df)
plt.title('Fraudulent vs. Non-Fraudulent Transactions')
plt.show()

print("\n")

# Count fraud vs non-fraud transactions
fraud_counts = df['isFraud'].value_counts()
labels = ['Non-Fraudulent', 'Fraudulent']
sizes = [fraud_counts[0], fraud_counts[1]]
colors = ['#66b3ff', '#ff6666']
explode = (0, 0.1)  # Explode the Fraud slice

# Print exact figures
print("Fraudulent Transactions:", fraud_counts[1])
print("Non-Fraudulent Transactions:", fraud_counts[0])
print("Total Transactions:", fraud_counts.sum())

# Plot pie chart
plt.figure(figsize=(6,6))
plt.pie(sizes, labels=labels, autopct='%1.2f%%', startangle=90, colors=colors, explode=explode, shadow=True)
plt.title('Fraudulent vs Non-Fraudulent Transactions')
plt.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle
plt.show()

print("\n")

# Heatmap to show feature correlations
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# STEP 6: Feature-Target Split
X = df.drop(['isFraud', 'isFlaggedFraud'], axis=1)
y = df['isFraud']

# Handle missing values in the target variable y
# Drop rows where the target variable is NaN
initial_rows = df.shape[0]
df.dropna(subset=['isFraud'], inplace=True)
rows_after_dropna = df.shape[0]
if initial_rows != rows_after_dropna:
    print(f"Dropped {initial_rows - rows_after_dropna} rows with missing 'isFraud' values.")

# Re-create X and y after dropping rows
X = df.drop(['isFraud', 'isFlaggedFraud'], axis=1)
y = df['isFraud']

# STEP 7: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("Training Set Size:", X_train.shape)
print("Test Set Size:", X_test.shape)


# STEP 8: Train the ML Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model Trained Successfully")

# STEP 9: Evaluate the Model
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print("\n")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(roc_auc_score(y_test, y_proba)))
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend()
plt.grid()
plt.show()


# STEP 10: Feature Importance
# Need to use the columns from X_train or X_test since X might have changed shape
importances = model.feature_importances_
features = X_train.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))
plt.title('Top 10 Important Features')
plt.tight_layout()
plt.show()


# STEP 11: Summary
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))
print("\nTop Features:\n")
print(importance_df.head(10))




def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
