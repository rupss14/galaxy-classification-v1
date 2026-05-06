import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Load dataset
df = pd.read_csv(
    "Barchi19_Morph-catalog_670k-galaxies.csv",
    low_memory=False
)

# Convert TType to numeric
df['TType'] = pd.to_numeric(df['TType'], errors='coerce')

# Remove missing target values
df = df.dropna(subset=['TType'])

# Handle invalid values
df = df.replace(-9999, pd.NA)
df = df.dropna()

# Create target variable
df['GalaxyType'] = df['TType'].apply(
    lambda x: 0 if x < 0 else 1
)

# Select features
features = ['K', 'C', 'A', 'S', 'G2', 'H']

# Sample data for faster training
df_sample = df.sample(n=100000, random_state=42)

X = df_sample[features]
y = df_sample['GalaxyType']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10
)

# Train model
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Feature importance
importances = model.feature_importances_

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(feature_importance)

# Save model
joblib.dump(model, "galaxy_model.pkl")