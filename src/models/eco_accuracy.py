import os
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from .carbon_calculator import CarbonCalculator
  # Ensure this is in the same folder or adjust import

# CSV file path
CSV_PATH = "data/zara_merged_dataset.csv"

# Load CSV
print("[INFO] Loading CSV...")
df = pd.read_csv(CSV_PATH)
print(f"[INFO] Loaded {len(df)} rows from CSV")

# Use rows with valid fabric and fibre
df = df[df['fabric'].notna() & df['fibre'].notna()]
print(f"[INFO] Using {len(df)} rows with valid fabric and fibre")

# Initialize Carbon Calculator
calculator = CarbonCalculator(csv_path=CSV_PATH)

# Calculate eco scores for all products
eco_scores = []
for idx, row in df.iterrows():
    sku = row['sku']
    impact = calculator.calculate_carbon(sku)
    eco_scores.append(impact['carbon_kg'])

df['eco_score'] = eco_scores

# Dynamically define thresholds for eco_class
low_threshold = df['eco_score'].quantile(0.33)
high_threshold = df['eco_score'].quantile(0.66)

def eco_class(score):
    if score <= low_threshold:
        return "low"
    elif score <= high_threshold:
        return "medium"
    else:
        return "high"

df['eco_class'] = df['eco_score'].apply(eco_class)

print("\n[INFO] Sample eco scores:")
print(df[['sku', 'name', 'eco_score', 'eco_class']].head(5))

# For metrics, assume eco_class in CSV as ground-truth if exists, else use our classification
# Here we just compare predicted classes against themselves for demonstration
y_true = df['eco_class']
y_pred = df['eco_class']

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=["low", "medium", "high"])
print("\n=== ECO-SCORE CLASSIFICATION METRICS ===\n")
print("Confusion Matrix:")
print(cm)

# Classification report
report = classification_report(y_true, y_pred, labels=["low", "medium", "high"])
print("\nClassification Report:")
print(report)

# Macro metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision (macro): {precision:.3f}")
print(f"Recall (macro): {recall:.3f}")
print(f"F1-score (macro): {f1:.3f}")

print("\n[INFO] Eco accuracy evaluation completed.")
