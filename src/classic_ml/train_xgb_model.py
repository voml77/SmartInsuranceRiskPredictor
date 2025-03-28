import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import json
import os

# Daten laden
df = pd.read_csv("data/customers/train.csv")

# Zielspalte
target_col = "Response"

# Drop ID-Spalte, falls vorhanden
if "id" in df.columns:
    df = df.drop(columns=["id"])

# Kategorische Features enkodieren (Label Encoding für Einfachheit)
categorical_cols = df.select_dtypes(include="object").columns.tolist()
for col in categorical_cols:
    df[col] = df[col].astype("category").cat.codes

# Features und Ziel
X = df.drop(columns=[target_col])
y = df[target_col]

# Klassen-Mapping
class_mapping = {0: "Low Risk", 1: "High Risk"}
with open("model/class_mapping_tabular.json", "w") as f:
    json.dump(class_mapping, f)
print("📄 Klassen-Mapping gespeichert: model/class_mapping_tabular.json")

# Split in Train/Test
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Modell trainieren
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
model.fit(X_train, y_train)

# Validierung
y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"✅ Validation Accuracy: {acc:.2%}")

# Modell speichern
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/classic_model.pkl")
print("💾 Modell gespeichert: model/classic_model.pkl")
