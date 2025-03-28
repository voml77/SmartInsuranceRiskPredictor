import pandas as pd
import joblib
import json
import sys
import numpy as np

# Beispielkunde (du kannst diesen Dict anpassen oder dynamisch laden)
sample_customer = {
    "Age": 35,
    "Gender": 0,
    "Driving_License": 1,
    "Previously_Insured": 0,
    "Vehicle_Age": 1,
    "Vehicle_Damage": 1,
    "Annual_Premium": 30000,
    "Policy_Sales_Channel": 26,
    "Vintage": 150,
    "Region_Code": 28
}

# Modell laden
model = joblib.load("model/classic_model.pkl")

# Klassen-Mapping laden
with open("model/class_mapping_tabular.json", "r") as f:
    class_mapping = json.load(f)
idx_to_class = {int(k): v for k, v in class_mapping.items()}

# DataFrame erzeugen
df = pd.DataFrame([sample_customer])

# Spaltenreihenfolge anpassen
df = df[model.feature_names_in_]

# Vorhersage + Wahrscheinlichkeit
pred = model.predict(df)[0]
probs = model.predict_proba(df)[0]

predicted_class = idx_to_class[int(pred)]
confidence = np.max(probs)

print(f"🧠 Vorhergesagte Klasse: {predicted_class} ({confidence:.2%} Confidence)")
print(f"✅ Low Risk: {probs[0]:.2%} | ⚠️ High Risk: {probs[1]:.2%}")
