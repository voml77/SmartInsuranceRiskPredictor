import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd
import joblib
import json
import numpy as np
import sys
import os

sys.path.append(os.path.abspath("src/deep_learning"))
from train_model import InsuranceCNN

# Beispielkunde
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

# Beispielbildpfad
image_path = "data/images/test/00-damage/0161.jpeg"

# Gerät wählen
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# 🔹 CNN-Inferenz
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

model_cnn = InsuranceCNN().to(device)
model_cnn.load_state_dict(torch.load("model/deep_model.pth", map_location=device))
model_cnn.eval()

with open("model/class_mapping.json", "r") as f:
    class_mapping_cnn = json.load(f)
idx_to_class_cnn = {v: k for k, v in class_mapping_cnn.items()}

with torch.no_grad():
    output = model_cnn(image)
    _, pred_cnn = torch.max(output.data, 1)
    cnn_conf = torch.nn.functional.softmax(output, dim=1)[0][pred_cnn.item()].item()
    cnn_label = idx_to_class_cnn[pred_cnn.item()]

# 🔹 XGBoost-Inferenz
model_xgb = joblib.load("model/classic_model.pkl")
with open("model/class_mapping_tabular.json", "r") as f:
    class_mapping_xgb = json.load(f)
idx_to_class_xgb = {int(k): v for k, v in class_mapping_xgb.items()}

df_customer = pd.DataFrame([sample_customer])
df_customer = df_customer[model_xgb.feature_names_in_]

pred_xgb = model_xgb.predict(df_customer)[0]
xgb_probs = model_xgb.predict_proba(df_customer)[0]
xgb_conf = np.max(xgb_probs)
xgb_label = idx_to_class_xgb[int(pred_xgb)]

# 🔁 Kombinierter Score
cnn_probs = torch.nn.functional.softmax(output, dim=1)[0].cpu().numpy()
cnn_risk = cnn_probs[list(idx_to_class_cnn.values()).index("00-damage")]
xgb_risk = xgb_probs[list(idx_to_class_xgb.values()).index("High Risk")]

combined_score = (cnn_risk * 0.6) + (xgb_risk * 0.4)

# 📊 Ausgabe
print(f"🔍 CNN: {cnn_label} ({cnn_conf:.2%})")
print(f"📊 XGBoost: {xgb_label} ({xgb_conf:.2%})")
print(f"🧮 Kombinierter Risiko-Score: {combined_score:.2%}")
