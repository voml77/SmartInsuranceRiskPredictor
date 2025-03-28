import sys
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
from train_model import InsuranceCNN

# Bildpfad als Argument
if len(sys.argv) != 2:
    print("❌ Verwendung: python predict.py <bildpfad>")
    sys.exit(1)

image_path = sys.argv[1]

# Gerät erkennen
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Modell laden
model = InsuranceCNN().to(device)
model.load_state_dict(torch.load("model/deep_model.pth", map_location=device))
model.eval()

# Klassen-Mapping laden
with open("model/class_mapping.json", "r") as f:
    class_mapping = json.load(f)
idx_to_class = {v: k for k, v in class_mapping.items()}

# Transforms wie beim Training
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Bild laden und vorbereiten
try:
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
except Exception as e:
    print(f"❌ Fehler beim Laden des Bildes: {e}")
    sys.exit(1)

# Vorhersage
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output.data, 1)
    predicted_class = idx_to_class[predicted.item()]

    # Softmax-Konfidenz berechnen
    confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted.item()].item()

print(f"🧠 Vorhergesagte Klasse: {predicted_class} ({confidence:.2%} Confidence)")
