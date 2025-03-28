import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from train_model import InsuranceCNN
import json

# Gerät erkennen
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("🖥️ Verwende Gerät:", device)

# Transforms wie im Training
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Testdaten laden
test_dataset = datasets.ImageFolder("data/images/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Modell laden
model = InsuranceCNN().to(device)
model.load_state_dict(torch.load("model/deep_model.pth", map_location=device))
model.eval()

# Klassen-Mapping laden
with open("model/class_mapping.json", "r") as f:
    class_mapping = json.load(f)
idx_to_class = {v: k for k, v in class_mapping.items()}

# Evaluation
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_acc = 100 * correct / total
print(f"✅ Test Accuracy: {test_acc:.2f}%")
