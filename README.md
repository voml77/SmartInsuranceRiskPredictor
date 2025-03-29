# 🚗 Smart Insurance Risk Predictor

Ein hybrides Deep Learning & Machine Learning System zur Klassifikation von Versicherungsschäden – basierend auf **Bilderkennung** und **strukturierter Risikobewertung**.

---

## 🔍 Use Case

Versicherungen müssen täglich Risiken einschätzen – ob anhand von Kundendaten oder Schadensbildern.  
Dieses Projekt kombiniert beides:

- 📊 **Tabellarische Daten**: XGBoost-Modell (Scikit-Learn API) klassifiziert Kundenrisiko auf Basis strukturierter Daten
- 🖼️ **Bilddaten (Car Damage)**: CNN-Modell erkennt beschädigte Fahrzeuge

---

## ⚙️ Tech Stack

| Kategorie     | Tools / Frameworks                    |
|---------------|----------------------------------------|
| ML / DL       | PyTorch, Scikit-Learn                 |
| Cloud         | AWS (S3, SageMaker, Terraform)        |
| CI/CD         | GitHub Actions                        |
| Deployment    | API Gateway + Lambda (geplant)        |

---

## 📁 Projektstruktur

```
SmartInsuranceRiskPredictor/
├── data/                # Bilder + strukturierte Daten
├── model/               # Trainierte Modelle (.pth, .pkl)
├── src/
│   ├── classic_ml/      # Scikit-Learn Logik
│   ├── deep_learning/   # PyTorch CNN
│   └── utils/           # Hilfsfunktionen & Datenaufbereitung
├── terraform/           # Infrastrukturdefinition (AWS)
└── .github/workflows/   # GitHub Actions CI/CD
```

---

## 📈 Modell-Performance (CNN)

- **Train Accuracy**: 96.94%
- **Validation Accuracy**: 84.35%
- **Train Loss**: 0.0819
- **Epochs**: 100
- **Augmentation aktiv:** ✅  
- **Regularisierung (Dropout):** ✅

---

## 📊 Modell-Performance (XGBoost)

- **Validation Accuracy**: 87.47%
- **Modell**: XGBoost Classifier (`sklearn` API)
- **Features**: Alter, Region, Versicherungsverlauf, Fahrzeugdaten u.a.
- **Feature Encoding**: LabelEncoder + manuelle Selektion

---

## 🚀 Anwendung & Weiterentwicklung

### 📦 Installation

```bash
pip install -r requirements.txt
```

Alternativ: Nutzung via Docker oder als AWS Lambda (siehe `lambda_py39/`)

---

### 🔍 Beispielaufruf – Kombiniertes Modell

```bash
python src/combine_models/predict_combined.py
```

Beispielausgabe:

```
📄 Klassen-Index-Mapping gespeichert: model/class_mapping.json  
🔍 CNN: 01-whole (100.00%)  
📊 XGBoost: Low Risk (91.13%)  
🧮 Kombinierter Risiko-Score: 3.55%
```

---

### 🐳 Lokaler Lambda-Test via Docker

Für lokale Tests der Lambda-Funktion (z. B. XGBoost Vorhersage per REST):

```bash
cd lambda_py39
docker build -t lambda-xgb-local .
docker run -p 9000:8080 lambda-xgb-local
```

Dann lokal testen via:

```bash
curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" \
  -H "Content-Type: application/json" \
  -d '{ ... }'
```

---

### ⚠️ Hinweis zu den Modellen

> 📁 Die trainierten Modelle (`deep_model.pth`, `classic_model.pkl`) wurden **nicht in GitHub hochgeladen** (Dateigröße).  
> Bei Interesse oder für Reproduktion → bitte manuell hinzufügen oder Kontakt aufnehmen.

---

## 👨‍💻 Autor

**Vadim Ott**  
→ [LinkedIn](https://www.linkedin.com/in/vadim-ott-b66429251)  
→ Data Engineer & AI Developer mit Fokus auf Cloud & MLOps

---

## ⚠️ Lizenz

Dieses Projekt dient zu Demonstrationszwecken und ist nicht für den produktiven Einsatz gedacht.
