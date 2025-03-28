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

## 🚀 Nächste Schritte

- Deployment des Modells als REST-API via FastAPI oder AWS Lambda
- Integration des Scikit-Learn Moduls zur Risikoeinschätzung
- GitHub Actions CI/CD für automatisiertes Training & Deployment
- Morgen: Tests der XGBoost-Integration und Optimierung der Hyperparameter

---

## 👨‍💻 Autor

**Vadim Ott**  
→ [LinkedIn](https://linkedin.com/in/vadimott)  
→ Data Engineer & AI Developer mit Fokus auf Cloud & MLOps

---

## ⚠️ Lizenz

Dieses Projekt dient zu Demonstrationszwecken und ist nicht für den produktiven Einsatz gedacht.
