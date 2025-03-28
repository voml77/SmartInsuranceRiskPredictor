import json
import joblib
import pandas as pd

# Modell und Mapping laden
model = joblib.load("model/classic_model.pkl")
with open("model/class_mapping_tabular.json", "r") as f:
    class_mapping = json.load(f)

# Lambda-Handler
def lambda_handler(event, context):
    try:
        print("📥 Eingehendes Event:")
        print(json.dumps(event))

        body = json.loads(event.get("body", "{}"))
        df = pd.DataFrame([body])
        pred = model.predict_proba(df)[0]
        class_idx = int(pred.argmax())
        confidence = float(pred[class_idx]) * 100
        predicted_class = class_mapping[str(class_idx)]

        print("✅ Vorhersage erfolgreich:")
        print(f"Klasse: {predicted_class}, Confidence: {confidence:.2f}%")

        return {
            "statusCode": 200,
            "body": json.dumps({
                "predicted_class": predicted_class,
                "confidence": f"{confidence:.2f}%"
            })
        }

    except Exception as e:
        print("❌ Fehler im Lambda:")
        print(str(e))
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }