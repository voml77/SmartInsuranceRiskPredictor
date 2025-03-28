import requests
import json

# 🛑 Deine API-URL hier eintragen (aus Terraform Output oder AWS Console)
API_URL = "https://vn9p83i3dk.execute-api.eu-central-1.amazonaws.com/predict"

# Beispielkunde (muss mit Modell-Features übereinstimmen)
sample_customer = {
    "Gender": "Male",
    "Age": 35,
    "Driving_License": 1,
    "Previously_Insured": 0,
    "Vehicle_Age": "1-2 Year",
    "Vehicle_Damage": "Yes",
    "Annual_Premium": 35000,
    "Policy_Sales_Channel": 152,
    "Vintage": 230,
    "Region_Code": 28.0
}

def call_api(data):
    headers = {"Content-Type": "application/json"}
    response = requests.post(API_URL, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        print("✅ Erfolgreiche Vorhersage:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"❌ Fehler beim Aufruf: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    call_api(sample_customer)
