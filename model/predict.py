
import joblib
import json
import os

def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        data = json.loads(request_body)
        # Handle both CamelCase and snake_case keys
        return {
            "Ambient_Temperature": data.get("Ambient_Temperature", data.get("ambient_temperature")),
            "Humidity": data.get("Humidity", data.get("humidity")),
            "Soil_Moisture": data.get("Soil_Moisture", data.get("soil_moisture")),
            "Light_Intensity": data.get("Light_Intensity", data.get("light_intensity")),
            "Rainfall": data.get("Rainfall", data.get("rainfall")),
            "Annual CO₂ emissions (tonnes )": data.get("Annual CO₂ emissions (tonnes )", data.get("annual_co2_ppm"))
        }
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    features = [
        input_data["Ambient_Temperature"],
        input_data["Humidity"],
        input_data["Soil_Moisture"],
        input_data["Light_Intensity"],
        input_data["Rainfall"],
        input_data["Annual CO₂ emissions (tonnes )"]
    ]
    prediction = model.predict([features])
    return {"irrigation_needed": int(prediction[0])}

def output_fn(prediction, content_type):
    return json.dumps(prediction)
