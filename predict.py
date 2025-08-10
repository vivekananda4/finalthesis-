
import joblib
import json
import os

def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "irrigation_model.joblib"))
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        return json.loads(request_body)
    raise ValueError("Unsupported content type: " + request_content_type)

def predict_fn(input_data, model):
    data = [input_data['Ambient_Temperature'],
            input_data['Humidity'],
            input_data['Soil_Moisture'],
            input_data['Light_Intensity'],
            input_data['Rainfall'],
            input_data['Annual CO₂ emissions (tonnes )']]
    prediction = model.predict([data])
    return {"irrigation_needed": int(prediction[0])}

def output_fn(prediction, content_type):
    return json.dumps(prediction)
