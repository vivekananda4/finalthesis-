import json
import os
import joblib
import numpy as np

# 6 features in this order (as in training):
# [Ambient_Temperature, Humidity, Soil_Moisture, Light_Intensity, Rainfall, Annual CO2]
FEATURES = 6

def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))

def _to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)

def _map_from_dict(d):
    # tolerate many key variants
    temp = d.get("Ambient_Temperature", d.get("ambient_temperature", d.get("temperature")))
    hum  = d.get("Humidity", d.get("humidity"))
    soil = d.get("Soil_Moisture", d.get("soil_moisture", d.get("soilMoisture")))
    light= d.get("Light_Intensity", d.get("light_intensity", d.get("light")))
    rain = d.get("Rainfall", d.get("rainfall", d.get("rain")))
    co2  = (d.get("Annual CO₂ emissions (tonnes )")
            or d.get("annual_co2_ppm")
            or d.get("annual_co2_emissions")
            or d.get("co2"))
    return [
        _to_float(temp, 0.0),
        _to_float(hum, 0.0),
        _to_float(soil, 0.0),
        _to_float(light, 0.0),
        _to_float(rain, 0.0),
        _to_float(co2, 0.0),
    ]

def input_fn(request_body, request_content_type):
    ct = (request_content_type or "").lower()

    # ---- CSV: "t,hum,soil,light,rain,co2"
    if ct.startswith("text/csv"):
        parts = [p.strip() for p in request_body.replace("\n","").split(",") if p.strip() != ""]
        vals = [_to_float(p, 0.0) for p in parts]
        if len(vals) < FEATURES:
            raise ValueError(f"CSV expects {FEATURES} values; got {len(vals)}")
        vals = vals[:FEATURES]
        return np.array([vals], dtype=float)

    # ---- JSON formats
    if ct.startswith("application/json"):
        data = json.loads(request_body)

        # {"instances":[[...]]} or {"inputs":[[...]]}
        if isinstance(data, dict):
            rows = data.get("instances") or data.get("inputs")
            if rows:
                row = rows[0]
                if isinstance(row, dict):
                    vals = _map_from_dict(row)
                else:
                    vals = [_to_float(v, 0.0) for v in list(row)]
                    if len(vals) < FEATURES:
                        raise ValueError(f"instances row must have {FEATURES} values")
                    vals = vals[:FEATURES]
                return np.array([vals], dtype=float)

            # flat dict
            vals = _map_from_dict(data)
            return np.array([vals], dtype=float)

        # list or list-of-lists
        if isinstance(data, list):
            row = data[0] if (data and isinstance(data[0], (list, dict))) else data
            if isinstance(row, dict):
                vals = _map_from_dict(row)
            else:
                vals = [_to_float(v, 0.0) for v in list(row)]
                if len(vals) < FEATURES:
                    raise ValueError(f"JSON list must have {FEATURES} values")
                vals = vals[:FEATURES]
            return np.array([vals], dtype=float)

    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    pred = model.predict(input_data)           # shape: [1]
    return {"prediction": int(pred[0])}

def output_fn(prediction, content_type):
    return json.dumps(prediction)
