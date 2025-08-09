from flask import Flask, request, jsonify, render_template
import base64, json
import os
from io import BytesIO
from model.model import MyModel
from model.numpy_model import NumpyModel
from model.optimized_model import OptimizedModel
import numpy as np

HOST = '0.0.0.0'
PORT = 8888

app = Flask(__name__)

# Configuration: Choose which model to use
MODEL_TYPE = 'optimized'  # Options: 'optimized', 'numpy', 'original'

if MODEL_TYPE == 'optimized':
    print("Using Optimized CNN model for inference...")
    # Try to load the best model from notebook training first
    model_path = '../best_emnist_model.pth'
    if not os.path.exists(model_path):
        # Fallback to model directory
        model_path = './model/best_emnist_model.pth'
    if not os.path.exists(model_path):
        print(f"Warning: Optimized model not found at {model_path}")
        print("Please run the notebook training first or use fallback model")
        # Fallback to original model
        MODEL_TYPE = 'original'
    
    if MODEL_TYPE == 'optimized':
        model = OptimizedModel(model_path, 'cpu')
    
if MODEL_TYPE == 'numpy':
    print("Using NumPy model for inference...")
    model = NumpyModel('./model/trained_weights.pth', 'cpu')
elif MODEL_TYPE == 'original':
    print("Using original PyTorch model for inference...")
    model = MyModel('./model/trained_weights.pth', 'cpu')
CLASS_MAPPING = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/model-info')
def model_info():
    """Get information about the currently loaded model"""
    try:
        if hasattr(model, 'get_model_info'):
            info = model.get_model_info()
        else:
            info = {
                'model_type': MODEL_TYPE,
                'status': 'loaded'
            }
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict', methods=['GET','POST'])
def predict():
    results = {"prediction": "Empty", "probability": 0.0, "model_type": MODEL_TYPE}

    try:
        # get data
        input_img = BytesIO(base64.urlsafe_b64decode(request.form['img']))

        # Check if debug mode is requested
        debug_mode = request.form.get('debug', 'false').lower() == 'true'

        # model.predict method takes the raw data and output a vector of probabilities
        if hasattr(model, 'predict') and 'debug_visualization' in model.predict.__code__.co_varnames:
            res = model.predict(input_img, debug_visualization=debug_mode)
        else:
            res = model.predict(input_img)

        # Get prediction and probability
        prediction_idx = np.argmax(res)
        results["prediction"] = str(CLASS_MAPPING[prediction_idx])
        results["probability"] = float(np.max(res)) * 100
        
        # Add confidence level
        if results["probability"] > 80:
            results["confidence"] = "High"
        elif results["probability"] > 60:
            results["confidence"] = "Medium"
        else:
            results["confidence"] = "Low"
        
        # Add debug info if requested
        if debug_mode:
            results["debug"] = {
                "tensor_min": float(np.min(res)),
                "tensor_max": float(np.max(res)),
                "top_3_predictions": []
            }
            # Get top 3 predictions
            top_3_indices = np.argsort(res)[-3:][::-1]
            for idx in top_3_indices:
                results["debug"]["top_3_predictions"].append({
                    "character": str(CLASS_MAPPING[idx]),
                    "probability": float(res[idx]) * 100,
                    "class_index": int(idx)
                })
            
    except Exception as e:
        print(f"Prediction error: {e}")
        results["error"] = "Prediction failed"
        results["probability"] = 0.0
    
    # output data
    return json.dumps(results)

@app.route('/predict-debug', methods=['POST'])
def predict_debug():
    """Debug endpoint that always enables visualization and returns detailed info"""
    results = {"prediction": "Empty", "probability": 0.0, "model_type": MODEL_TYPE}

    try:
        # get data
        input_img = BytesIO(base64.urlsafe_b64decode(request.form['img']))

        # Force debug mode for this endpoint
        if hasattr(model, 'predict') and 'debug_visualization' in model.predict.__code__.co_varnames:
            res = model.predict(input_img, debug_visualization=True)
        else:
            res = model.predict(input_img)

        # Get prediction and probability
        prediction_idx = np.argmax(res)
        results["prediction"] = str(CLASS_MAPPING[prediction_idx])
        results["probability"] = float(np.max(res)) * 100
        
        # Add confidence level
        if results["probability"] > 80:
            results["confidence"] = "High"
        elif results["probability"] > 60:
            results["confidence"] = "Medium"
        else:
            results["confidence"] = "Low"
        
        # Add detailed debug information
        results["debug"] = {
            "tensor_min": float(np.min(res)),
            "tensor_max": float(np.max(res)),
            "tensor_mean": float(np.mean(res)),
            "tensor_std": float(np.std(res)),
            "all_predictions": [],
            "top_5_predictions": []
        }
        
        # Get all predictions
        for i, prob in enumerate(res):
            results["debug"]["all_predictions"].append({
                "character": str(CLASS_MAPPING[i]),
                "probability": float(prob) * 100,
                "class_index": int(i)
            })
        
        # Get top 5 predictions
        top_5_indices = np.argsort(res)[-5:][::-1]
        for idx in top_5_indices:
            results["debug"]["top_5_predictions"].append({
                "character": str(CLASS_MAPPING[idx]),
                "probability": float(res[idx]) * 100,
                "class_index": int(idx)
            })
            
        results["debug"]["message"] = "Debug images saved as debug_*.png in Flask app directory"
            
    except Exception as e:
        print(f"Prediction error: {e}")
        results["error"] = "Prediction failed"
        results["probability"] = 0.0
    
    # output data
    return json.dumps(results)

if __name__ == '__main__':
    
    app.run(host=HOST,
            debug=True,
            port=PORT)

