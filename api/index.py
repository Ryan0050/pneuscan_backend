from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from flask_cors import CORS
import os

app = Flask(__name__)

# Fix CORS to allow your Vercel domain
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://pneuscan.vercel.app",
            "http://localhost:3000"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'models', 'pneumonia_model.keras')
model = load_model(model_path)
labels = ['PNEUMONIA', 'NORMAL']
img_size = 200

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "PneuScan API is running", "status": "ok"})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return '', 200
    
    if 'image' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    images = request.files.getlist('image')
    predictions = []
    
    try:
        for img_file in images:
            img_arr = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
            resized_arr = cv2.resize(img_arr, (img_size, img_size)).reshape(1, img_size, img_size, 1)
            normalized_img = resized_arr / 255.0
            prediction = model.predict(normalized_img, verbose=0)
            confidence = float(prediction[0][0])
            
            if confidence > 0.80:
                label = 'NORMAL'
            else:
                label = 'PNEUMONIA'
            
            predictions.append({'result': label})
        
        return jsonify(predictions), 200
    
    except Exception as e:
        print(f"Error: {str(e)}")  # This will show in Railway logs
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
