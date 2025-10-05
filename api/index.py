from flask import Flask, request, jsonify
import tensorflow as tf
import cv2
import numpy as np
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["https://pneuscan.vercel.app", "http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "supports_credentials": False
    }
})

# Load TFLite model
model_path = os.path.join(os.path.dirname(__file__), 'models', 'pneumonia_model.tflite')
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

labels = ['PNEUMONIA', 'NORMAL']
img_size = 200

@app.route('/')
def home():
    return jsonify({"message": "PneuScan API is running", "status": "ok"})

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 200
    
    if 'image' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    images = request.files.getlist('image')
    predictions = []
    
    try:
        for img_file in images:
            img_arr = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
            resized_arr = cv2.resize(img_arr, (img_size, img_size))
            input_data = resized_arr.reshape(1, img_size, img_size, 1).astype(np.float32)
            normalized_img = input_data / 255.0
            
            interpreter.set_tensor(input_details[0]['index'], normalized_img)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])
            
            confidence = float(prediction[0][0])
            
            if confidence > 0.80:
                label = 'NORMAL'
            else:
                label = 'PNEUMONIA'
            
            predictions.append({'result': label})
        
        return jsonify(predictions), 200
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
