from flask import Flask, request, jsonify
import joblib
import numpy as np
import sklearn

app = Flask(__name__)

# Load your trained machine learning model
model = joblib.load('random_forest_model.pkl')

@app.route('/')
def hello():
    try:
        return 'Hello, World!'
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from POST request
        data = request.json
        
        # Convert data to numpy array for prediction
        input_data = np.array(list(data.values())).reshape(1, -1)
        
        # Make prediction using the loaded model
        prediction = model.predict(input_data)
        
        # Prepare the prediction result
        output = {'prediction': int(prediction[0])}  # Assuming it's a single value prediction
        
        return jsonify(output)
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
