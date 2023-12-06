from flask import Flask, request, jsonify
import joblib
import numpy as np


app = Flask(__name__)

# Load your trained machine learning model
#model = joblib.load('random_forest_model.pkl')

@app.route('/')
def hello():
    try:
        return 'Hello, World!'
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
