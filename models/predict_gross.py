import sys
import time
from flask import Flask, request, jsonify, render_template
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
from flask_cors import CORS
sys.path.insert(0, '.\\project_root\\models')
from predict_gross import train_model, predict_new_movie, feature_sets, load_and_prepare_data

# Flask app setup
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing (CORS) for frontend-backend communication

# Load data and prepare models
FILEPATH = 'C:\\Users\\Ravi\\Desktop\\ug_bmsit\\projects\\AIML_boxoffice\\project_root\\data\\merged_1.csv'
data = load_and_prepare_data(FILEPATH)

# Train models for all target variables (train only once at the start of the server)
models = {}
for target, features in feature_sets.items():
    X = data[features]
    y = data[target]
    task = 'classification' if target == 'success' else 'regression'
    model, _, _ = train_model(X, y, model_type=task)
    models[target] = model

@app.route('/')
def index():
    return render_template('index.html')  # Renders the frontend page

@app.route('/predict_gross', methods=['POST'])
def predict_gross():
    try:
        # Parse incoming JSON data from frontend
        input_data = request.json
        if not input_data:
            return jsonify({'error': 'No input data provided!'}), 400

        predictions = {}
        plots = {}

        # For each target (score, gross, profitability, etc.), predict and generate plot
        # Simulate processing delay for each plot (e.g., 2 seconds delay per plot)
        for idx, (target, model) in enumerate(models.items()):
            # Introduce a time delay between processing
            time.sleep(2)  # Delay of 2 seconds for each prediction/plot
            
            features = feature_sets[target]
            input_df = pd.DataFrame([input_data])[features]
            
            # Make prediction for the input data using the model
            prediction = predict_new_movie(model, input_df)[0]
            predictions[target] = prediction

            # Generate a plot for the prediction
            plt.figure()
            plt.bar([f"{target}_Prediction"], [prediction], color='skyblue')
            plt.title(f"{target.capitalize()} Prediction")
            plt.xlabel('Target')
            plt.ylabel('Value')

            # Save the plot as a base64-encoded string to include in the response
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plots[target] = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            plt.close()

        # Return predictions and plots in JSON format after a 3-second delay
        time.sleep(2)  # Optional additional delay before sending response
        response = {
            'predictions': predictions,
            'plots': plots
        }
        return jsonify(response)

    except Exception as e:
        # Handle unexpected errors
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
