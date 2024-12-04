from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
import time

app = Flask(__name__)
CORS(app)

# Global variables for model, scaler, and encoder
model_success, scaler, genre_encoder = None, None, None

# Function to load models and resources
def load_resources():
    global model_success, scaler, genre_encoder
    try:
        # Simulate loading time with a delay for loading model and scaler
        print("Loading resources... This may take a few moments.")
        time.sleep(2)  # Simulated delay for loading models
        
        # Load the movie success model
        with open('moviemodel_success.pkl', 'rb') as model_file:
            model_success = pickle.load(model_file)
        
        # Simulate additional delay for scaler and other resources
        time.sleep(1)  
        
        # Load the budget scaler
        with open('budget_scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)

        # Simulate delay after scaler loading
        time.sleep(1)
        
        # Using predefined genres instead of loading from file
        predefined_genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance']  # Update with actual genres
        genre_encoder = LabelEncoder()
        genre_encoder.fit(predefined_genres)  # Fit the encoder to predefined genres
        
        time.sleep(1)  # Simulate delay for genre classes loading

        print("Resources loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading resources: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while loading resources: {e}")

# Route for index page
@app.route('/')
def index():
    return render_template('page2.html')

# Route for prediction API
@app.route('/predict_win', methods=['POST'])
def predict_win():
    try:
        # Step 1: Receive data from the frontend
        data = request.get_json()

        # Expected fields from frontend
        expected_fields = ['actor_name', 'genre', 'budget', 'director', 'production_house']

        # Step 2: Validate inputs
        missing_fields = [field for field in expected_fields if field not in data]
        if missing_fields:
            return jsonify({
                "status": "error",
                "message": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400

        # Extract fields
        actor_name = data.get('actor_name')
        genre = data.get('genre')
        budget_inr = data.get('budget')
        director = data.get('director')
        production_house = data.get('production_house')

        # Validate budget
        try:
            budget_inr = int(budget_inr)
            if budget_inr < 0:
                raise ValueError("Budget must be non-negative.")
        except ValueError as ve:
            return jsonify({"status": "error", "message": str(ve)}), 400

        # Step 3: Prepare input DataFrame
        input_df = pd.DataFrame({
            'actor_name': [actor_name],
            'genre': [genre],
            'budget': [budget_inr],
            'director': [director],
            'production_house': [production_house]
        })

        # Step 4: Check if model and resources are loaded, with a generic delay before use
        if model_success is None or scaler is None or genre_encoder is None:
            return jsonify({"status": "error", "message": "Model or resources are still loading. Please try again later."}), 503

        # Adding a small delay to simulate processing time
        time.sleep(1)

        # Normalize budget
        input_df['budget'] = scaler.transform(input_df[['budget']])

        # Encode genre
        try:
            input_df['genre_encoded'] = genre_encoder.transform(input_df['genre'])
        except ValueError:
            return jsonify({
                "status": "error",
                "message": f"Genre '{genre}' not recognized. Available genres: {list(genre_encoder.classes_)}"
            }), 400

        # Step 5: Predict success
        if model_success:
            X_new = input_df[['genre_encoded', 'budget']]
            prediction = model_success.predict(X_new)
            
            result = "Success" if prediction[0] == 1 else "Flop"
            return jsonify({
                "status": "success",
                "actor_name": actor_name,
                "director": director,
                "genre": genre,
                "budget": budget_inr,
                "prediction": result
            }), 200
        else:
            return jsonify({"status": "error", "message": "Model not loaded."}), 500

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Running the app
if __name__ == '__main__':
    load_resources()
    app.run(debug=True)
