import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import pickle  # Import pickle to save and load the model

# Load the historical data
historical_data = pd.read_csv('.\\data\\cleanedmovies_2.csv')

# Encode the 'director' and 'star' categorical columns using LabelEncoder
le_director = LabelEncoder()
le_star = LabelEncoder()

historical_data['director_encoded'] = le_director.fit_transform(historical_data['director'])
historical_data['star_encoded'] = le_star.fit_transform(historical_data['star'])

# Select features and target
X = historical_data[['director_encoded', 'star_encoded', 'votes']]
y = historical_data['gross']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training - Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
print("Mean Absolute Error on Test Set:", mean_absolute_error(y_test, y_pred))

# Save the entire process (model, encoders) to a single pickle file
model_data = {
    'model': model,
    'director_encoder': le_director,
    'star_encoder': le_star
}

with open('movie_gross_predictor_full_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("\nEntire model and encoders saved to a single pickle file.")

# To load the model and encoders from the pickle file
with open('movie_gross_predictor_full_model.pkl', 'rb') as f:
    loaded_model_data = pickle.load(f)

# Accessing the loaded components
loaded_model = loaded_model_data['model']
loaded_le_director = loaded_model_data['director_encoder']
loaded_le_star = loaded_model_data['star_encoder']

# Take real-time input for a new movie
name_new = input("Enter the new movie name: ")
director_new = input("Enter the director's name: ")
star_new = input("Enter the star actor's name: ")
votes_new = int(input("Enter the number of votes: "))

# Prepare the input data for prediction
# Encoding 'director' and 'star' names for the new input
if director_new in loaded_le_director.classes_:
    director_encoded_new = loaded_le_director.transform([director_new])[0]
else:
    print("Director not found in historical data. Assigning as unknown.")
    # Use a fallback value like -1 or the median director encoding
    director_encoded_new = loaded_le_director.transform([loaded_le_director.classes_[0]])[0]

if star_new in loaded_le_star.classes_:
    star_encoded_new = loaded_le_star.transform([star_new])[0]
else:
    print("Star not found in historical data. Assigning as unknown.")
    # Use a fallback value like -1 or the median star encoding
    star_encoded_new = loaded_le_star.transform([loaded_le_star.classes_[0]])[0]

# Create a DataFrame for the input to match the model's expected input structure
upcoming_movie_df = pd.DataFrame({
    'director_encoded': [director_encoded_new],
    'star_encoded': [star_encoded_new],
    'votes': [votes_new]
})

# Make prediction for the new movie using the loaded model
predicted_gross = loaded_model.predict(upcoming_movie_df)[0]
print(f"\nPredicted Gross for '{name_new}': ${predicted_gross:,.2f}")




