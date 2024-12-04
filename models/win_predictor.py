import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load your dataset
data = pd.read_csv(r".\\project_root\\data\\processed_dataset_with_success_updated (1).csv")

# Step 2: Prepare features (X) and targets (y) for predicting gross and profitability
X = data[['genre_encoded', 'budget', 'star_encoded', 'score']]  # Features
y_gross = data['gross']  # Target for gross prediction
y_profitability = data['Profitability']  # Target for profitability prediction

# Initialize MinMaxScaler (fit on budget column from the dataset)
scaler = MinMaxScaler()
scaler.fit(data[['budget']])  # Fit the scaler on the existing normalized budget column

# Save the scaler for later reuse
with open('budget_scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Step 3: Train-Test Split (80% training, 20% testing)
X_train, X_test, y_train_gross, y_test_gross = train_test_split(X, y_gross, test_size=0.1, random_state=42)
_, _, y_train_profitability, y_test_profitability = train_test_split(X, y_profitability, test_size=0.1, random_state=42)

# Step 4: Train RandomForestRegressor models
model_gross = RandomForestRegressor(random_state=42, n_estimators=100)
model_profitability = RandomForestRegressor(random_state=42, n_estimators=100)

model_gross.fit(X_train, y_train_gross)
model_profitability.fit(X_train, y_train_profitability)

# Step 5: Predict gross and profitability
predicted_gross = model_gross.predict(X)
predicted_profitability = model_profitability.predict(X)

# Align predictions with the entire dataset
data['predicted_gross'] = pd.Series(predicted_gross, index=X.index)
data['predicted_profitability'] = pd.Series(predicted_profitability, index=X.index)

# Save predictions
with open('predicted_gross.pkl', 'wb') as gross_file:
    pickle.dump(predicted_gross, gross_file)

with open('predicted_profitability.pkl', 'wb') as profitability_file:
    pickle.dump(predicted_profitability, profitability_file)

# Step 6: Prepare features and target for movie success prediction
X = data[['genre_encoded', 'budget', 'predicted_gross', 'predicted_profitability']]
y = data['success']  # Use existing 'success' column

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Step 7: Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


# Step 8: Train and evaluate the RandomForestClassifier
model_success = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')
model_success.fit(X_train_resampled, y_train_resampled)

# Predict on the test set
y_pred = model_success.predict(X_test)

# Evaluate performance
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Visualize the confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Flop', 'Success'], yticklabels=['Flop', 'Success'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Movie Success Prediction')
plt.show()
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Step 1: Load your dataset
data = pd.read_csv(r".\\project_root\\data\\finaldataset_merged.csv")

# Step 2: Extract the min and max values from the dataset's 'budget' column
min_budget = data['budget'].min()
max_budget = data['budget'].max()

# Step 3: Sample data for an upcoming movie (user input)

actor_name = input("Enter the actor's name: ")
genre = input("Enter the movie genre: ")
budget_inr = float((input("Enter the budget of the movie (in INR): ")))
director_name = input("Enter the director's name: ")
production_house = input("Enter the production house: ")

sample_data = {
    'actor_name': [actor_name],
    'genre': [genre],
    'budget': [budget_inr],  # Example budget in INR (This is the user input)
    'director': [director_name],
    'production_house': [production_house]
}

# Create a DataFrame for the sample movie
upcoming_movie_df = pd.DataFrame(sample_data)

# Step 4: Normalize the user's input budget (INR)
user_budget_inr = upcoming_movie_df['budget'].values[0]  # Get the user's input budget (INR)

# Apply Min-Max scaling formula to normalize the user's INR budget
normalized_budget = (user_budget_inr - min_budget) / (max_budget - min_budget)

# Step 5: Update the budget in the dataframe with the normalized value
upcoming_movie_df['budget'] = normalized_budget

# Debugging: Check the original and normalized budget values
print(f"Original Budget (INR): {user_budget_inr}")
print(f"Normalized Budget (scaled to 0-1): {normalized_budget}")


# Initialize LabelEncoders for categorical features
genre_encoder = LabelEncoder()

# Fit and transform the categorical features

upcoming_movie_df['genre_encoded'] = genre_encoder.fit_transform(upcoming_movie_df['genre'])
# Add predicted values to the upcoming movie dataframe
upcoming_movie_df['predicted_gross'] = predicted_gross[0]
upcoming_movie_df['predicted_profitability'] = predicted_profitability[0]



# Show the dataframe with all features
print(upcoming_movie_df)

# Select the features for movie success prediction (adding predicted gross and profitability)
X_new = upcoming_movie_df[['genre_encoded', 'budget', 'predicted_gross', 'predicted_profitability']]

# Step 9: Predict the movie success using the trained model
movie_success_prediction = model_success.predict(X_new)

# Output the prediction result
if movie_success_prediction[0] == 1:
    print("The movie is predicted to be a success!")
else:
    print("The movie is predicted to be a flop!")
import pickle
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Assuming your model variables are model_gross, model_profitability, and model_success


with open('moviemodel_success.pkl', 'wb') as success_model_file:
    pickle.dump(model_success, success_model_file)

print("Models saved successfully!")