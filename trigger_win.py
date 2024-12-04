import random
import subprocess

# Define possible values
genres = ["Action", "Comedy", "Drama", "Horror", "Sci-Fi"]
actors = ["Actor A", "Actor B", "Actor C", "Actor D"]
directors = ["Director X", "Director Y", "Director Z"]
production_houses = ["Studio 1", "Studio 2", "Studio 3"]

# Generate random values
random_payload = {
    "actor_name": random.choice(actors),
    "genre": random.choice(genres),
    "budget": random.randint(1000000, 100000000),  # Random budget between 1M and 100M
    "director": random.choice(directors),
    "production_house": random.choice(production_houses)
}

# Convert payload to JSON
import json
json_payload = json.dumps(random_payload)

# Construct the curl command
curl_command = f"""
curl -X POST http://127.0.0.1:5000/predict_win \
-H "Content-Type: application/json" \
-d '{json_payload}'
"""

# Print or execute the curl command
print("Generated curl command:")
print(curl_command)

# Optionally, execute the command
subprocess.run(curl_command, shell=True)
