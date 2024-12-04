document.addEventListener('DOMContentLoaded', () => {
    const predictWinBtn = document.getElementById('predictWinBtn');
    const movieForm = document.getElementById('movieForm');
    const responseDiv = document.getElementById('response');

    // Show the form when "Predict Win" button is clicked
    predictWinBtn.addEventListener('click', () => {
        movieForm.style.display = 'block';
    });

    // Handle form submission
    movieForm.addEventListener('submit', async (event) => {
        event.preventDefault(); // Prevent page reload

        // Gather form data
        const formData = {
            actor_name: document.getElementById('actor_name').value,
            genre: document.getElementById('genre').value,
            budget: document.getElementById('budget').value,
            director: document.getElementById('director').value,
            production_house: document.getElementById('production_house').value,
        };

        // Validate form data
        if (!formData.actor_name || !formData.genre || !formData.budget || !formData.director || !formData.production_house) {
            responseDiv.style.display = 'block';
            responseDiv.textContent = 'Error: All fields are required.';
            return;
        }

        if (isNaN(formData.budget) || formData.budget <= 0) {
            responseDiv.style.display = 'block';
            responseDiv.textContent = 'Error: Budget must be a positive number.';
            return;
        }

        try {
            // Send JSON data to Flask backend
            const response = await fetch('/predict_win', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData),
            });

            const data = await response.json();
            responseDiv.style.display = 'block';
            responseDiv.textContent = `Backend Response: ${data.message || data.prediction}`;
        } catch (error) {
            console.error('Error:', error);
            responseDiv.style.display = 'block';
            responseDiv.textContent = 'Error: Unable to process the request.';
        }
    });
});
