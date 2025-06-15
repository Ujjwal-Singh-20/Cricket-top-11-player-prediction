Cricket Top 11 Predictor

A simple web application that predicts the top 11 cricket players using their historical performance data. It fetches player stats from a cricket API, calculates a composite performance score, and ranks players to select the best performing 11.

---

Features:

- Fetches player data from a cricket API <br>
    (cloned github repo -> Cricket-API)
- Calculates a performance score using key statistics (batting averages, strike rates, wickets, centuries)
- Uses a machine learning model (Random Forest) to predict player performance
- Provides a web interface to input player names and view the top 11 predicted players

---

How to Run

1. Clone the repository.

2. Install the required Python packages:
    <br>
   ```pip install flask requests scikit-learn joblib pandas python-dotenv googlesearch-python beautifulsoup4```

3. Ensure the cricket API server is running locally <br>
    STEPS:<br>
    `cd Cricket_API` <br>
    `python main.py`

4. Run the Flask application: (in a new terminal) <br>
   `python app.py`

5. Open your browser and go to http://localhost:8080.

6. Enter comma-separated player names to get the predicted top 11 players.

---

Notes

- Scoring system: Composite metric based on batting and bowling stats.
- ML model: Trained on historical player data and saved locally.
- Customizable: The player list and scoring weights can be adjusted as needed.

