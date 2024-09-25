from locust import HttpUser, task, between

class StreamlitUser(HttpUser):
    wait_time = between(1, 5)  # Wait time between tasks
    host = "http://localhost:8501"  # Change to the actual host and port of your Streamlit app

    @task
    def view_homepage(self):
        self.client.get("/")  # Adjust the URL as needed

    @task
    def select_league_and_predict(self):
        self.client.get("/?selected_league=English%20Premier%20League")  # Select a league (URL parameters may vary)
        self.client.post("/predict", json={"home_team": "Team A", "away_team": "Team B", "neutral": False})  # Simulate a prediction
