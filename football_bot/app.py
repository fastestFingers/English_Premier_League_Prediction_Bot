import pandas as pd
import os
import streamlit as st
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import lightgbm as lgb
from bot.data_loader import load_data, preprocess_data
from bot.feature_engineering import create_features

# Global variable to hold the last retraining time and model
last_retrain_time = None
model = None

# Function to check if retraining is needed
def needs_retraining():
    global last_retrain_time
    if last_retrain_time is None:
        return True  # If we have never retrained, we need to retrain
    return datetime.now() - last_retrain_time > timedelta(hours=1)  # Retrain every hour

# Function to retrain the model
def retrain_model():
    global last_retrain_time, model
    data = load_data()
    league_datasets = data['league_datasets']
    processed_data = preprocess_data(league_datasets)

    features_arg, labels_arg = create_features(processed_data['argentina'], league_type='argentina')
    features_epl, labels_epl = create_features(processed_data['english'], league_type='english')

    features = pd.concat([features_arg, features_epl], ignore_index=True)
    labels = pd.concat([labels_arg, labels_epl], ignore_index=True)

    model, accuracy, report = train_model(features, labels)
    last_retrain_time = datetime.now()  # Update last retrain time

    return model, accuracy, report

# Train model function
def train_model(features, labels):
    if features.empty or labels.empty:
        raise ValueError("Features or labels are empty.")

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define models
    xgb_model = xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, random_state=42)
    lgb_model = lgb.LGBMClassifier(random_state=42)

    # Create an ensemble model using VotingClassifier
    ensemble_model = VotingClassifier(
        estimators=[('xgboost', xgb_model), ('lightgbm', lgb_model)],
        voting='soft'
    )

    # Perform cross-validation
    cv_scores = cross_val_score(ensemble_model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-Validation Accuracy: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")

    # Train the ensemble model
    ensemble_model.fit(X_train, y_train)

    # Evaluate the model
    accuracy = ensemble_model.score(X_test, y_test)
    y_pred = ensemble_model.predict(X_test)
    report = classification_report(y_test, y_pred)

    return ensemble_model, accuracy, report

# Function to predict winner
def predict_winner(model, home_team, away_team, neutral):
    # Create a DataFrame for the input match
    input_data = pd.DataFrame({
        'local_team': [home_team],
        'visitor_team': [away_team],
        'neutral': [1 if neutral else 0]
    })

    # One-hot encode the teams as per the training data
    local_teams = pd.get_dummies(input_data['local_team'], prefix='local_team')
    visitor_teams = pd.get_dummies(input_data['visitor_team'], prefix='visitor_team')

    # Combine the features
    input_features = pd.concat([local_teams, visitor_teams, input_data[['neutral']]], axis=1)

    # Align input features with model training features
    all_columns = model.feature_names_in_
    input_features = input_features.reindex(columns=all_columns, fill_value=0)
    input_features.columns = input_features.columns.astype(str)

    # Predict outcome probabilities
    probabilities = model.predict_proba(input_features)

    # Extract probabilities
    prob_home_win = probabilities[0][0]  # Probability of home win
    prob_draw = probabilities[0][1]      # Probability of draw
    prob_away_win = probabilities[0][2]  # Probability of away win

    result = {
        "prob_home_win": prob_home_win,
        "prob_draw": prob_draw,
        "prob_away_win": prob_away_win
    }

    score_outcomes = calculate_possible_scores(home_team, away_team)  # Implement this
    result["score_outcomes"] = score_outcomes

    return result

# Example score calculation function
def calculate_possible_scores(home_team, away_team):
    possible_scores = [
        ((1, 0), 0.25),  # Example: score (1-0) with a probability of 25%
        ((2, 1), 0.15),  # Example: score (2-1) with a probability of 15%
    ]
    return possible_scores

# Streamlit app configuration
st.set_page_config(page_title="Football Match Predictor", page_icon="⚽", layout="wide")

# Main function for the app
def main():
    global model
    predictions = []

    if needs_retraining():
        model, accuracy, report = retrain_model()
        st.success("Model retrained successfully!")

    if model:
        st.markdown(f"<p style='text-align: center;'>Model Accuracy: <strong>{accuracy:.2%}</strong></p>", unsafe_allow_html=True)

    data = load_data()
    league_datasets = data['league_datasets']
    processed_data = preprocess_data(league_datasets)

    # League teams
    league_teams = {}
    if 'argentina' in processed_data:
        league_teams['Argentina'] = get_league_teams(processed_data['argentina'])
    if 'english' in processed_data:
        league_teams['English Premier League'] = get_league_teams(processed_data['english'])

    selected_league = st.selectbox('Select League', list(league_teams.keys()))

    if selected_league:
        teams = league_teams[selected_league]
        col1, col2 = st.columns(2)
        with col1:
            home_team = st.selectbox('Select Home Team', teams)
        with col2:
            away_team = st.selectbox('Select Away Team', teams)

        neutral = st.checkbox('Neutral Venue')

        if st.button('⚡ Predict Winner'):
            if not home_team or not away_team:
                st.error("Please select both teams.")
            else:
                result = predict_winner(model, home_team, away_team, neutral)

                st.write(f"**Probability of Home Win:** {result['prob_home_win']:.2%}")
                st.write(f"**Probability of Draw:** {result['prob_draw']:.2%}")
                st.write(f"**Probability of Away Win:** {result['prob_away_win']:.2%}")

                expected_winner = "Draw"
                if result['prob_home_win'] > result['prob_away_win']:
                    expected_winner = home_team
                elif result['prob_away_win'] > result['prob_home_win']:
                    expected_winner = away_team
                st.write(f"**Expected Winner:** {expected_winner}")

                # Display possible score outcomes
                st.write("**Possible Score Outcomes:**")
                for score, score_prob in result["score_outcomes"]:
                    st.write(f"Score: {score} - Probability: **{score_prob:.2%}**")

                predictions.append(f"{home_team} vs {away_team}: Home Win Probability: {result['prob_home_win']:.2%}, Draw Probability: {result['prob_draw']:.2%}, Away Win Probability: {result['prob_away_win']:.2%}, Expected Winner: {expected_winner}")

    if st.button("Show Prediction History"):
        display_prediction_history(predictions)

# Run the app
if __name__ == "__main__":
    main()
