import pandas as pd

def predict_winner(model, local_team, visitor_team, neutral):
    # Create a DataFrame for the input match
    input_data = pd.DataFrame({
        'local_team': [local_team],
        'visitor_team': [visitor_team],
        'neutral': [1 if neutral else 0]
    })
    
    # One-hot encode the teams as per the training data
    local_teams = pd.get_dummies(input_data['local_team'], prefix='local_team')
    visitor_teams = pd.get_dummies(input_data['visitor_team'], prefix='visitor_team')

    # Combine the features
    input_features = pd.concat([local_teams, visitor_teams, input_data[['neutral']]], axis=1)
    
    # Align input features with model training features
    all_columns = model.feature_names_in_  # Ensure model has this attribute
    input_features = input_features.reindex(columns=all_columns, fill_value=0)
    
    # Ensure the columns are of string type
    input_features.columns = input_features.columns.astype(str)

    # Predict outcome probabilities
    probabilities = model.predict_proba(input_features)

    # Extract probabilities
    prob_home_win = probabilities[0][0]  # Probability of home win
    prob_draw = probabilities[0][1]       # Probability of draw
    prob_away_win = probabilities[0][2]   # Probability of away win

    # Create a dictionary to hold the results
    result = {
        "prob_home_win": prob_home_win,
        "prob_draw": prob_draw,
        "prob_away_win": prob_away_win
    }

    # Assuming you have a function to calculate possible score outcomes
    score_outcomes = calculate_possible_scores(local_team, visitor_team)  # Ensure this function exists
    result["score_outcomes"] = score_outcomes

    return result

def calculate_possible_scores(local_team, visitor_team):
    # This function should return a list of tuples with score and its probability
    # For example, return [(score, probability), ...]
    # You will need to implement the logic based on your dataset and how you define score probabilities.
    possible_scores = [
        ((1, 0), 0.25),  # Example: score (1-0) with a probability of 25%
        ((2, 1), 0.15),  # Example: score (2-1) with a probability of 15%
        # Add more scores and their probabilities based on your model
    ]
    return possible_scores
