import pandas as pd
from sklearn.preprocessing import LabelEncoder

def create_features(league_data, league_type):
    features = pd.DataFrame()

    if league_type == 'argentina':
        home_teams = pd.get_dummies(league_data['local_team'], prefix='argentina_local_team')
        away_teams = pd.get_dummies(league_data['visitor_team'], prefix='argentina_visitor_team')
        features = pd.concat([home_teams, away_teams], axis=1)
        labels = league_data['local_result']

    elif league_type == 'english':
        home_teams = pd.get_dummies(league_data['HomeTeam'], prefix='epl_home_team')
        away_teams = pd.get_dummies(league_data['AwayTeam'], prefix='epl_away_team')
        features = pd.concat([home_teams, away_teams], axis=1)
        labels = league_data['FTR']  # Full Time Result

    # Convert labels to numeric values using LabelEncoder
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    # Ensure labels are in a Series
    labels = pd.Series(labels)

    # Convert all features to numeric, if applicable
    features = features.apply(pd.to_numeric, errors='coerce')

    # Check if features or labels are empty
    if features.empty or labels.empty:
        raise ValueError("Features or labels are empty. Check the input data.")

    return features, labels