import pandas as pd
import os
import streamlit as st

@st.cache_data
def load_data():
    # Load core datasets
    all_matches = pd.read_csv(r'football_bot/data/all_matches.csv')  

    # Load league datasets from the folder
    league_datasets = {}
    league_folder = r'football_bot/data/league_datasets/'
    for file in os.listdir(league_folder):
        if file.endswith('.csv'):
            league_name = file.split('_')[0].lower()  # Convert to lower case for consistency
            league_data = pd.read_csv(os.path.join(league_folder, file))
            league_datasets[league_name] = league_data

    return {
        'all_matches': all_matches,
        'league_datasets': league_datasets
    }

@st.cache_data
def preprocess_data(league_datasets):
    processed_data = {}
    
    # Process Argentina league
    if 'argentina' in league_datasets:
        argentina_data = league_datasets['argentina']
        argentina_data['date_name'] = pd.to_datetime(argentina_data['date_name'], errors='coerce')
        argentina_data['goal_difference'] = argentina_data['local_result'] - argentina_data['visitor_result']
        argentina_data['result'] = argentina_data['goal_difference'].apply(lambda x: 'Local Win' if x > 0 else ('Visitor Win' if x < 0 else 'Draw'))
        processed_data['argentina'] = argentina_data
    else:
        st.warning("Argentina dataset not found!")

    # Process English Premier League
    if 'english' in league_datasets:
        epl_data = league_datasets['english']
        epl_data['Date'] = pd.to_datetime(epl_data['Date'], errors='coerce')
        epl_data['goal_difference'] = epl_data['FTHG'] - epl_data['FTAG']
        epl_data['result'] = epl_data['goal_difference'].apply(lambda x: 'Home Win' if x > 0 else ('Away Win' if x < 0 else 'Draw'))
        processed_data['english'] = epl_data
    else:
        st.warning("English Premier League dataset not found!")

    return processed_data
