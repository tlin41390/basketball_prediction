import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report 
from sklearn.preprocessing import LabelEncoder 
from nba_api.stats.endpoints import leaguegamefinder 
from nba_api.stats.static import teams 
import shap
from requests.exceptions import ReadTimeout, ConnectionError
import time

#get the nba teams and map their abbreviations to their team ids
nba_teams = teams.get_teams()
team_abbr_to_id = {team['abbreviation']: team['id'] for team in nba_teams}
all_games = pd.DataFrame()

for team in nba_teams:
    team_id = team['id']
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id, timeout=30)
            games = gamefinder.get_data_frames()[0]
            all_games = pd.concat([all_games, games], ignore_index=True)
            break  # Success, exit retry loop
        except (ReadTimeout, ConnectionError) as e:
            if attempt < max_retries - 1:  # If not the last attempt
                print(f"Timeout for team {team['full_name']}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"Failed to fetch data for team {team['full_name']} after {max_retries} attempts.")
                # Continue with other teams instead of failing completely


all_games['GAME_DATE'] = pd.to_datetime(all_games['GAME_DATE'])
all_games['WIN'] = all_games['WL'].apply(lambda x: 1 if x == 'W' else 0)
all_games['PTS'] = all_games['PTS'].astype(float)
all_games['Points_Per_Game']= all_games.groupby('TEAM_ID')['PTS'].transform('mean')

def get_opponent_team_id(matchup, team_abbr_to_id, team_id):
    if '@' in matchup:
        opponent_abbr = matchup.split(' @ ')[-1]
    else:
        opponent_abbr = matchup.split(' vs. ')[-1]
    return team_abbr_to_id.get(opponent_abbr, team_id)
    
all_games['OPPONENT_TEAM_ID'] = all_games.apply(lambda row: get_opponent_team_id(row['MATCHUP'], team_abbr_to_id, row['TEAM_ID']), axis = 1)

all_games['HOME_GAME'] = all_games['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
all_games['LAST_GAME_RESULT'] = all_games.groupby('TEAM_ID')['WIN'].shift(1).fillna(0)

le = LabelEncoder()
all_games['TEAM_ID'] = le.fit_transform(all_games['TEAM_ID'])
all_games['OPPONENT_TEAM_ID'] = le.fit_transform(all_games['OPPONENT_TEAM_ID'])

#split the data
X = all_games[['TEAM_ID','OPPONENT_TEAM_ID', 'Points_Per_Game', 'HOME_GAME','LAST_GAME_RESULT']]
y = all_games['WIN']
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state = 42)

#train model
model = RandomForestClassifier(n_estimators=100, random_state = 42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

feature_importances = pd.DataFrame(model.feature_importances_,
                                   index = X_train.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
print("Feature Importances:\n", feature_importances)

team_abbr = 'BOS'
opponent_abbr = 'CHA'
average_points_per_game = 110.5

new_data = pd.DataFrame({
    'TEAM_ID': [le.transform([team_abbr_to_id[team_abbr]])[0]],
    'OPPONENT_TEAM_ID': [le.transform([team_abbr_to_id[opponent_abbr]])[0]],
    'Points_Per_Game':[average_points_per_game],
    'HOME_GAME': [1],
    'LAST_GAME_RESULT': [1]
})

predictions = model.predict(new_data)
prediction_probabilities = model.predict_proba(new_data)

print("Predictions: ", predictions)
print("Prediction Probabilities: ", prediction_probabilities)
