from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams
import pandas as pd
import openai
import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

app = Flask(__name__)

def get_games(abrv):
    """Get the last 10 games for two teams"""
    nba_teams = teams.get_teams()
    team_info= [team for team in nba_teams if team['abbreviation'] == abrv][0]
    team_id = team_info['id']
    gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id)
    games = gamefinder.get_data_frames()[0]
    team_games = games[(games['SEASON_ID'] == '22024') & (games['TEAM_ID'] == team_id)]
    return team_games

def format_team_stats(team_df):
    "Format team stats for OpenAi"
    wins = team_df['WL'].value_counts()['W']
    losses = team_df['WL'].value_counts()['L']
    average_points = team_df['PTS'].mean()
    win_percentage = (wins/ (wins + losses)) * 100

    stats_summary = f"""
    2025 statistics:
    -Wins and Losses: {wins} - {losses}
    -Average Points scored: {average_points:.1f}
    -Win Percentage: {win_percentage:.1f}%

    Game by Game Results:
    """

    for _, game in team_df.iterrows():
        stats_summary += f"\n{game['GAME_DATE']}: {game['MATCHUP']} - {game['WL']} ({game['PTS']})"

    return stats_summary, team_df['TEAM_NAME']

def get_prediction(team1,team2):
    """Generate a prediction from OpenAI for a game between two teams"""
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')

    team1_stats,team1_name = team1
    team2_stats, team2_name = team2

    prompt = f"""Based on the following NBA statistics, predict the winner and final score of a game between {team1_name[0]} and {team2_name[0]}:
    
    {team1_name[0]} Stats:
    {team1_stats}

    {team2_name[0]} Stats:
    {team2_stats}

    Please provide your prediction in this following format:
    Winner: [team1 name]
    Score: [team2 name][score] - [team2 name][score]
    Confidence: [percentage]
    Brief explanation:[1-2 sentences]

    """

    try:
        response = openai.chat.completions.create(
            model = "gpt-4o-mini",
            messages = [{"role": "system",
                         "content": "You are a professional NBA analyst with expertise in predicting game outcomes based on team statistics"},
                        {"role": "user",
                         "content": prompt}],
                         temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting prediction: {str(e)}"
@app.route('/')
def home():
    # get the list of all NBA teams for the dropdown
    nba_teams = teams.get_teams()
    team_list = sorted([{
        'abbr': team['abbreviation'], 
        'name': team['full_name'], 
        'id': team['id']  # This is needed for the logo URLs
    } for team in nba_teams], key=lambda x: x['name'])
    return render_template('index.html', teams=team_list)
@app.route('/predict', methods=['POST'])
def predict():
    team1_abbr = request.form['team1']
    team2_abbr = request.form['team2']

    if team1_abbr == team2_abbr:
        return jsonify({
            'success': False,
            'error': 'Please select different teams for comparison'
        })

    try:
        games = get_games(team1_abbr)
        games2 = get_games(team2_abbr)

        team1_stats = format_team_stats(games)
        team2_stats = format_team_stats(games2)
        prediction = get_prediction(team1_stats, team2_stats)
        return jsonify({'success': True,'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
 


