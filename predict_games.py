from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.static import teams
import pandas as pd
import openai
import os

def get_teams(abrv1,abrv2):
    nba_teams = teams.get_teams()
    team1 = [team for team in nba_teams if team['abbreviation'] == abrv1][0]
    team2 = [team for team in nba_teams if team['abbreviation'] == abrv2][0]
    return team1, team2

