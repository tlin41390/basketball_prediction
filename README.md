# NBA Game Predictor

An AI-powered NBA game prediction system that combines real-time NBA statistics with machine learning to forecast game outcomes.

## 🏀 Features

- Real-time NBA team and player statistics using the NBA API
- Game prediction using OpenAI's GPT models
- Team performance analysis including:
  - Win/Loss records
  - Average points per game
  - Recent game results
  - Team performance trends

## 📋 Prerequisites

- Python 3.10+
- NBA API access
- OpenAI API key

## 🛠️ Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd nba-game-predictor
```

2. Install required packages:
```bash
pip install nba_api pandas openai python-dotenv
```

3. Set up your environment variables:
Create a `.env` file in the root directory and add:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

## 💻 Usage

1. Import the required libraries and initialize the API:
```python
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams
import pandas as pd
import openai
```

2. Get team statistics:
```python
# Example for getting Nuggets games
nuggets = [team for team in teams.get_teams() if team['abbreviation'] == 'DEN'][0]
nuggets_id = nuggets['id']
gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=nuggets_id)
games = gamefinder.get_data_frames()[0]
```

3. Generate predictions:
```python
prediction = get_prediction(team1_stats, team2_stats, "Team1 Name", "Team2 Name")
print(prediction)
```

## 📊 Sample Output

```
Game Prediction:
Winner: Denver Nuggets
Score: Denver Nuggets 115 - Golden State Warriors 108
Confidence: 75%
Brief Explanation: The Nuggets' superior recent form and higher scoring average 
suggests they'll maintain their momentum against the Warriors.
```

## 🔄 Current Limitations

- Predictions don't account for player injuries/availability
- Limited to current season statistics
- Weather and travel factors not considered

## 🚀 Future Enhancements

- [ ] Add player injury tracking
- [ ] Implement historical head-to-head analysis
- [ ] Add prediction accuracy tracking
- [ ] Include player-specific statistics
- [ ] Develop web interface for predictions

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check [issues page](your-issues-url).

## 👥 Authors

- Your Name
  - GitHub: [@yourusername](your-github-url)

## 🙏 Acknowledgments

- NBA API for providing access to NBA statistics
- OpenAI for their GPT models