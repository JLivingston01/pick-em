# Pick-em

This is a model for my pick-em tournament. The strategy is:

1. Merge player-level team in season t onto player-level fantasy stats from season t-1
2. Aggregate player-level stats up to team level, so teams in season t have aggregated player stats from season t-1, wherever those players previously played
3. Merge team stats onto home teams and away teams in season t
4. Model the score difference of each game: home score - away score. 

I'm currently using a perceptron, and stopping training when validation stops improving.

The 2023 season is currently in progress. I've hardcoded all training to look at pre-2023, and inference to only look at 2023. In the future I'll parameterize the year as a sys-arg, as has been done for the inference week (see the inference section).

## How to run

### Setup:
```
python -m venv .venv
pip install --upgrade pip
pip install -r requirements.txt
source .venv/<scripts|bin>/activate

mkdir data
```

### Data prep:
Ensure the necessary data is in a top level data/ directory. 
Rosters: https://github.com/nflverse/nflverse-data/releases/tag/rosters
Player Stats: https://github.com/nflverse/nflverse-data/releases/tag/player_stats
Games: https://github.com/nflverse/nfldata/blob/master/data/games.csv

I use rosters from 2018 to 2023 presently, with stats from 2017 to 2022. 

```
python scripts/data_preparation.py
```

Credit to members of https://github.com/nflverse for their data work:
https://github.com/guga31bb
https://github.com/mrcaseb
https://github.com/tanho63


