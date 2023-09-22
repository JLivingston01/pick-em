
import pandas as pd
import logging


def fetch_features_one(home_team,away_team,spread=0,season=2023):

    data = pd.read_csv("data/seasonal_team_stats.csv")

    pred_frame = pd.DataFrame({
        'season':[season],
        'away_team':[away_team],
        'home_team':[home_team],
        'spread_line':[spread],
        })
    
    home_data = data.copy()
    home_data.columns = ['home_'+i for i in data.columns]
    away_data = data.copy()
    away_data.columns = ['away_'+i for i in data.columns]
    home_data.rename({'home_season':'season'},axis=1,inplace=True)
    away_data.rename({'away_season':'season'},axis=1,inplace=True)

    out = pred_frame.merge(away_data,on=['season','away_team'],how='left').merge(home_data,on=['season','home_team'],how='left')

    return out

def main() -> None:

    logging.basicConfig(level=logging.INFO)

    team_stats = pd.read_csv("data/seasonal_team_stats.csv")

    logging.info(f"Team stats: {len(team_stats)}")

    return 

if __name__=='__main__':
    main()