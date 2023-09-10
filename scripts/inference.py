import pandas as pd
from joblib import load
import sys

def main()->None:

    week = sys.argv[1]
    print(week)
    data = pd.read_csv("data/model_data.csv")
    this_season = data[data['season']==2023].copy()

    features = [
        i for i in data.columns if i not in ['week','away_team','home_team','away_score','home_score','result','season',]
    ]

    model = load('artifacts/model.joblib')

    this_seasonX = this_season[features].copy()
    this_season_prediction = pd.Series(model.predict(this_seasonX.fillna(0)),this_seasonX.index)
    this_season['PRED'] = this_season_prediction
    this_season[this_season['week']==int(week)][['week','away_team','home_team','spread_line','PRED']].to_csv(
        f"data/predictions/inference_week_{week}.csv",index=False
    )
    return

if __name__=='__main__':
    main()