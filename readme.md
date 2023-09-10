# Pick-em

This is a model for my pick-em tournament. The strategy is:

1. Merge player-level team in season t onto player-level fantasy stats from season t-1
2. Aggregate player-level stats up to team level, so teams in season t have aggregated player stats from season t-1, wherever those players previously played
3. Merge team stats onto home teams and away teams in season t
4. Model the score difference: home score - away score. 

I'm currently using an MLPRegressor, and stopping training when validation stops improving.