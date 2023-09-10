
import pandas as pd
import numpy as np


def make_merge_frame(target_season_num:int,games:pd.DataFrame) -> pd.DataFrame:
    """Merges player stats from season t-1 onto rosters in season t, aggregates
    to the game level (home and away stats) with game-level scores and the spread.

    Args:
        target_season_num (int): year of the target season
        games (DataFrame): df of games, sliced to the target season

    Returns:
        DataFrame: Game data for season t with aggregated player stats from season t-1
    """
    rosters = pd.read_parquet(f"data/roster_{target_season_num}.parquet")
    stats = pd.read_parquet(f"data/player_stats_{target_season_num-1}.parquet")
    rosters['player_display_name'] = rosters['full_name'] 
    target_season = games[games['season']==target_season_num].copy()#[(games['season']==2023)&(games['week']==1)]

    agg_stats = stats[stats['season_type']=='REG'].groupby(['player_display_name']).agg({
        'completions':'mean',
        'attempts':'mean',
        'passing_yards':'mean',
        'passing_tds':'mean',
        'interceptions':'mean',
        'sacks':'mean',
        'sack_yards':'mean',
        'sack_fumbles':'mean',
        'sack_fumbles_lost':'mean',
        'passing_air_yards':'mean',
        'passing_yards_after_catch':'mean',
        'passing_first_downs':'mean',
        'passing_epa':'mean',
        'passing_2pt_conversions':'mean',
        'pacr':'mean',
        'carries':'mean',
        'rushing_yards':'mean',
        'rushing_tds':'mean',
        'rushing_fumbles':'mean',
        'rushing_fumbles_lost':'mean',
        'rushing_first_downs':'mean',
        'rushing_epa':'mean',
        'rushing_2pt_conversions':'mean',
        'receptions':'mean',
        'targets':'mean',
        'receiving_yards':'mean',
        'receiving_tds':'mean',
        'receiving_fumbles':'mean',
        'receiving_fumbles_lost':'mean',
        'receiving_air_yards':'mean',
        'receiving_yards_after_catch':'mean',
        'receiving_first_downs':'mean',
        'receiving_epa':'mean',
        'receiving_2pt_conversions':'mean',
        'racr':'mean',
        'target_share':'mean',
        'air_yards_share':'mean',
        'wopr':'mean',
        'special_teams_tds':'mean',
        'fantasy_points':'mean',
        'fantasy_points_ppr':'mean',
        'week':'count'
    }).reset_index()
    agg_stats.rename({'week':'games'},axis=1,inplace=True)
    agg_stats = pd.DataFrame(np.where(agg_stats==0,np.nan,agg_stats),columns=agg_stats.columns)
    roster_stats = rosters.merge(agg_stats,on=['player_display_name'],how='inner')
    team_stats = roster_stats.groupby(['team']).agg({
        i:'sum' for i in agg_stats.columns if i not in ['player_display_name']
    }).reset_index()
    prep_frame = target_season[['week','away_team','home_team','away_score','home_score','result','spread_line']+
                               [i for i in target_season.columns if ('roof_' in i)|('surface_' in i)|('away_team_id' in i)|('home_team_id' in i)]].copy()
    away_team_stats = team_stats.copy()
    home_team_stats = team_stats.copy()

    away_team_stats.columns = ['away_'+i for i in team_stats.columns]
    home_team_stats.columns = ['home_'+i for i in team_stats.columns]

    merge = prep_frame.merge(away_team_stats,on='away_team',how='left').merge(home_team_stats,on='home_team',how='left')

    merge['season'] = target_season_num
    return merge

def main() -> None:

    games = pd.read_csv('data/games.csv')

    games['surface']=games['surface'].str.replace(" ","")
    games['roof']=games['roof'].str.replace(" ","")
    away_surface_roof = games.groupby(['season','home_team']).agg({
        'surface':'first',
        'roof':'first'
    }).reset_index()
    away_surface_roof.columns = ['season','away_team','away_surface','away_roof']
    games = games.merge(away_surface_roof,on=['season','away_team'],how='left')

    roof = pd.get_dummies(games['roof'],drop_first=True,dtype=int,prefix='roof')
    surface = pd.get_dummies(games['surface'],drop_first=True,dtype=int,prefix='surface')
    away_roof = pd.get_dummies(games['away_roof'],drop_first=True,dtype=int,prefix='away_roof')
    away_surface = pd.get_dummies(games['away_surface'],drop_first=True,dtype=int,prefix='away_surface')

    games[roof.columns]=roof
    games[surface.columns]=surface
    games[away_roof.columns]=away_roof
    games[away_surface.columns]=away_surface

    merges = [make_merge_frame(target_season_num=i,games=games) for i in [2023,2022,2021,2020,2019,2018]]

    all_dat = pd.concat(merges)

    all_dat.to_csv("data/model_data.csv",index=False)

    return

if __name__=='__main__':
    main()