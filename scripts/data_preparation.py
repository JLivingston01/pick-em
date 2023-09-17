
import pandas as pd
import numpy as np
import os

def make_merge_frame(games:pd.DataFrame) -> pd.DataFrame:
    """Merges player stats from season t-1 onto rosters in season t, aggregates
    to the game level (home and away stats) with game-level scores and the spread.

    Args:
        games (DataFrame): df of games, sliced to the target season

    Returns:
        DataFrame: Game data for season t with aggregated player stats from season t-1
    """
    all_data = os.listdir("data/")
    roster_list = [i for i in all_data if 'roster_' in i]
    stats_list = [i for i in all_data if 'player_stats_' in i]

    all_rosters = pd.DataFrame()
    all_stats = pd.DataFrame()

    for filename in roster_list:
        tmp = pd.read_parquet(f"data/{filename}")
        all_rosters = pd.concat([all_rosters,tmp])

    for filename in stats_list:
        tmp = pd.read_parquet(f"data/{filename}")
        all_stats = pd.concat([all_stats,tmp])

    all_stats['merge_season'] = all_stats['season']+1

    all_rosters['player_display_name'] = all_rosters['full_name'] 
    all_rosters['merge_season'] = all_rosters['season']

    agg_stats = all_stats[all_stats['season_type']=='REG'].groupby(
        ['player_display_name','merge_season']).agg({
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

    roster_stats = all_rosters.merge(agg_stats,on=['player_display_name','merge_season'],how='inner')

    team_stats = roster_stats.groupby(['team','season']).agg({
            i:'sum' for i in agg_stats.columns if i not in ['player_display_name','merge_season']
        }).reset_index()

    team_stats.to_csv("data/seasonal_team_stats.csv",index=False)

    prep_frame = games[['season','week','away_team','home_team','away_score','home_score','result','spread_line']+
                    [i for i in games.columns if ('roof_' in i)|('surface_' in i)|('away_team_id' in i)|('home_team_id' in i)]].copy()

    away_team_stats = team_stats.copy()
    home_team_stats = team_stats.copy()

    away_team_stats.columns = ['away_'+i for i in team_stats.columns]
    home_team_stats.columns = ['home_'+i for i in team_stats.columns]

    home_team_stats['season'] = home_team_stats['home_season']
    away_team_stats['season'] = away_team_stats['away_season']

    home_team_stats.drop(["home_season"],axis=1,inplace=True)
    away_team_stats.drop(["away_season"],axis=1,inplace=True)

    merge = prep_frame.merge(away_team_stats,on=['away_team','season'],how='left').merge(home_team_stats,on=['home_team','season'],how='left')

    return merge

def main() -> None:

    games = pd.read_csv('data/games.csv')
    pd.set_option("display.max_columns",55)

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

    games = games[games['season']>=2018].copy()

    all_dat = make_merge_frame(games=games)

    all_dat.to_csv("data/model_data.csv",index=False)

    return

if __name__=='__main__':
    main()