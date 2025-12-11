import streamlit as st
import pandas as pd

@st.cache_data
def load_player_stats():
    player_stats = pd.read_csv('all-player-stats.csv')
    player_stats = player_stats.fillna(0)
    team_stats = pd.read_csv('stats-team.csv')
    return player_stats, team_stats



def add_target_variable(player_df, team_df, target_column, player_team_col='Team', team_name_col='Team Name'):
    """
    Merge a target variable from team_stats to player_stats
    
    Parameters:
    - player_df: player_stats dataframe
    - team_df: team_stats dataframe  
    - target_column: name of column in team_df to add as target
    - player_team_col: column name in player_df with team names (default: 'Team')
    - team_name_col: column name in team_df with team names (default: 'Team Name')
    
    Returns:
    - player_df with target variable added
    """
    # Select only Team Name and target column from team_df
    team_subset = team_df[[team_name_col, target_column]].copy()
    
    # Rename target column before merge to avoid conflicts
    target_name = f'{target_column}_target'
    team_subset = team_subset.rename(columns={target_column: target_name})
    
    # Merge - no need for suffixes since we only have 2 columns and renamed the target
    merged_df = player_df.merge(
        team_subset, 
        left_on=player_team_col, 
        right_on=team_name_col, 
        how='left'
    )
    
    # Drop the team name column from team_df (we already have it in player_df as 'Team')
    if team_name_col in merged_df.columns:
        merged_df = merged_df.drop(columns=[team_name_col])
    
    return merged_df

# Example usage (commented out):
# player_stats = add_target_variable(player_stats, team_stats, 'Goal Difference')


