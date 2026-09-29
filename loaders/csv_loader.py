import pandas as pd

def load_csv(filepath):
    df = pd.read_csv(filepath)
    required = ["date", "home_team", "away_team", "home_score", "away_score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return df

def save_game_log(df, filepath):
    df.to_csv(filepath, index=False)
