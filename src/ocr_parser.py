"""OCR parser — extract game data from PNG/JPG screenshots."""
import re
import io
from PIL import Image
import pandas as pd

def extract_text_from_image(image_file) -> str:
    """Extract text from uploaded image using EasyOCR (better for sports screenshots)."""
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False)
        img = Image.open(image_file)
        results = reader.readtext(img)
        text = " ".join([r[1] for r in results])
        return text
    except Exception as e:
        # Fallback to pytesseract
        try:
            import pytesseract
            img = Image.open(image_file)
            return pytesseract.image_to_string(img)
        except Exception as e2:
            return ""

def parse_games_from_text(text: str) -> list:
    """
    Parse game data from OCR text.
    Looks for patterns like: TeamA vs TeamB, scores, odds
    """
    games = []
    
    # Pattern 1: "TeamA vs TeamB" or "TeamA - TeamB"
    matchup_pattern = r'([A-Z][a-zA-Z\s\.\']+)\s+(?:vs|v\.?|vs\.?|-|—)\s+([A-Z][a-zA-Z\s\.\']+)'
    
    # Pattern 2: Scores like "102-98" or "102 : 98" or "102 98"
    score_pattern = r'(\d{2,3})\s*[-:]\s*(\d{2,3})'
    
    # Pattern 3: Odds like "1.85" or "2.10"
    odds_pattern = r'(\d+\.\d{2})'
    
    # Find all matchups
    matchups = re.findall(matchup_pattern, text)
    
    # Find all scores
    scores = re.findall(score_pattern, text)
    
    # Find all odds
    odds = re.findall(odds_pattern, text)
    
    # Combine into games
    for i, (home, away) in enumerate(matchups):
        home = home.strip()
        away = away.strip()
        
        # Skip if too short (likely not a team name)
        if len(home) < 3 or len(away) < 3:
            continue
        
        # Skip common false positives
        skip_words = ['the', 'and', 'for', 'basketball', 'league', 'points', 'total', 'over', 'under']
        if home.lower() in skip_words or away.lower() in skip_words:
            continue
        
        game = {
            "home_team": home,
            "away_team": away,
            "home_score": 0,
            "away_score": 0,
            "odds": 1.90,
        }
        
        # Attach scores if available
        if i < len(scores):
            try:
                game["home_score"] = int(scores[i][0])
                game["away_score"] = int(scores[i][1])
            except:
                pass
        
        # Attach odds if available
        if i < len(odds):
            try:
                odds_val = float(odds[i])
                if 1.01 <= odds_val <= 50.0:
                    game["odds"] = odds_val
            except:
                pass
        
        games.append(game)
    
    return games

def process_image_to_dataframe(image_file) -> pd.DataFrame:
    """Full pipeline: image -> OCR -> parsed games -> DataFrame."""
    text = extract_text_from_image(image_file)
    
    if not text.strip():
        return pd.DataFrame()
    
    games = parse_games_from_text(text)
    
    if not games:
        return pd.DataFrame()
    
    df = pd.DataFrame(games)
    
    # Add required columns if missing
    if "date" not in df.columns:
        from datetime import datetime
        df["date"] = datetime.now().strftime("%Y-%m-%d")
    
    if "home_h1_score" not in df.columns:
        df["home_h1_score"] = 0
    if "away_h1_score" not in df.columns:
        df["away_h1_score"] = 0
    if "ft_line" not in df.columns:
        df["ft_line"] = None
    
    return df
