"""OCR parser — extract game data from betting app screenshots."""
import re
from PIL import Image
import pandas as pd
from datetime import datetime

def extract_text_from_image(image_file) -> str:
    """Extract text from uploaded image using EasyOCR."""
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False)
        img = Image.open(image_file)
        results = reader.readtext(img)
        # Get text with positions for better parsing
        lines = []
        for r in results:
            lines.append({"text": r[1], "bbox": r[0]})
        return lines
    except Exception as e:
        try:
            import pytesseract
            img = Image.open(image_file)
            return [{"text": pytesseract.image_to_string(img), "bbox": None}]
        except:
            return []

def parse_betting_app_screenshot(lines) -> list:
    """
    Parse betting app screenshot format:
    - Date line: "Sep 30, 07:30 League Name - Sub League"
    - Team lines: "Team A..." and "Team B..."
    - Odds: two decimal numbers side by side (e.g., 1.17 and 4.40)
    """
    games = []
    current_date = None
    current_league = None
    current_teams = []
    current_odds = []
    
    date_pattern = r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{1,2}:\d{2})'
    odds_pattern = r'^(\d+\.\d{2})$'
    team_skip = {'winner', '1st', 'half', 'o/u', 'home', 'matches', 'outrights', 'time', 'league', 'odds', 'sort', 'live', 'betting', 'all'}
    
    for line_info in lines:
        text = line_info["text"].strip()
        if not text:
            continue
        
        # Check for date line
        date_match = re.search(date_pattern, text, re.IGNORECASE)
        if date_match:
            # Save previous game if exists
            if len(current_teams) >= 2:
                games.append({
                    "date": current_date,
                    "league": current_league,
                    "home_team": current_teams[0],
                    "away_team": current_teams[1],
                    "home_odds": float(current_odds[0]) if len(current_odds) >= 1 else 1.90,
                    "away_odds": float(current_odds[1]) if len(current_odds) >= 2 else 1.90,
                })
            
            current_date = date_match.group(1)
            # Extract league from rest of line
            rest = text[date_match.end():].strip()
            if rest:
                current_league = rest
            current_teams = []
            current_odds = []
            continue
        
        # Check for odds (standalone decimal numbers)
        odds_match = re.match(odds_pattern, text)
        if odds_match:
            val = float(text)
            if 1.01 <= val <= 100.0:
                current_odds.append(text)
                continue
        
        # Check for team names (skip short words and headers)
        if len(text) >= 3 and text.lower() not in team_skip:
            # Skip if it looks like a number or date
            if not re.match(r'^\d', text):
                # Clean up team name (remove trailing dots)
                clean_name = text.rstrip('.').strip()
                if len(clean_name) >= 3:
                    current_teams.append(clean_name)
    
    # Save last game
    if len(current_teams) >= 2:
        games.append({
            "date": current_date,
            "league": current_league,
            "home_team": current_teams[0],
            "away_team": current_teams[1],
            "home_odds": float(current_odds[0]) if len(current_odds) >= 1 else 1.90,
            "away_odds": float(current_odds[1]) if len(current_odds) >= 2 else 1.90,
        })
    
    return games

def process_image_to_dataframe(image_file) -> pd.DataFrame:
    """Full pipeline: image -> OCR -> parsed games -> DataFrame."""
    lines = extract_text_from_image(image_file)
    
    if not lines:
        return pd.DataFrame()
    
    games = parse_betting_app_screenshot(lines)
    
    if not games:
        return pd.DataFrame()
    
    df = pd.DataFrame(games)
    
    # Add required columns
    if "home_score" not in df.columns:
        df["home_score"] = 0
    if "away_score" not in df.columns:
        df["away_score"] = 0
    if "home_h1_score" not in df.columns:
        df["home_h1_score"] = 0
    if "away_h1_score" not in df.columns:
        df["away_h1_score"] = 0
    if "ft_line" not in df.columns:
        df["ft_line"] = None
    
    return df
