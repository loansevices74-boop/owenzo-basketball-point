"""Enhanced OCR parser for dark betting app screenshots — no manual entry needed."""
import re
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pandas as pd
import numpy as np

def preprocess_image(image_file):
    """Advanced preprocessing for dark betting app screenshots."""
    img = Image.open(image_file)
    
    # Convert to RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize if too small
    width, height = img.size
    if width < 1000:
        img = img.resize((width * 2, height * 2), Image.LANCZOS)
    
    # Enhance contrast significantly
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(3.0)
    
    # Enhance brightness
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.5)
    
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(3.0)
    
    return img

def extract_text_from_image(image_file) -> list:
    """Extract text with multiple OCR attempts for maximum accuracy."""
    try:
        import pytesseract
        
        # Preprocess image
        img_processed = preprocess_image(image_file)
        
        # Try multiple OCR configurations
        configs = [
            r'--oem 3 --psm 6',
            r'--oem 3 --psm 11',
            r'--oem 3 --psm 4',
        ]
        
        all_lines = []
        for config in configs:
            try:
                text = pytesseract.image_to_string(img_processed, config=config)
                lines = [{"text": line.strip(), "bbox": None} for line in text.split('\n') if line.strip()]
                all_lines.extend(lines)
            except:
                pass
        
        # Remove duplicates while preserving order
        seen = set()
        unique_lines = []
        for line in all_lines:
            if line["text"] not in seen:
                seen.add(line["text"])
                unique_lines.append(line)
        
        return unique_lines
    except Exception as e:
        print(f"OCR Error: {e}")
        return []

def parse_betting_app_screenshot(lines) -> list:
    """Parse betting app screenshot - optimized for dark theme with green odds."""
    games = []
    current_date = None
    current_league = None
    current_teams = []
    current_odds = []
    
    # Date pattern: "Sep 30, 10:30" or "Sep 30 10:30"
    date_pattern = r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{1,2}:\d{2})'
    
    # Odds pattern: decimal numbers like 1.51, 2.50, 11.0
    odds_pattern = r'^(\d+\.\d{1,2})$'
    
    # Words to skip
    team_skip = {'winner', '1st', 'half', 'o/u', 'home', 'matches', 'outrights', 'time', 'league', 
                 'odds', 'sort', 'live', 'betting', 'all', 'ftv', 'load', 'code', 'share'}
    
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
        
        # Check for team names
        if len(text) >= 2 and text.lower() not in team_skip:
            # Skip if it's just numbers
            if not re.match(r'^\d+$', text):
                # Clean up team name (remove trailing dots/ellipsis)
                clean_name = text.rstrip('.').rstrip('…').strip()
                if len(clean_name) >= 2:
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