import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure the VADER lexicon is downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# Initialize the analyzer
sia = SentimentIntensityAnalyzer()

def get_sentiment_multiplier(text):
    """
    Analyzes text using VADER. 
    Returns a multiplier: >1.0 for positive sentiment, <1.0 for negative.
    """
    if not isinstance(text, str) or text.strip() == "":
        return 1.0 # Neutral multiplier if no text
        
    # Get the compound score (-1.0 to 1.0)
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    
    # Convert compound score to a gentle multiplier (e.g., 0.9 to 1.1)
    # This ensures a terrible review slightly hurts the SVD rating, 
    # while a glowing review gives it a slight boost.
    multiplier = 1.0 + (compound * 0.1) 
    
    return multiplier