"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.19.12
"""

import pandas as pd
from pysentimiento import create_analyzer


#  Item/Review Related Nodes
#######################################################################################################################################


# calling model at module level to ensure it is called once per run
analyzer = create_analyzer(task="sentiment", lang="pt") # using a portugese model to directly analyze portugese

def map_sentiment(sentiment_label: str) -> str:
    return {
        "POS": "positive",
        "NEU": "neutral",
        "NEG": "negative"
    }.get(sentiment_label, "neutral")

def add_verified_rating(reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 'sentiment' and 'verified_rating' features using a Portuguese BERT model.

    Rows without review_comment_message are marked verified_rating=False.

    Args:
        df: DataFrame with 'review_comment_message' and 'review_score'

    Returns:
        DataFrame with 'sentiment' and 'verified_rating'
    """
    reviews = reviews.copy()

    # Ensure inputs are str to ensure model does not choke
    texts = reviews["review_comment_message"].fillna("").astype(str)

    # Updated sentiment analysis to include more sensitivity to short and positive reviews   
    def heuristic_sentiment(text: str, model_sentiment: str) -> str:
        text_lower = text.strip().lower()
        words = text_lower.split()

        # Reclassify as positive if short but clearly affirmative
        positive_keywords = {"ótimo", "excelente", "bom", "recomendo", "confiável", "satisfeito", "tranquilo", "correto", "tudo certo", "tudo ok"} # General words translated
        if model_sentiment == "neutral":
            for word in positive_keywords:
                if word in text_lower:
                    return "positive"

            # Short positive-sounding sentences
            if len(words) <= 5 and any(w in text_lower for w in ["tudo certo", "tudo ok", "site confiável", "cumpriu", "conforme"]): # General words translated
                return "positive"

        return model_sentiment
    
    # Main logic for analysis
    def safe_sentiment(text):
        if not text.strip():
            return None
        raw = analyzer.predict(text)
        return heuristic_sentiment(text, map_sentiment(raw.output))

    # Model predicitons
    reviews["sentiment"] = texts.apply(safe_sentiment)

    def is_verified(row):
        sentiment = row["sentiment"]
        score = row["review_score"]

        if sentiment is None:
            return False  # No comment = no verification

        if sentiment == "positive":
            return score in [4, 5]
        elif sentiment == "neutral":
            return score == 3
        elif sentiment == "negative":
            return score in [1, 2]
        return False

    reviews["is_verified"] = reviews.apply(is_verified, axis=1)
    return reviews


