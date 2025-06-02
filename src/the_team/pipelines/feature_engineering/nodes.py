"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.19.12
"""

import pandas as pd
from pysentimiento import create_analyzer

# calling model at module level to ensure it is called once per run
analyzer = create_analyzer(task="sentiment", lang="pt") # using a portugese model to directly analyze portugese

def map_label(label: str) -> str:
    """Map model label to sentiment category."""
    return {
        "LABEL_0": "negative",
        "LABEL_1": "neutral",
        "LABEL_2": "positive"
    }.get(label, "neutral")  # default to neutral if unknown, outputs/labels are given in documentation

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

    # Model predictions
    sentiments = texts.apply(lambda t: map_label(analyzer.predict(t).output))
    reviews["sentiment"] = sentiments

     # Compute verified_rating
    def is_verified(row):
        if pd.isna(row["review_comment_message"]):
            return False
        sentiment = row["sentiment"]
        score = row["review_score"]
        if sentiment == "positive":
            return score in [4, 5]
        elif sentiment == "neutral":
            return score == 3
        elif sentiment == "negative":
            return score in [1, 2]
        return False

    reviews["verified_rating"] = reviews.apply(is_verified, axis=1)
    return reviews

