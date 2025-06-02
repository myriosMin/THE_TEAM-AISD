import sys
from pathlib import Path

# Add '../src' to Python path so imports work
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from the_team.pipelines.feature_engineering.nodes import add_verified_rating

# Load full dataset and sample 100 rows
df_full = pd.read_csv("data/01_raw/olist_order_reviews_dataset.csv")
df_sample = df_full.sample(n=100, random_state=42)

# Run the Kedro node function
result_df = add_verified_rating(df_sample)

# Print important columns to console
print(result_df[["order_id", "review_score", "sentiment", "verified_rating"]])

for _, row in result_df.iterrows():
    score = row["review_score"]
    sentiment = row["sentiment"]
    comment = str(row["review_comment_message"])[:50]
    
    match = (
        (sentiment == "positive" and score in [4, 5]) or
        (sentiment == "neutral" and score == 3) or
        (sentiment == "negative" and score in [1, 2])
    )
    
    status = "✅" if match else "❌"
    print(f"{status} {score} | {comment}... → {sentiment}")
