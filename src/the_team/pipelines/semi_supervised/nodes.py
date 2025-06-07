"""
This is a boilerplate pipeline 'semi_supervised'
generated using Kedro 0.19.12
"""
from typing import Tuple
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    precision_recall_curve, classification_report,
    roc_auc_score, auc
)
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from logging import getLogger
logger = getLogger(__name__)

def define_weak_positive_criteria(df: pd.DataFrame) -> pd.DataFrame:
    """
    Define business logic for weak positive labeling.

    Args:
        df (pd.DataFrame): Unlabeled input data with features.

    Returns:
        pd.Series: Boolean mask where True indicates a weak positive sample.
    """
    # Find top 10 most frequent product categories
    top_categories = (
        df["product_category_name"]
        .value_counts()
        .head(10)
        .index
        .tolist()
    )

    mask = (
        (df["review_score"] > 3)
        | (df["deli_duration_exp"] <= -7)
        | (df["voucher"] >= 0.3)
        | (df["total_spent"] >= df["total_spent"].quantile(0.8))
        | (df["product_category_name"].isin(top_categories))
    )
    return pd.DataFrame({"weak_positive": mask})

def generate_pseudo_labels(
    model,
    X_unlabeled: pd.DataFrame,
    weak_criteria_df: pd.DataFrame,
    train_columns: list[str],
) -> pd.DataFrame:
    """
    Assign pseudo-labels using model predictions + business criteria for confident positive examples.

    Args:
        model: Trained sklearn-like classifier with predict_proba()
        X_unlabeled (pd.DataFrame): Feature matrix without labels
        weak_criteria (pd.Series): Boolean Series identifying confident examples

    Returns:
        pd.DataFrame: Unlabeled features with pseudo-label column.
    """
    # Dummy-encode
    X_dummies = pd.get_dummies(X_unlabeled)

    # Align to training columns (fills any missing dummies with 0; drops extras)
    X_aligned = X_dummies.reindex(columns=train_columns, fill_value=0)

    # Get model probabilities
    preds = model.predict_proba(X_aligned)[:, 1]

    # Combine with your weak-positive mask
    weak_mask = weak_criteria_df["weak_positive"]
    pseudo_labels = (preds > 0.8) | weak_mask

    return X_unlabeled.assign(is_repeat_buyer=pseudo_labels.astype(int))

def retrain_model_with_pseudo_labels(df: pd.DataFrame) -> Tuple[Pipeline, dict, pd.DataFrame, pd.DataFrame]:
    """
    Retrain logistic model using pseudo-labeled data and return detailed evaluation metrics.

    Args:
        df (pd.DataFrame): Pseudo-labeled dataset

    Returns:
        model, metrics, full_predictions_df, top_10_df
    """
    logger.info("Retraining logistic regression model with pseudo-labeled data.")
    X = df.drop(columns=["is_repeat_buyer"])
    y = df["is_repeat_buyer"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    numeric_features = [
        'deli_duration_exp', 'deli_duration_paid', 'deli_cost', 'item_price', 'total_spent',
        'installment', 'distance_km', 'high_density_customer_area', 'seller_repeat_buyer_rate',
        'review_score', 'product_name_length', 'product_description_length',
        'product_photos_qty', 'product_weight_g', 'product_length_cm',
        'product_height_cm', 'product_width_cm'
    ]
    payment_features = ['credit_card', 'voucher', 'debit_card', 'boleto']
    cat_features = ['product_category_name']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features + payment_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ], remainder='passthrough')

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    y_pred = (y_proba >= best_threshold).astype(int)

    predictions_df = X_test.copy()
    predictions_df["y_true"] = y_test.values
    predictions_df["y_pred"] = y_pred
    predictions_df["y_proba"] = y_proba

    top_k = int(len(y_test) * 0.1)
    top_10_df = predictions_df.sort_values("y_proba", ascending=False).head(top_k)

    metrics = {
        "best_threshold": float(best_threshold),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "prc_auc": auc(recall, precision),
        "top_10_precision": top_10_df["y_true"].mean(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "prc_curve": {
            "precision": precision.tolist(),
            "recall": recall.tolist()
        }
    }
    logger.info("Model retraining complete with pseudo-labels.")
    logger.info(f"Best threshold found: {best_threshold:.4f}")
    logger.info(f"PRC AUC: {metrics['prc_auc']:.4f}")
    logger.info(f"Top 10 precision: {metrics['top_10_precision']:.4f}")
    logger.info(f"Classification report:\n{metrics['classification_report']}")
    return pipeline, metrics, predictions_df, top_10_df

def analyze_ssl_bias(pseudo_labeled_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze pseudo-labeled positive samples to detect over-reliance on certain features like top categories.

    Args:
        pseudo_labeled_df (pd.DataFrame): DataFrame after pseudo-labeling, must include `product_category_name`
            and `is_repeat_buyer` (pseudo or true labels).

    Returns:
        pd.DataFrame: Summary statistics on how many pseudo-labeled positives came from top categories.
    """
    logger.info("Analyzing pseudo-labeled data for potential bias in top categories.")
    # Get top 10 product categories in overall data
    top_categories = (
        pseudo_labeled_df["product_category_name"]
        .value_counts()
        .head(10)
        .index
        .tolist()
    )

    # Filter pseudo-labeled positives 
    positives = pseudo_labeled_df[pseudo_labeled_df["is_repeat_buyer"] == 1]

    # Count how many came from top categories
    top_cat_counts = (
        positives["product_category_name"].isin(top_categories)
    ).sum()

    total_positives = len(positives)
    percent_top_category = top_cat_counts / total_positives if total_positives else 0

    result = pd.DataFrame({
        "total_pseudo_positives": [total_positives],
        "positives_from_top_categories": [top_cat_counts],
        "percentage_top_category": [round(percent_top_category * 100, 2)]
    })
    logger.info(f"Analysis complete: {result.to_dict(orient='records')}")
    logger.info(f"Total pseudo positives: {total_positives}, from top categories: {top_cat_counts}, "
                f"percentage: {percent_top_category:.2%}")
    return result
