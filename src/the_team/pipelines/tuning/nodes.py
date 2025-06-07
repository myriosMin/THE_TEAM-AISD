"""
This is a boilerplate pipeline 'tuning'
generated using Kedro 0.19.12
"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import precision_recall_curve, roc_auc_score, classification_report, auc
import pandas as pd
import numpy as np
from logging import getLogger
logger = getLogger(__name__)

def tune_logistic_model(
    X_train: pd.DataFrame,            
    X_test: pd.DataFrame,             
    y_train: pd.Series,            
    y_test: pd.Series,   
    tuning_params: dict,
) -> tuple:
    """
    Tune a logistic regression model using GridSearchCV and find the best threshold based on F1 score.
    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training target variable.
        y_test (pd.Series): Testing target variable.
        tuning_params (dict): Parameters for tuning the logistic regression model.
    Returns:
        tuple: Best model, metrics dictionary, predictions DataFrame, and top 10 predictions DataFrame.
    """
    logger.info("Starting logistic regression model tuning.")
    pipeline = Pipeline([
        ('classifier', LogisticRegression())
    ])

    search = GridSearchCV(
        pipeline,
        param_grid=tuning_params,
        scoring='average_precision',  # PRC-AUC
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # Find optimal threshold via F1
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-10)
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
        "best_params": search.best_params_,
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
    logger.info("Logistic regression model tuning completed.")
    logger.info(f"Best parameters: {search.best_params_}")
    logger.info(f"Best threshold: {best_threshold}")
    logger.info(f"PRC AUC: {metrics['prc_auc']}")
    logger.info(f"Top 10 precision: {metrics['top_10_precision']}")
    logger.info(f"Classification report: {metrics['classification_report']}")
    return best_model, metrics, predictions_df, top_10_df