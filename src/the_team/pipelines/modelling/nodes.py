"""
This is a boilerplate pipeline 'modelling'
generated using Kedro 0.19.12
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report, auc, roc_curve
from sklearn.dummy import DummyClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import logging
logger = logging.getLogger(__name__)

# Create dummy model and fit minimal data when skipping certain models
dummy_model = DummyClassifier(strategy="most_frequent")
dummy_model.fit([[0]], [0])  # type: ignore # Minimal valid fit
dummy_metrics = {"skipped": True}
    
# Function to collect evaluation metrics and predictions
def evaluate_predictions(y_test, y_proba, y_pred, X_test):
    """
    Evaluate the model's predictions and collect various metrics.
    Args:
        y_test (pd.Series): True labels for the test set.
        y_proba (np.ndarray): Predicted probabilities for the positive class.
        y_pred (np.ndarray): Predicted labels for the test set.
        X_test (pd.DataFrame): Features of the test set.
    Returns:
        dict: A dictionary containing evaluation metrics such as ROC AUC, PRC AUC, top 10% precision, classification report, and curves.
        pd.DataFrame: DataFrame containing the test set features along with true labels, predicted labels, and predicted probabilities.
        pd.DataFrame: DataFrame containing the top 10% predictions based on predicted probabilities.
    """
    if y_test.empty or y_proba.size == 0 or y_pred.size == 0:
        raise ValueError("y_test, y_proba, and y_pred must not be empty or zero-length.")
    # predictions dataframe
    df = X_test.copy()
    df["y_true"]  = y_test.values
    df["y_pred"]  = y_pred
    df["y_proba"] = y_proba

    # top 10% precision
    top_k     = int(len(y_test) * 0.1)
    top_df    = df.nlargest(top_k, "y_proba")
    top_prec  = top_df["y_true"].mean()

    # PRC & ROC curves
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    prc_auc      = auc(rec, prec)
    fpr, tpr, _  = roc_curve(y_test, y_proba)
    roc_auc      = roc_auc_score(y_test, y_proba)

    return {
        "roc_auc": roc_auc,
        "prc_auc": prc_auc,
        "top_10_precision": top_prec,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "feature_importance": None,   # fill later
        "prc_curve": {"precision": prec.tolist(), "recall": rec.tolist()},
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    }, df, top_df

def train_model(
    X_train: pd.DataFrame,            
    X_test: pd.DataFrame,             
    y_train: pd.Series,            
    y_test: pd.Series,      
    clf_builder,    # a callable that returns an instantiated estimator
    params: dict,   # hyperparams for that classifier
    skip: bool,
    use_feature_importance: bool = True,
    model_name: str = "default_model"  # name for logging purposes
):
    """
    clf_builder(**params) should give you an unfitted estimator,
    e.g. lambda **p: RandomForestClassifier(**p)
    """
    if skip:
        logger.info("Skipping %s model training", model_name)
        return dummy_model, dummy_metrics, pd.DataFrame(), pd.DataFrame()
    logger.info("Training %s model", model_name)

    # 2) build pipeline
    pipe = Pipeline([
        ('classifier', clf_builder(**params))
    ])

    pipe.fit(X_train, y_train)

    # 3) feature importance or coefficients
    metrics, pred_df, top_df = evaluate_predictions(
        y_test=y_test,
        y_proba=pipe.predict_proba(X_test)[:,1],
        y_pred=pipe.predict(X_test),
        X_test=X_test
    )

    if use_feature_importance:
        clf = pipe.named_steps['classifier']
        if hasattr(clf, 'feature_importances_'):
            imps = clf.feature_importances_
        else:
            # e.g. logistic: absolute coef
            imps = np.abs(clf.coef_[0])
        feat_names = X_test.columns
        metrics["feature_importance"] = {
            name: float(score)
            for name, score in sorted(
                zip(feat_names, imps), key=lambda x: -x[1]
            )
        }

    logger.info(
        "Model %s trained with ROC AUC: %.4f, PRC AUC: %.4f, Top 10%% Precision: %.4f",
        model_name, metrics["roc_auc"], metrics["prc_auc"], metrics["top_10_precision"]
    )

    return pipe, metrics, pred_df, top_df

def train_random_forest_model(X_train: pd.DataFrame, 
                            X_test: pd.DataFrame, 
                            y_train: pd.Series, 
                            y_test: pd.Series, 
                            random_state: int,             
                            n_estimators: int, 
                            max_depth: int, 
                            class_weight: str, 
                            skip: bool):

    """
    Train a Random Forest model to predict repeat buyers.
    Args:
        data (pd.DataFrame): Input data containing features and target variable.
        test_size (float): Proportion of the dataset to include in the test split.
        stratify (bool): Whether to stratify the split based on the target variable.
        random_state (int): Random seed for reproducibility.
        n_estimators (int): Number of trees in the forest.
        max_depth (int): Maximum depth of the tree.
        class_weight (str): Weights associated with classes in the form {class_label: weight}.
        skip (bool): Flag to skip model training if set to True.
    Returns:
        tuple: A tuple containing the trained model, metrics, predictions DataFrame, and top 10% predictions DataFrame.
    """
    return train_model(
        X_train, X_test, y_train, y_test,
        clf_builder=lambda **p: RandomForestClassifier(random_state=random_state, **p, n_jobs=-1),
        params={"n_estimators": n_estimators, "max_depth": max_depth, "class_weight": class_weight},
        skip=skip, model_name="RandomForest"
    )

def train_logistic_model(X_train: pd.DataFrame, 
                        X_test: pd.DataFrame, 
                        y_train: pd.Series, 
                        y_test: pd.Series,                      
                        C: float, 
                        max_iter: int,
                        solver: str, 
                        class_weight: str, 
                        skip: bool):
    """
    Train a logistic regression model to predict repeat buyers.
    Args:
        data (pd.DataFrame): Input data containing features and target variable.
        test_size (float): Proportion of the dataset to include in the test split.
        stratify (bool): Whether to stratify the split based on the target variable.
        random_state (int): Random seed for reproducibility.
        C (float): Inverse of regularization strength; smaller values specify stronger regularization.
        max_iter (int): Maximum number of iterations for the solver.
        solver (str): The algorithm to use in the optimization problem.
        class_weight (str): Weights associated with classes in the form {class_label: weight}.
        skip (bool): Flag to skip model training if set to True.
    Returns:
        tuple: A tuple containing the trained model, metrics, predictions DataFrame, and top 10% predictions DataFrame.
    """
    return train_model(
        X_train, X_test, y_train, y_test,
        clf_builder=lambda **p: LogisticRegression(class_weight=class_weight, solver=solver, max_iter=max_iter, **p), # type: ignore
        params={"C": C},
        skip=skip, model_name="LogisticRegression"
    )

def train_lightgbm_model(X_train: pd.DataFrame, 
                        X_test: pd.DataFrame, 
                        y_train: pd.Series, 
                        y_test: pd.Series, 
                        random_state: int,                 
                        n_estimators: int, 
                        learning_rate: float, 
                        max_depth: int, 
                        scale_pos_weight: float, 
                        skip: bool):
    """
    Train a LightGBM model to predict repeat buyers.

    Args:
        data (pd.DataFrame): Input data containing features and target variable.
        test_size (float): Proportion of the dataset to include in the test split.
        stratify (bool): Whether to stratify the split based on the target variable.
        random_state (int): Random seed for reproducibility.
        n_estimators (int): Number of boosting iterations.
        learning_rate (float): Learning rate for boosting.
        max_depth (int): Maximum depth of the tree.
        scale_pos_weight (float): Scaling factor for the positive class in imbalanced datasets.
        skip (bool): Flag to skip model training if set to True.
        
    Returns:
        tuple: A tuple containing the trained model, metrics, predictions DataFrame, and top 10% predictions DataFrame.
    """
    return train_model(
        X_train, X_test, y_train, y_test,
        clf_builder=lambda **p: LGBMClassifier(random_state=random_state, **p),
        params={
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "scale_pos_weight": scale_pos_weight
        },
        skip=skip, model_name="LightGBM"
    )

def train_xgboost_model(X_train: pd.DataFrame, 
                        X_test: pd.DataFrame, 
                        y_train: pd.Series, 
                        y_test: pd.Series, 
                        random_state: int,                       
                        n_estimators: int, 
                        learning_rate: float, 
                        max_depth: int, 
                        scale_pos_weight: float, 
                        skip: bool):
    """
    Train an XGBoost model to predict repeat buyers.
    Args:
        data (pd.DataFrame): Input data containing features and target variable.
        test_size (float): Proportion of the dataset to include in the test split.
        stratify (bool): Whether to stratify the split based on the target variable.
        random_state (int): Random seed for reproducibility.
        n_estimators (int): Number of boosting rounds.
        learning_rate (float): Learning rate for boosting.
        max_depth (int): Maximum depth of a tree.
        scale_pos_weight (float): Scaling factor for the positive class in imbalanced datasets.
        skip (bool): Flag to skip model training if set to True.
    Returns:
        tuple: A tuple containing the trained model, metrics, predictions DataFrame, and top 10% predictions DataFrame.
    """
    return train_model(
        X_train, X_test, y_train, y_test, 
        clf_builder=lambda **p: XGBClassifier(
            random_state=random_state, use_label_encoder=False, eval_metric="logloss", **p
        ),
        params={
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "scale_pos_weight": scale_pos_weight
        },
        skip=skip, model_name="XGBoost"
    )