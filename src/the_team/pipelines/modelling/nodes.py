"""
This is a boilerplate pipeline 'modelling'
generated using Kedro 0.19.12
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report, auc, roc_curve
from sklearn.dummy import DummyClassifier
from lightgbm import LGBMClassifier

# Create dummy model and fit minimal data when skipping certain models
dummy_model = DummyClassifier(strategy="most_frequent")
dummy_model.fit([[0]], [0])  # type: ignore # Minimal valid fit

dummy_metrics = {"skipped": True}
        
def train_logistic_model(
    data: pd.DataFrame,
    test_size: float,
    stratify: bool,
    random_state: int,
    C: float,
    max_iter: int,
    solver: str,
    class_weight: str,
    skip: bool
) -> tuple:
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
    if skip:
        print("Skipping model training as per the skip flag.")
        return dummy_model, dummy_metrics, pd.DataFrame(), pd.DataFrame()
    
    X = data.drop(columns=["is_repeat_buyer"])
    y = data["is_repeat_buyer"]
    stratify_obj = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=stratify_obj, random_state=random_state
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

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features + payment_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ], remainder='passthrough')

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver=solver, # type: ignore
            class_weight=class_weight
        ))
    ])

    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Build full predictions DataFrame
    predictions_df = X_test.copy()
    predictions_df["y_true"] = y_test.values
    predictions_df["y_pred"] = y_pred
    predictions_df["y_proba"] = y_proba

    # Top 10% prediction
    top_k = int(len(y_test) * 0.1)
    top_10_df = predictions_df.sort_values("y_proba", ascending=False).head(top_k)

    # PRC-AUC
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    prc_auc = auc(recall, precision)
    
    # ROC curve data
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    # Metrics
    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "prc_auc": prc_auc,
        "top_10_precision": top_10_df["y_true"].mean(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "prc_curve": {
            "precision": precision.tolist(),
            "recall": recall.tolist()
        },
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist()
        }
    }

    return model, metrics, predictions_df, top_10_df

def train_lightgbm_model(
    data: pd.DataFrame,
    test_size: float,
    stratify: bool,
    random_state: int,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    scale_pos_weight: float,
    skip: bool
) -> tuple:
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
    if skip:
        print("Skipping model training as per the skip flag.")
        return dummy_model, dummy_metrics, pd.DataFrame(), pd.DataFrame()
    
    # Split
    X = data.drop(columns=["is_repeat_buyer"])
    y = data["is_repeat_buyer"]
    stratify_obj = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=stratify_obj, random_state=random_state
    )

    # Feature engineering
    numeric_features = [
        'deli_duration_exp', 'deli_duration_paid', 'deli_cost', 'item_price', 'total_spent',
        'installment', 'distance_km', 'high_density_customer_area', 'seller_repeat_buyer_rate',
        'review_score', 'product_name_length', 'product_description_length',
        'product_photos_qty', 'product_weight_g', 'product_length_cm',
        'product_height_cm', 'product_width_cm'
    ]
    payment_features = ['credit_card', 'voucher', 'debit_card', 'boleto']
    cat_features = ['product_category_name']

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features + payment_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ], remainder='passthrough')

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            class_weight=None  # Not needed when using scale_pos_weight
        ))
    ])

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Build predictions DataFrame
    predictions_df = X_test.copy()
    predictions_df["y_true"] = y_test.values
    predictions_df["y_pred"] = y_pred
    predictions_df["y_proba"] = y_proba

    # Top 10%
    top_k = int(len(y_test) * 0.1)
    top_10_df = predictions_df.sort_values("y_proba", ascending=False).head(top_k)

    # Evaluation
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    prc_auc = auc(recall, precision)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    metrics = {
        "roc_auc": roc_auc,
        "prc_auc": prc_auc,
        "top_10_precision": top_10_df["y_true"].mean(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "prc_curve": {"precision": precision.tolist(), "recall": recall.tolist()}
    }

    return model, metrics, predictions_df, top_10_df