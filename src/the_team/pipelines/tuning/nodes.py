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

def tune_logistic_model(
    data: pd.DataFrame,
    tuning_params: dict,
    test_size: float,
    stratify: bool,
    random_state: int
) -> tuple:

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

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
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

    return best_model, metrics, predictions_df, top_10_df