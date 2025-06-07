"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.19.12
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from logging import getLogger
logger = getLogger(__name__)

# Function to split data into training and testing sets
def split_data(
    df: pd.DataFrame,
    test_size: float,
    stratify: bool,
    random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and testing sets.
    Args:
        df (pd.DataFrame): The input DataFrame containing features and target variable.
        test_size (float): Proportion of the dataset to include in the test split.
        stratify (bool): Whether to stratify the split based on the target variable.
        random_state (int): Random seed for reproducibility.
    Returns:
        tuple: A tuple containing the training features, testing features, training target, and testing target.
    """
    logger.info("Splitting data into training and testing sets.")
    if "is_repeat_buyer" not in df.columns:
        raise ValueError("DataFrame must contain 'is_repeat_buyer' column for target variable.")
    if df.empty:
        raise ValueError("DataFrame is empty. Cannot perform train-test split.")
    X = df.drop(columns=["is_repeat_buyer"])
    y = df["is_repeat_buyer"]
    stratify_obj = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=stratify_obj,
        random_state=random_state
    )
    logger.info(f"Training set size: {X_train.shape[0]}, Testing set size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test

# Create a common preprocessor for the data
def build_preprocessor(numeric_features: list[str],
                      payment_features: list[str],
                      cat_features: list[str]) -> ColumnTransformer:
    """
    Create a preprocessor for the dataset that scales numeric features and one-hot encodes categorical features.
    Args:
        numeric_features (list[str]): List of numeric feature names to be scaled.
        payment_features (list[str]): List of payment-related feature names to be scaled.
        cat_features (list[str]): List of categorical feature names to be one-hot encoded.
    Returns:
        ColumnTransformer: A transformer that applies StandardScaler to numeric features and OneHotEncoder to categorical features.
    """
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features + payment_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ],
        remainder='passthrough'
    )
    
# Function to apply the preprocessor to the training DataFrame
def apply_preprocessor_train(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame
) -> tuple[ColumnTransformer, pd.DataFrame]:
    """Fits the transformer on X and returns the fitted transformer and transformed data.
    Args:
        preprocessor (ColumnTransformer): The preprocessor to fit and transform the data.
        X (pd.DataFrame): The input DataFrame containing features to be transformed.
    Returns:
        tuple: A tuple containing the fitted preprocessor and a DataFrame with transformed features.
    """
    arr = preprocessor.fit_transform(X)
    cols = preprocessor.get_feature_names_out()
    transformed = pd.DataFrame(arr, columns=cols, index=X.index) 
    return preprocessor, transformed  

# Function to apply the preprocessor to the testing DataFrame
def apply_preprocessor_test(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame
) -> pd.DataFrame:
    """Applies the transformer to X and returns a DataFrame."""
    arr = preprocessor.transform(X)
    cols = preprocessor.get_feature_names_out()
    return pd.DataFrame(arr, columns=cols, index=X.index) # type: ignore

def extract_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """
    After fitting, grab the exact output column names your model saw.
    """
    return preprocessor.get_feature_names_out().tolist()