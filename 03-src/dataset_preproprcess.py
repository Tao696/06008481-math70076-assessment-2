import pandas as pd
import numpy as np



def dataset_overview(df: pd.DataFrame, FEATURES):
    """
    Display an overview of the dataset features: including column name, inferred data type, 
    number of unique values, and missing rate.

    Args:
        df: Input Pandas DataFrame.
        FEATURES: List of feature column names to analyze.

    Returns:
        A DataFrame showing the summary information for each feature.
    """
    print("Dataset Feature Overview")
    summary = []

    for var in FEATURES:
        if var in df.columns:
            dtype = df[var].dtype.name
            missing = df[var].isnull().mean()
            nunique = df[var].nunique(dropna=True)

            summary.append({
                "Feature": var,
                "Data Type": dtype,
                "Unique Values": nunique,
                "Missing Rate": f"{missing:.2%}"
            })

    return pd.DataFrame(summary)


def preprocess_dataset(df: pd.DataFrame, features: list):
    """
    Preprocess the specified features in the dataset:
    - Object-type columns → Label encoded as categorical
    - Numeric columns with fewer than 15 unique values → treated as categorical
    - Other numeric columns → missing value imputation + type downcasting

    Args:
        df: Original Pandas DataFrame
        features: List of feature column names

    Returns:
        A new DataFrame with preprocessed features
    """
    df = df.copy()
    categorical_vars = []
    numeric_vars = []
    cat_as_num_vars = []

    for col in features:
        if df[col].dtype == "object":
            # Handle string-type features
            df[col] = df[col].fillna("NAN")
            categorical_vars.append(col)

        else:
            # For numeric features, determine if treated as categorical
            nunique = df[col].nunique()
            if nunique < 20:
                df[col] = df[col].fillna("NAN").astype(str)  # Convert to string for encoding
                cat_as_num_vars.append(col)
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].fillna(df[col].median())
                numeric_vars.append(col)

    # Combine all categorical features
    all_cats = categorical_vars + cat_as_num_vars

    print(f"Total categorical features (including low-cardinality numerics): {len(all_cats)} → {all_cats}")
    for col in all_cats:
        df[col], _ = df[col].factorize()
        df[col] -= df[col].min()
        df[col] = df[col].astype("int32").astype("category")

    # Downcast numeric types for memory efficiency
    for col in numeric_vars:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
        if df[col].dtype == "int64":
            df[col] = df[col].astype("int32")

    return df
