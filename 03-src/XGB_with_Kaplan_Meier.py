# xgb_kaplan.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor
from lifelines.utils import concordance_index


def train_xgb_with_kaplan(data, features, n_folds=10, test_size=0.3, random_state=42,
                          plot_feature_importance=True, save_fig=None):
    """
    Train an XGBoost model to predict Kaplan-Meier transformed survival probabilities,
    with optional feature importance visualization.

    Parameters:
    - data: DataFrame including features and target y
    - features: list of feature column names
    - n_folds: number of folds for cross-validation
    - test_size: proportion of data to be used as test set
    - random_state: random seed for reproducibility
    - plot_feature_importance: whether to plot feature importance
    - save_fig_path: if provided, saves the feature importance plot to this path

    Returns:
    - model_xgb: trained XGBoost model
    - test_c_index: concordance index on the test set
    - importance_df: DataFrame of feature importances
    """
    train, test = train_test_split(data, test_size=test_size, random_state=random_state)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for i, (train_index, valid_index) in enumerate(kf.split(train)):
        print("#" * 25)
        print(f"### Fold {i+1}")
        print("#" * 25)

        x_train = train.iloc[train_index][features]
        y_train = train.iloc[train_index]["y"]
        x_valid = train.iloc[valid_index][features]
        y_valid = train.iloc[valid_index]["y"]

        x_test = test[features]

        model_xgb = XGBRegressor(
            device="cuda",                # Use GPU for training
            max_depth=3,                 # Maximum tree depth to prevent overfitting
            colsample_bytree=0.5,        # Randomly sample 50% of features per tree
            subsample=0.8,               # Use 80% of samples per tree
            n_estimators=2000,           # Maximum boosting rounds
            learning_rate=0.02,          # Learning rate (step size)
            enable_categorical=True,     # Enable categorical features (XGBoost 1.6+)
            min_child_weight=80          # Minimum child weight to control pruning
        )

        model_xgb.fit(x_train, y_train,
                      eval_set=[(x_valid, y_valid)],
                      verbose=500)

    y_pred = model_xgb.predict(x_test)
    test_c_index = concordance_index(
        event_times=test["efs_time"].values,
        predicted_scores=-y_pred,  # Survival probability → risk score
        event_observed=test["efs"].values.astype(int)
    )
    print(f"Test C-index: {test_c_index:.4f}")

    feature_importance = model_xgb.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": feature_importance
    }).sort_values(by="Importance", ascending=False)

    if plot_feature_importance:
        plt.figure(figsize=(10, 15))
        plt.barh(importance_df["Feature"], importance_df["Importance"])
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title("XGBoost Feature Importance (Kaplan-Meier Target)")
        plt.gca().invert_yaxis()

        if save_fig:
            plt.savefig(save_fig, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_fig}")
        else:
            plt.show()

    return model_xgb, test_c_index, importance_df
