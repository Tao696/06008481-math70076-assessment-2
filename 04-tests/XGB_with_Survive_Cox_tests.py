import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor
from lifelines.utils import concordance_index


def train_xgb_survival_cox(data, features, n_folds=10, test_size=0.3, random_state=42,
                         plot_feature_importance=True, save_fig=None):
    """
    Run XGBoost with a Cox survival objective and return the trained model,
    test set concordance index (C-index), and feature importances.

    Parameters:
    - data: pandas DataFrame that includes features, 'efs_time', and 'efs'
    - features: list of feature column names
    - n_folds: number of cross-validation folds
    - test_size: proportion of data to be used for testing
    - random_state: random seed for reproducibility
    - plot_feature_importance: whether to plot feature importances
    - output_dir: if specified, save the feature importance plot to this directory

    Returns:
    - model_xgb_cox: the final trained XGBRegressor model
    - test_c_index_xgb_cox: concordance index on the test set
    - importance_df: pandas DataFrame with feature importances
    """
    data = data.copy()

    # Transform time-to-event column for Cox model
    data["efs_time2"] = data["efs_time"].copy()
    data.loc[data["efs"] == 0, "efs_time2"] *= -1

    # Train/test split
    train, test = train_test_split(data, test_size=test_size, random_state=random_state)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    model_xgb_cox = None  # to store last fold's model

    for i, (train_index, val_index) in enumerate(kf.split(train)):
        print("#" * 30)
        print(f"### Fold {i + 1}")
        print("#" * 30)

        x_train = train.iloc[train_index][features]
        y_train = train.iloc[train_index]["efs_time2"]
        x_val = train.iloc[val_index][features]
        y_val = train.iloc[val_index]["efs_time2"]

        x_test = test[features]
        y_test_time = test["efs_time"]
        y_test_event = test["efs"].astype(int)

        model_xgb_cox = XGBRegressor(
            device="cuda",
            max_depth=3,
            colsample_bytree=0.5,
            subsample=0.8,
            n_estimators=2000,
            learning_rate=0.02,
            enable_categorical=True,
            min_child_weight=80,
            objective='survival:cox',
            eval_metric='cox-nloglik',
        )

        model_xgb_cox.fit(
            x_train, y_train,
            eval_set=[(x_val, y_val)],
            verbose=500
        )

    # Prediction and evaluation on the test set
    y_pred = model_xgb_cox.predict(x_test)
    test_c_index_xgb_cox = concordance_index(
        event_times=y_test_time.values,
        predicted_scores=-y_pred,
        event_observed=y_test_event.values
    )
    print(f"Test C-index: {test_c_index_xgb_cox:.4f}")

    # Feature importance
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": model_xgb_cox.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    # Plot feature importance
    if plot_feature_importance:
        plt.figure(figsize=(10, 15))
        plt.barh(importance_df["Feature"], importance_df["Importance"])
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title("XGBoost Survival:Cox Feature Importance")
        plt.gca().invert_yaxis()

        if save_fig:
            plt.savefig(save_fig, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_fig}")
        else:
            plt.show()

    return model_xgb_cox, test_c_index_xgb_cox, importance_df
