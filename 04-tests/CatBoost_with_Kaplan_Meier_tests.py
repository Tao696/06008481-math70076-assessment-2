# catboost_kaplan_module.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from catboost import CatBoostRegressor
from lifelines.utils import concordance_index
import os

def train_catboost_kaplan(data, features, CAT_vir, n_folds=10, test_size=0.3, random_state=42,
                          plot_feature_importance=True, save_fig=None):
    """
    Train CatBoostRegressor using KaplanMeier transformed targets and return C-index and feature importance.

    Parameters:
        train (pd.DataFrame): Training data including 'y', FEATURES, and 'efs_time', 'efs'.
        test (pd.DataFrame): Testing data.
        FEATURES (list): List of feature column names.
        CAT_vir (list): List of categorical feature indices.
        output_dir (str): Directory to save output plots.

    Returns:
        float: Test C-index.
        pd.DataFrame: Feature importance dataframe.
    """

    train, test = train_test_split(data, test_size=test_size, random_state=random_state)

    n_folds = 10
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for i, (train_index, test_index) in enumerate(kf.split(train)):
        print("#"*25)
        print(f"### Fold {i+1}")
        print("#"*25)

        x_train = train.iloc[train_index][features].copy()
        y_train = train.iloc[train_index]["y"]
        x_valid = train.iloc[test_index][features].copy()
        y_valid = train.iloc[test_index]["y"]

        model = CatBoostRegressor(
            task_type="GPU",
            learning_rate=0.1,
            grow_policy='Lossguide',
            verbose=250
        )
        model.fit(x_train, y_train, eval_set=(x_valid, y_valid), cat_features=CAT_vir)

    y_pred = model.predict(test[features])

    true_efs_time = test["efs_time"].values
    efs_true = test["efs"].values.astype(int)

    c_index = concordance_index(
        event_times=true_efs_time,
        predicted_scores=-y_pred,
        event_observed=efs_true
    )
    print(f"Test C-index: {c_index:.4f}")

    # Feature importance
    feature_importance = model.get_feature_importance()
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": feature_importance
    }).sort_values(by="Importance", ascending=False)

    # Save plot
    plt.figure(figsize=(10, 15))
    plt.barh(importance_df["Feature"], importance_df["Importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("CatBoost KaplanMeier Feature Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    if save_fig:
        plt.savefig(save_fig, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_fig}")
    else:
        plt.show()
    #plt.savefig(f"{savefig}/02_catboost_Kaplanmerier_feature_importance.png")
    plt.close()

    return model, c_index, importance_df
