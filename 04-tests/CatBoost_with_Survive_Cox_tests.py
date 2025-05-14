import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from lifelines.utils import concordance_index
from catboost import CatBoostRegressor
import os


def train_catboost_survival_cox(data, features, CAT_vir=None, n_folds=10, test_size=0.3,
                              random_state=42, plot_feature_importance=True, save_fig=None):
    """
    Train a CatBoost Cox survival model using cross-validation.

    Parameters:
    - data: pandas DataFrame, must contain 'efs_time' and 'efs'
    - features: list of feature column names
    - cat_features: list of categorical feature names or indices
    - n_folds: number of CV folds
    - test_size: float between 0 and 1 for train/test split
    - random_state: seed for reproducibility
    - plot_feature_importance: whether to plot feature importance
    - save_fig: path to save the feature importance figure (e.g., 'output/importance.png')

    Returns:
    - model_cat_cox: trained CatBoostRegressor model
    - test_c_index_cat_cox: concordance index on the test set
    - importance_df: DataFrame with sorted feature importances
    """

    data = data.copy()

    # Prepare survival target
    data["efs_time2"] = data["efs_time"].copy()
    data.loc[data["efs"] == 0, "efs_time2"] *= -1

    train, test = train_test_split(data, test_size=test_size, random_state=random_state)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    model_cat_cox = None  # to store the last model

    for i, (train_index, val_index) in enumerate(kf.split(train)):
        print("#" * 30)
        print(f"### Fold {i + 1}")
        print("#" * 30)

        x_train = train.iloc[train_index][features].copy()
        y_train = train.iloc[train_index]["efs_time2"]
        x_val = train.iloc[val_index][features].copy()
        y_val = train.iloc[val_index]["efs_time2"]

        model_cat_cox = CatBoostRegressor(
            loss_function="Cox",
            iterations=400,
            learning_rate=0.1,
            grow_policy='Lossguide',
            use_best_model=False,
            verbose=100,
        )

        model_cat_cox.fit(
            x_train, y_train,
            eval_set=(x_val, y_val),
            cat_features=CAT_vir
        )

    # Evaluate on test set
    x_test = test[features].copy()
    y_test_time = test["efs_time"].values
    y_test_event = test["efs"].astype(int).values

    y_pred = model_cat_cox.predict(x_test)
    test_c_index_cat_cox = concordance_index(
        event_times=y_test_time,
        predicted_scores=-y_pred,
        event_observed=y_test_event
    )
    print(f"Test C-index: {test_c_index_cat_cox:.4f}")

    # Feature importance
    importances = model_cat_cox.get_feature_importance()
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    # Plot feature importance
    if plot_feature_importance:
        plt.figure(figsize=(10, 15))
        plt.barh(importance_df["Feature"], importance_df["Importance"])
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title("CatBoost Survival:Cox Feature Importance")
        plt.gca().invert_yaxis()

        if save_fig:
            os.makedirs(os.path.dirname(save_fig), exist_ok=True)
            plt.savefig(save_fig)
        else:
            plt.show()

    return model_cat_cox, test_c_index_cat_cox, importance_df
