# km_transform.py
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

def transform_survival_probability(df, time_col='efs_time', event_col='efs', plot=False):
    """
    Use Kaplan-Meier to transform the target variable, with optional visualization.

    Parameters:
    - df: input DataFrame (must include survival time and event columns)
    - time_col: column name for survival time
    - event_col: column name for event indicator
    - plot: whether to display survival curve and histogram

    Returns:
    - y: array of survival probabilities
    - kmf: fitted KaplanMeierFitter object
    """
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], df[event_col])
    y = kmf.survival_function_at_times(df[time_col]).values

    if plot:
        # Plot the survival function
        kmf.plot_survival_function()
        plt.title("Kaplan-Meier Estimated Survival Curve")
        plt.xlabel("Time")
        plt.ylabel("Survival Probability")
        plt.grid(True)
        plt.show()

        # Plot histogram of transformed survival probabilities
        df = df.copy()
        df["y"] = y
        plt.hist(df.loc[df[event_col]==1, "y"], bins=100, alpha=0.6, label="Event=1")
        plt.hist(df.loc[df[event_col]==0, "y"], bins=100, alpha=0.6, label="Event=0")
        plt.xlabel("Transformed Target y (Survival Probability)")
        plt.ylabel("Frequency")
        plt.title("Kaplan-Meier Transformed Target y")
        plt.legend()
        plt.grid(True)
        plt.show()

    return y, kmf
