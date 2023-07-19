import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def corrfunc(x, y, hue=None, ax=None, **kws):
    """Plot the correlation coefficient in the top left corner of a plot."""
    r = np.corrcoef(x, y)[0][1]
    ax = ax or plt.gca()
    ax.annotate(f"ρ = {r:.3f}", xy=(0.1, 0.9), xycoords=ax.transAxes)


def plot_lags(df: pd.DataFrame, col: str, lags: list[int]) -> pd.DataFrame:
    """Plot variable with its lags."""
    df_lags = df.loc[:, [col]].copy()
    for lag in lags:
        df_lags[f"lag_{col}_{lag}d"] = df_lags[col].shift(lag)
    df_lags.dropna(inplace=True)

    plot_kws = dict(scatter_kws={"alpha": 0.1})
    g = sns.pairplot(df_lags, kind="reg", y_vars=[col], plot_kws=plot_kws)
    g = g.map_upper(corrfunc)
    return df_lags
