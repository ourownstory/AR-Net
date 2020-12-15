import pandas as pd
import numpy as np
import logging

from arnet.create_ar_data import load_from_file

log = logging.getLogger("ARNet")


def estimate_noise(series):
    return float(np.mean(np.abs(series.iloc[:-1].values - series.iloc[1:].values)))


def split_by_p_valid(valid_p, n_sample, verbose=False):
    split_idx = int(n_sample * (1 - valid_p))
    splits = [list(range(split_idx)), list(range(split_idx, n_sample))]
    if verbose:
        print("split on idx: ", split_idx)
        print("split sizes: ", [len(x) for x in splits])
    return splits


def tabularize_univariate(series, n_lags, n_forecasts=1, nested_list=False):
    """
    Create a tabular dataset with ar_order lags for supervised forecasting
    Arguments:
        series: Sequence of observations as a Pandas DataFrame
        n_lags: Number of lag observations as input (X).
        n_forecasts: Number of observations as output (y).
    Returns:
        df: Pandas DataFrame  of input lags and forecast values (as nested lists)
            shape (n_samples, 2).
            Cols: "x": list(n_lags)
            Cols: "y": list(n_forecasts)
    """
    n_samples = len(series) - n_lags + 1 - n_forecasts

    x = pd.DataFrame([series.iloc[i : i + n_lags, 0].values for i in range(n_samples)])
    y = pd.DataFrame([series.iloc[i + n_lags : i + n_lags + n_forecasts, 0].values for i in range(n_samples)])
    if nested_list:
        df = pd.concat([x.apply(list, axis=1), y.apply(list, axis=1)], axis=1)
        df.columns = ["x", "y"]
    else:
        df = pd.concat([x, y], axis=1)
        df.columns = ["x_{}".format(num) for num in list(range(len(x.columns)))] + [
            "y_{}".format(num) for num in list(range(len(y.columns)))
        ]
    return df


def main():
    verbose = True
    data_path = "ar_data"
    data_name = "ar_3_ma_0_noise_0.100_len_10000"

    ## if created AR data with create_ar_data, we can use the helper function:
    df, data_config = load_from_file(data_path, data_name, load_config=True)
    n_lags = data_config["ar_order"]

    ## else we can manually load any file that stores a time series, for example:
    # df = pd.read_csv(os.path.join(data_path, data_name + '.csv'), header=None, index_col=False)
    # n_lags = 3

    if verbose:
        print(data_config)
        print(df.shape)

    ## create a tabularized dataset from time series
    df_tab = tabularize_univariate(
        df,
        n_lags=n_lags,
        n_forecasts=1,
        nested_list=False,
    )

    if verbose:
        print("tabularized df")
        print(df_tab.shape)
        # print(df_tab.columns)
        # if nested_list:
        #     print("x_dim:", len(df_tab['x'][0]), "y_dim:", len(df_tab['y'][0]))
        print(df_tab.head())


if __name__ == "__main__":
    main()
