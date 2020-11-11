import os
import json
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess


def _get_config(noise_std=0.1, n_samples=10000, random_ar_params=True):
    # option 1: Randomly generated AR parameters
    data_config_random = {"noise_std": noise_std, "ar_order": 3, "ma_order": 0, "params": None, "samples": n_samples}
    # option 2: Manually define AR parameters
    data_config_manual = {"noise_std": noise_std, "params": ([0.2, 0.3, -0.5], [])}
    data_config_manual["ar_order"] = int(sum(np.array(data_config_manual["params"][0]) != 0.0))
    data_config_manual["ma_order"] = int(sum(np.array(data_config_manual["params"][1]) != 0.0))
    data_config_manual["samples"] = n_samples  # + int(data_config_manual["ar_order"])
    return data_config_random if random_ar_params else data_config_manual


def _generate_random_arparams(ar_order, ma_order, limit_abs_sum=True, maxiter=100):
    is_stationary = False
    iteration = 0
    while not is_stationary:
        iteration += 1
        # print("Iteration", iteration)
        if iteration > maxiter:
            raise RuntimeError("failed to find stationary coefficients")
        # Generate random parameters
        arparams = []
        maparams = []
        for i in range(ar_order):
            arparams.append(2 * np.random.random() - 1)
        for i in range(ma_order):
            maparams.append(2 * np.random.random() - 1)
        # print(arparams)
        arparams = np.array(arparams)
        maparams = np.array(maparams)
        if limit_abs_sum:
            ar_abssum = sum(np.abs(arparams))
            ma_abssum = sum(np.abs(maparams))
            if ar_abssum > 1:
                arparams = arparams / (ar_abssum + 10e-6)
                arparams = arparams * (0.5 + 0.5 * np.random.random())
            if ma_abssum > 1:
                maparams = maparams / (ma_abssum + 10e-6)
                maparams = maparams * (0.5 + 0.5 * np.random.random())

        arparams = arparams - np.mean(arparams)
        maparams = maparams - np.mean(maparams)
        arma_process = ArmaProcess.from_coeffs(arparams, maparams, nobs=100)
        is_stationary = arma_process.isstationary
    return arparams, maparams


def generate_armaprocess_data(samples, ar_order, ma_order, noise_std, params=None):
    if params is not None:
        # use specified params (make sure to sum up to 1 or less)
        arparams, maparams = params
    else:
        # iterate to find random arparams that are stationary
        arparams, maparams = _generate_random_arparams(ar_order, ma_order)
    arma_process = ArmaProcess.from_coeffs(arparams, maparams, nobs=samples)
    # sample output from ARMA Process
    series = arma_process.generate_sample(samples, scale=noise_std)
    # make zero-mean:
    series = series - np.mean(series)
    return series, list(arparams), list(maparams)


def save_to_file(save_path, series, data_config):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_data = "ar_{}_ma_{}_noise_{:.3f}_len_{}".format(
        data_config["ar_order"], data_config["ma_order"], data_config["noise_std"], data_config["samples"]
    )
    # data_config["ar_params"] = list(data_config["ar_params"])
    # data_config["ma_params"] = list(data_config["ma_params"])
    np.savetxt(os.path.join(save_path, file_data + ".csv"), series, delimiter=",")
    with open(os.path.join(save_path, "info_" + file_data + ".json"), "w") as f:
        json.dump(data_config, f)
    return file_data


def load_from_file(data_path, data_name, load_config=True, verbose=False):
    df = pd.read_csv(os.path.join(data_path, data_name + ".csv"), header=None, index_col=False)
    if load_config:
        with open(os.path.join(data_path, "info_" + data_name + ".json"), "r") as f:
            data_config = json.load(f)
    else:
        data_config = None
    if verbose:
        print("loaded series from file")
        print("data_config", data_config)
        print(df.shape)
        print(df.head())
    return df, data_config


def main():
    verbose = True
    random = False
    save = True
    save_path = "ar_data"

    data_config = _get_config(random_ar_params=random)
    if verbose:
        print(data_config)

    series, data_config["ar_params"], data_config["ma_params"] = generate_armaprocess_data(**data_config)
    del data_config["params"]

    if save:
        data_name = save_to_file(save_path, series, data_config)

        # just to test:
        df, data_config2 = load_from_file(save_path, data_name, load_config=True)
        if verbose:
            print("loaded from saved files:")
            print(data_config2)
            print(df.head())


if __name__ == "__main__":
    main()
