import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from torch.utils.data.dataset import Dataset
import torch
import copy


def sample(y, offset, sample_inp_size, sample_out_size):
    Xin = np.arange(offset, offset + sample_inp_size)
    Xout = np.arange(sample_inp_size + offset, offset + sample_inp_size + sample_out_size)
    out = y[Xout]
    inp = y[Xin]
    return inp, out


def create_dataset(series, n_samples=None, sample_inp_size=51, sample_out_size=1, test=None, verbose=False, plot=False):
    if n_samples is None:
        n_samples = len(series)
    data_inp = np.zeros((n_samples, sample_inp_size))
    data_out = np.zeros((n_samples, sample_out_size))

    for i in range(n_samples):
        sample_inp, sample_out = sample(series, i, sample_inp_size, sample_out_size)
        data_inp[i, :] = sample_inp
        data_out[i, :] = sample_out
    if test is not None:
        assert 0 < test < 1
        split = int(n_samples * (1 - test))
        train_inp, train_out = data_inp[:split], data_out[:split]
        test_inp, test_out = data_inp[split:], data_out[split:]
        series_train = series[:split]
        series_test = series[split:]
    else:
        train_inp, train_out = data_inp, data_out
        test_inp, test_out = data_inp, data_out
        series_train = series
        series_test = series

    dataset_train = LocalDataset(x=train_inp, y=train_out)
    dataset_test = LocalDataset(x=test_inp, y=test_out)

    if verbose:
        print("Train set size: ", dataset_train.length)
        print("Test set size: ", dataset_test.length)

    if plot:
        # Plot generated process.
        plt.plot(np.array(series)[:200])
        plt.show()
    return dataset_train, dataset_test, series_train, series_test


class LocalDataset(Dataset):
    def __init__(self, x, y):
        x_dtype = torch.FloatTensor
        y_dtype = torch.FloatTensor     # for MSE or L1 Loss

        self.length = x.shape[0]

        self.x_data = torch.from_numpy(x).type(x_dtype)
        self.y_data = torch.from_numpy(y).type(y_dtype)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.length


def generate_armaprocess_data(samples, noise_std, random_order=None, params=None, limit_abs_sum=True):
    if params is not None:
        # use specified params, make sure to sum up to 1 or less
        arparams, maparams = params
        arma_process = ArmaProcess.from_coeffs(arparams, maparams, nobs=samples)
    else:
        is_stationary = False
        iteration = 0
        while not is_stationary:
            iteration += 1
            # print("Iteration", iteration)
            if iteration > 100:
                raise RuntimeError("failed to find stationary coefficients")
            # Generate random parameters
            arparams = []
            maparams = []
            ar_order, ma_order = random_order
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
                    arparams = arparams * (0.5 + 0.5*np.random.random())
                if ma_abssum > 1:
                    maparams = maparams / (ma_abssum + 10e-6)
                    maparams = maparams * (0.5 + 0.5*np.random.random())

            arparams = arparams - np.mean(arparams)
            maparams = maparams - np.mean(maparams)
            arma_process = ArmaProcess.from_coeffs(arparams, maparams, nobs=samples)
            is_stationary = arma_process.isstationary

    # sample output from ARMA Process
    series = arma_process.generate_sample(samples, scale=noise_std)
    # make zero-mean:
    series = series - np.mean(series)
    return series, arparams, maparams


def init_ar_dataset(n_samples, ar_val, ar_params=None, noise_std=1.0, plot=False, verbose=False, test=None, pad_to=None):
    # AR-Process
    if ar_params is not None:
        ar_val = len(ar_params)
        params = (ar_params, [])
    else:
        params = None

    if pad_to is None:
        inp_size = ar_val
    else:
        inp_size = pad_to

    series, ar, ma = generate_armaprocess_data(
        samples=n_samples+inp_size,
        noise_std=noise_std,
        random_order=(ar_val, 0),
        params=params,
    )
    # print("series mean", np.mean(series))

    if pad_to is not None:
        ar_pad = [0.0] * max(0, pad_to - ar_val)
        ar = list(ar) + ar_pad

    if verbose:
        print("AR params: ")
        print(ar)

    # Initialize data for DAR
    dataset_train, dataset_test, series_train, series_test = create_dataset(
        series=series,
        n_samples=n_samples,
        sample_inp_size=inp_size,
        sample_out_size=1,
        verbose=verbose,
        plot=plot,
        test=test,
    )

    return dataset_train, dataset_test, series_train, series_test, ar


def load_data(data_config_in, verbose=False, plot=False):
    data_config = copy.deepcopy(data_config_in)
    data_type = data_config.pop("type")
    data = {
        "type": data_type
    }
    if data_type == 'AR':
        data["train"], data["test"], data["series_train"], data["series_test"], data["ar"] = init_ar_dataset(
            **data_config,
            verbose=verbose,
            plot=plot,
        )
        data["pad_to"] = data_config_in["pad_to"]
    else:
        raise NotImplementedError
    return data
