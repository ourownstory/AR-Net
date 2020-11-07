import numpy as np
import torch


def pad_ar_params(ar_params, n_lags, n_forecasts=1):
    """ "
    pads ar_parameter lists to the length of n_lags
        ar_params: list of length n_forecasts with elements: lists of ar coeffs
        n_lags: length to which pad each of the ar coeffs
    """
    assert n_forecasts == len(ar_params)
    if n_forecasts != 1:
        if all([isinstance(ar_params[i], list) for i in range(n_forecasts)]):
            return [pad_ar_params([ar_params[i]], n_lags, 1)[0] for i in range(n_forecasts)]
        else:
            raise NotImplementedError("AR Coeff for each of the forecast targets are needed")
    return [ar_params[0] + [0.0] * (n_lags - len(ar_params[0]))]


def estimate_noise(series):
    return float(np.mean(np.abs(series.iloc[:-1].values - series.iloc[1:].values)))


def split_by_p_valid(valid_p, n_sample, verbose=False):
    split_idx = int(n_sample * (1 - valid_p))
    splits = [list(range(split_idx)), list(range(split_idx, n_sample))]
    if verbose:
        print("split on idx: ", split_idx)
        print("split sizes: ", [len(x) for x in splits])
    return splits


def nice_print_list(data):
    if all([isinstance(data[i], list) for i in range(len(data))]):
        return [nice_print_list(data[i]) for i in range(len(data))]
    return ["{:.3f}".format(x) for x in data]
    # return [["{:.2f}".format(x) for x in sublist] for sublist in data]


def compute_sTPE(est, real):
    est, real = np.array(est), np.array(real)
    sum_abs_diff = np.sum(np.abs(est - real))
    sum_abs = np.sum(np.abs(est) + np.abs(real))
    return 100.0 * sum_abs_diff / (10e-9 + sum_abs)


def coeff_from_model(model, reversed_weights=True):
    for layer in model.modules():
        if isinstance(layer, torch.nn.Linear):
            weights = [list(x[::-1] if reversed_weights else x) for x in layer.weight.detach().numpy()]
            return weights  # note: preliminary exit of loop is a feature.
