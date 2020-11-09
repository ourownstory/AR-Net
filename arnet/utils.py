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


def set_logger_level(logger, log_level=None, include_handlers=False):
    if log_level is None:
        logger.warning("Failed to set log_level to None.")
    elif log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", 10, 20, 30, 40, 50):
        logger.error(
            "Failed to set log_level to {}."
            "Please specify a valid log level from: "
            "'DEBUG', 'INFO', 'WARNING', 'ERROR' or 'CRITICAL'"
            "".format(log_level)
        )
    else:
        logger.setLevel(log_level)
        if include_handlers:
            for h in logger.handlers:
                h.setLevel(log_level)
        logger.debug("Set log level to {}".format(log_level))
