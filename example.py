import numpy as np

import utils
from data_loader import load_data
from training import run as run_training


def load_config(verbose=False, random=True):
    # load specified settings

    #### Data settings ####
    data_config = {
        "type": 'AR',
        "ar_val": 3,
        "pad_to": 10,  # set to >ar_val for sparse AR estimation
        "ar_params": None,  # for randomly generated AR params
        "noise_std": 1.0,
        "test": 0.2,
        "n_samples": int(1.25e5),  # for 1e5 train size
    }

    # OR manually define AR params:
    if not random:
        # data_config["ar_params"] = [0.2, 0.3, -0.5]
        # Alternative: sparse AR params:
        data_config["ar_params"] = [0.2, 0, 0.3, 0, 0, 0, 0, 0, 0, -0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # correct settings if manually set
        data_config["ar_val"] = sum(np.array(data_config["ar_params"]) != 0.0)
        data_config["pad_to"] = int(len(data_config["ar_params"]))

    #### Model settings ####
    model_config = {
        "ar": data_config["ar_val"],  # for normal AR
        "ma": 0,
        "num_layers": 1,
        "d_hidden": None
    }
    if data_config["pad_to"] is not None and data_config["pad_to"] > data_config["ar_val"]:
        model_config["ar"] = data_config["pad_to"]  # for sparse AR

    #### Train settings ####
    train_config = {
        "lr": 2e-4,
        "lr_decay": 0.9,
        "epochs": 10,
        "batch": 128,
        "est_sparsity": 1,  # 0 = fully sparse, 1 = not sparse
        "lambda_delay": 10,  # delays start of regularization by lambda_delay epochs
    }
    # For auto-regularization based on sparsity estimation:
    if data_config["pad_to"] is not None and data_config["pad_to"] > data_config["ar_val"]:
        train_config["est_sparsity"] = data_config["ar_val"] / (1.0 * data_config["pad_to"])

    # Note: find the right learning rate range with a learning rate range test
    # e.g. a LR range test on random AR data (with 5e5 data, batch 64, pad_to 100) led to
    # --->  min 5e-7, max 5e-4

    if verbose:
        print("data_config\n", data_config)
        print("model_config\n", model_config)
        print("train_config\n", train_config)

    return data_config, model_config, train_config


def main(verbose=False, plot=False, save=False, random_ar_param=True):
    # load configuration dicts. Could be implemented to load from JSON instead.
    data_config, model_config, train_config = load_config(verbose, random_ar_param)
    # loads randomly generated data. Could be implemented to load a specific dataset instead.
    data = load_data(data_config, verbose, plot)
    # runs training and testing.
    results_dar, stats_dar = run_training(data, model_config, train_config, verbose)

    # optional printing
    if verbose:
        print(stats_dar)

    # optional plotting
    if plot:
        utils.plot_loss_curve(
            losses=results_dar["losses"],
            test_loss=results_dar["test_mse"],
            epoch_losses=results_dar["epoch_losses"],
            show=False,
            save=save
        )
        utils.plot_weights(
            model_config["ar"],
            results_dar["weights"],
            data["ar"],
            model_name="AR-Net",
            save=save
        )
        utils.plot_results(
            results_dar,
            model_name="AR-Net",
            save=save
        )


if __name__ == "__main__":
    main(verbose=True, plot=True, save=True, random_ar_param=False)
