#!/usr/bin/env python3

import unittest
import os
import pathlib
import shutil
import logging
import random
import pandas as pd
import warnings

warnings.filterwarnings("ignore", message=".*nonzero.*", category=UserWarning)
import arnet

log = logging.getLogger("ARNet.test")
log.setLevel("WARNING")
log.parent.setLevel("WARNING")

DIR = pathlib.Path(__file__).parent.absolute()
data_path = os.path.join(DIR, "ar_data")
results_path = os.path.join(DIR, "results_test")
AR_FILE = "ar_3_ma_0_noise_0.100_len_1000"
EPOCHS = 2


class IntegrationTests(unittest.TestCase):
    plot = False
    save = False

    def test_random_data(self):
        df = pd.DataFrame({"x": [random.gauss(0.0, 1.0) for i in range(1000)]})
        m = arnet.ARNet(ar_order=3, n_epoch=3)
        m.tabularize(df)
        m.make_datasets()
        m.create_regularizer(sparsity=0.3)
        m.create_learner()
        m.find_lr(plot=False)
        m.fit(cycles=2, plot=False)
        log.info("coeff of random data: {}".format(m.coeff))

    def test_plot(self):
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        df = pd.DataFrame({"x": [random.gauss(0.0, 1.0) for i in range(1000)]})
        m = arnet.ARNet(ar_order=3, n_epoch=3)
        m.fit_with_defaults(series=df)
        if self.plot:
            m.learn.recorder.plot_loss()
        m.plot_weights(save=True, savedir=results_path)
        m.plot_fitted_obs(num_obs=100, save=True, savedir=results_path)
        m.plot_errors(save=True, savedir=results_path)

        shutil.rmtree(results_path)

    def test_save_load(self):
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        df = pd.DataFrame({"x": [random.gauss(0.0, 1.0) for i in range(1000)]})
        m = arnet.ARNet(ar_order=3, n_epoch=3)
        m.fit_with_defaults(series=df)

        # Optional:save and create inference learner
        sparsity = 1.0 if m.sparsity is None else m.sparsity
        model_name = "ar{}_sparse_{:.3f}_ahead_{}_epoch_{}.pkl".format(m.ar_order, sparsity, m.n_forecasts, m.n_epoch)
        m.save_model(results_path=results_path, model_name=model_name)
        # can be loaded like this
        m.load_model(results_path, model_name)
        # can unfreeze the model and fine_tune
        log.info("loaded coeff: {}".format(m.coeff))

        shutil.rmtree(results_path)

    def test_ar_data(self):
        df, data_config = arnet.load_from_file(data_path, AR_FILE, load_config=True, verbose=False)
        df = df[:1000]

        # Hyperparameters
        sparsity = 0.3
        ar_order = int(1 / sparsity * data_config["ar_order"])  # sparse AR: (for non-sparse, set sparsity to 1.0)
        ar_params = arnet.pad_ar_params([data_config["ar_params"]], ar_order, 1)  # to compute stats

        # run
        m = arnet.ARNet(
            ar_order=ar_order,
            n_epoch=EPOCHS,
            sparsity=sparsity,
            ar_params=ar_params,
        )
        m.fit_with_defaults(series=df)

        # Look at Coeff
        log.info("ar params: {}".format(arnet.nice_print_list(ar_params)))
        log.info("model weights: {}".format(arnet.nice_print_list(m.coeff)))
