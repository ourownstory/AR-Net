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

## lazy imports ala fastai2 style (needed for nice print functionality)
# from fastai.basics import *
# from fastai.tabular.all import *

import arnet

log = logging.getLogger("ARNet.test")
log.setLevel("WARNING")
log.parent.setLevel("WARNING")

DIR = pathlib.Path(__file__).parent.parent.absolute()
data_path = os.path.join(DIR, "ar_data")
results_path = os.path.join(data_path, "results_test")
EPOCHS = 2


class IntegrationTests(unittest.TestCase):
    verbose = False
    plot = False
    save = False

    def test_random_data(self):
        df = pd.DataFrame({"x": [random.gauss(0.0, 1.0) for i in range(1000)]})
        m = arnet.ARNet(
            ar_order=3,
            n_epoch=3,
        )
        m = m.tabularize(df)
        m = m.make_datasets()
        m = m.create_learner(
            sparsity=0.3,
        )
        m = m.find_lr(plot=True)
        m = m.fit(cycles=3, plot=True)
        log.info("coeff of random data: {}".format(m.coeff))

    def test_ar_data(self):
        self.save = True
        if self.save:
            if not os.path.exists(results_path):
                os.makedirs(results_path)

        data_name = "ar_3_ma_0_noise_0.100_len_10000"
        df, data_config = arnet.load_from_file(data_path, data_name, load_config=True, verbose=self.verbose)
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
        m = m.fit_with_defaults(series=df)

        # Look at Coeff
        log.info("ar params: {}".format(arnet.nice_print_list(ar_params)))
        log.info("model weights: {}".format(arnet.nice_print_list(m.coeff)))

        # should be [0.20, 0.30, -0.50, ...]

        preds, y = m.learn.get_preds()
        if self.plot or self.save:
            if self.plot:
                m.learn.recorder.plot_loss()
            arnet.plot_weights(
                ar_val=len(ar_params[0]), weights=m.coeff[0], ar=ar_params[0], save=not self.plot, savedir=results_path
            )
            arnet.plot_prediction_sample(preds, y, num_obs=100, save=not self.plot, savedir=results_path)
            arnet.plot_error_scatter(preds, y, save=not self.plot, savedir=results_path)

        # if self.save:
        #     # Optional:save and create inference learner
        #     learn.freeze()
        #     model_name = "ar{}_sparse_{:.3f}_ahead_{}_epoch_{}.pkl".format(ar_order, sparsity, n_forecasts, n_epoch)
        #     learn.export(fname=os.path.join(results_path, model_name))
        #     # can be loaded like this
        #     infer = load_learner(fname=os.path.join(results_path, model_name), cpu=True)
        # # can unfreeze the model and fine_tune
        # learn.unfreeze()
        # learn.fit_one_cycle(1, lr_at_min / 100)
        #
        # coeff2 = arnet.coeff_from_model(learn.model)
        # log.info("ar params", arnet.nice_print_list(ar_params))
        # log.info("model weights", arnet.nice_print_list(coeff))
        # log.info("model weights2", arnet.nice_print_list(coeff2))

        # if self.plot or self.save:
        #     if self.plot:
        #         learn.recorder.plot_loss()
        #     arnet.plot_weights(
        #         ar_val=len(ar_params[0]), weights=coeff2[0], ar=ar_params[0], save=not self.plot, savedir=results_path
        #     )
        # if self.save:
        #     shutil.rmtree(results_path)
