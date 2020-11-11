from dataclasses import dataclass, field
import logging
import os
import pandas as pd
import matplotlib.pyplot as plt

import fastai
from fastai.data.core import DataLoaders
from fastai.tabular.core import TabularPandas, TabDataLoader
from fastai.tabular.learner import tabular_learner, TabularLearner
from fastai.data.transforms import Normalize
from fastai.learner import load_learner

from arnet import utils, utils_data, plotting
from arnet.fastai_mods import SparsifyAR, huber, sTPE, get_loss_func

log = logging.getLogger("ARNet")


@dataclass
class ARNet:
    ar_order: int
    sparsity: float = None
    n_forecasts: int = 1
    n_epoch: int = 20
    lr: float = None
    est_noise: float = None
    start_reg_pct: float = 0.0
    full_reg_pct: float = 0.5
    use_reg_noise: bool = False
    reg_c1: float = 2.0
    reg_c2: float = 2.0
    loss_func: str = "huber"
    train_bs: int = 32
    valid_bs: int = 1024
    valid_p: float = 0.1
    normalize: bool = False
    ar_params: list = None
    log_level: str = None
    callbacks: list = None
    metrics: list = None
    dls: DataLoaders = field(init=False, default=None)
    learn: TabularLearner = field(init=False, default=None)
    coeff: list = field(init=False, default=None)
    df: pd.DataFrame = field(init=False, default=None)
    regularizer: SparsifyAR = field(init=False, default=None)

    def __post_init__(self):
        if self.log_level is not None:
            utils.set_logger_level(log, self.log_level)
        self.loss_func = get_loss_func(self.loss_func)

    def tabularize(self, series):
        if self.est_noise is None:
            self.est_noise = utils_data.estimate_noise(series)
            log.info("estimated noise of series: {}".format(self.est_noise))

        df_all = utils_data.tabularize_univariate(series, self.ar_order, self.n_forecasts)

        log.debug("tabularized df")
        log.debug("df columns: {}".format(list(df_all.columns)))
        log.debug("df shape: {}".format(df_all.shape))
        # log.debug("df head(3): {}".format(df_all.head(3)))
        self.df = df_all
        return self

    def make_datasets(
        self,
        series=None,
        valid_p=None,
        train_bs=None,
        valid_bs=None,
        normalize=None,
    ):
        if series is None:
            if self.df is None:
                raise ValueError("must pass a series.")
        else:
            self.tabularize(series)
        valid_p = self.valid_p if valid_p is None else valid_p
        train_bs = self.train_bs if train_bs is None else train_bs
        valid_bs = self.valid_bs if valid_bs is None else valid_bs
        normalize = self.normalize if normalize is None else normalize

        procs = []
        if normalize:
            procs.append(Normalize)

        df_all = self.df
        splits = utils_data.split_by_p_valid(valid_p, len(df_all))
        cont_names = [col for col in list(df_all.columns) if "x_" == col[:2]]
        target_names = [col for col in list(df_all.columns) if "y_" == col[:2]]
        tp = TabularPandas(
            df_all,
            procs=procs,
            cat_names=None,
            cont_names=cont_names,
            y_names=target_names,
            splits=splits,
        )
        log.debug("cont var num: {}, names: {}".format(len(tp.cont_names), tp.cont_names))

        trn_dl = TabDataLoader(tp.train, bs=train_bs, shuffle=True, drop_last=True)
        val_dl = TabDataLoader(tp.valid, bs=valid_bs)
        self.dls = DataLoaders(trn_dl, val_dl)
        log.debug("showing batch")
        log.debug("{}".format(self.dls.show_batch(show=False)))
        return self

    def create_regularizer(
        self,
        sparsity=None,
        start_reg_pct=None,
        full_reg_pct=None,
        est_noise=None,
        use_reg_noise=None,
        reg_c1=None,
        reg_c2=None,
    ):
        sparsity = self.sparsity if sparsity is None else sparsity
        start_reg_pct = self.start_reg_pct if start_reg_pct is None else start_reg_pct
        full_reg_pct = self.full_reg_pct if full_reg_pct is None else full_reg_pct
        est_noise = self.est_noise if est_noise is None else est_noise
        use_reg_noise = self.use_reg_noise if use_reg_noise is None else use_reg_noise
        reg_c1 = self.reg_c1 if reg_c1 is None else reg_c1
        reg_c2 = self.reg_c2 if reg_c2 is None else reg_c2

        self.regularizer = SparsifyAR(
            sparsity,
            est_noise=est_noise if use_reg_noise else None,
            start_pct=start_reg_pct,
            full_pct=full_reg_pct,
            c1=reg_c1,
            c2=reg_c2,
        )
        log.info("reg lam (max): {}".format(self.regularizer.lam_max))

    def create_learner(
        self,
        loss_func=None,
        metrics=None,
        ar_params=None,
        callbacks=None,
    ):
        loss_func = self.loss_func if loss_func is None else get_loss_func(loss_func)
        metrics = self.metrics if metrics is None else metrics
        ar_params = self.ar_params if ar_params is None else ar_params
        callbacks = self.callbacks if callbacks is None else callbacks

        if metrics is None:
            metrics = ["MSE", "MAE"]
            metrics = [get_loss_func(m) for m in metrics]
        if ar_params is not None:
            metrics.append(sTPE(ar_params, at_epoch_end=False))

        if callbacks is None:
            callbacks = []
        if self.sparsity is not None and self.regularizer is None:
            self.create_regularizer()
        if self.regularizer is not None:
            callbacks.append(self.regularizer)

        self.learn = tabular_learner(
            self.dls,
            layers=[],  # Note: None defaults to [200, 100]
            config={"use_bn": False, "bn_final": False, "bn_cont": False},
            n_out=self.n_forecasts,  # None calls get_c(dls)
            train_bn=False,  # passed to Learner
            metrics=metrics,  # passed to Learner
            loss_func=loss_func,
            cbs=callbacks,
        )
        log.debug("{}".format(self.learn.model))
        return self

    def find_lr(self, plot=True):
        if self.learn is None:
            raise ValueError("create learner first.")
        lr_at_min, lr_steep = self.learn.lr_find(start_lr=1e-6, end_lr=1, num_it=500, show_plot=plot)
        if plot:
            plt.show()
        log.debug("lr at minimum: {}; (steepest lr: {})".format(lr_at_min, lr_steep))
        lr = lr_at_min
        log.info("Optimal learning rate: {}".format(lr))
        self.lr = lr
        return self

    def fit(self, n_epoch=None, lr=None, cycles=1, plot=True):
        n_epoch = self.n_epoch if n_epoch is None else n_epoch
        lr = self.lr if lr is None else lr
        if lr is None:
            self.find_lr(plot=plot)
            lr = self.lr
        for i in range(0, cycles):
            self.learn.fit_one_cycle(n_epoch=n_epoch, lr_max=lr, div=25.0, div_final=10000.0, pct_start=0.25)
            lr = lr / 10
            if plot:
                self.learn.recorder.plot_loss()
        if plot:
            plt.show()
        # record Coeff
        self.coeff = utils.coeff_from_model(self.learn.model)
        return self

    def fit_with_defaults(self, series):
        self.make_datasets(series)
        self.create_learner()
        self.fit(plot=False)
        return self

    def plot_weights(self, **kwargs):
        plotting.plot_weights(
            ar_val=self.ar_order,
            weights=self.coeff[0],
            ar=self.ar_params,
            **kwargs,
        )

    def plot_fitted_obs(self, num_obs=100, **kwargs):
        preds, y = self.learn.get_preds()
        if num_obs is not None:
            y = y[0:num_obs]
            preds = preds[0:num_obs]
        plotting.plot_prediction_sample(preds, y, **kwargs)

    def plot_errors(self, **kwargs):
        preds, y = self.learn.get_preds()
        plotting.plot_error_scatter(preds, y, **kwargs)

    def save_model(self, results_path="results", model_name=None):
        # self.learn.freeze()
        sparsity = 1.0 if self.sparsity is None else self.sparsity
        if model_name is None:
            model_name = "ar{}_sparse_{:.3f}_ahead_{}_epoch_{}.pkl".format(
                self.ar_order, sparsity, self.n_forecasts, self.n_epoch
            )
        self.learn.export(fname=os.path.join(results_path, model_name))
        return self

    def load_model(self, results_path="results", model_name=None, cpu=True):
        self.learn = load_learner(fname=os.path.join(results_path, model_name), cpu=cpu)
        # can unfreeze the model and fine_tune
        self.learn.unfreeze()
        return self
