from dataclasses import dataclass, field
import logging

import fastai

## lazy imports ala fastai2 style (for nice print functionality)
from fastai.basics import *
from fastai.tabular.all import *

## explicit imports (for reference)
# from fastai.basics import Callback
# from fastai.data.core import DataLoaders
# from fastai.learner import Metric
# from fastai.metrics import mse, mae, huber
# from fastai.tabular.core import TabularPandas, TabDataLoader
# from fastai.tabular.learner import tabular_learner
# from fastai.torch_core import to_detach
# from fastai.data.transforms import Normalize


## import arnet
from arnet.make_dataset import load_from_file, tabularize_univariate
from arnet.utils import pad_ar_params, estimate_noise, split_by_p_valid, nice_print_list, compute_sTPE, coeff_from_model
from arnet import utils, plotting


log = logging.getLogger("ARNet")


class SparsifyAR(Callback):
    """Callback that adds regularization of first linear layer according to AR-Net paper"""

    def __init__(self, est_sparsity, est_noise=1.0, reg_strength=0.02, **kwargs):
        super().__init__(**kwargs)
        self.lam = 0.0
        if est_sparsity is not None:
            self.lam = reg_strength * est_noise * (1.0 / est_sparsity - 1.0)

    def after_loss(self):
        if not self.training:
            return
        abs_weights = None
        for layer in self.learn.model.modules():
            if isinstance(layer, torch.nn.Linear):
                abs_weights = torch.abs(layer.weight)
                break
        if abs_weights is None:
            raise NotImplementedError("weight regualarization only implemented for model with Linear layer")
        reg = torch.div(2.0, 1.0 + torch.exp(-3.0 * abs_weights.pow(1.0 / 3.0))) - 1.0
        self.learn.loss += self.lam * torch.mean(reg)

    _docs = dict(after_loss="Add regularization of first linear layer")


class sTPE(Metric):
    """ "
    Symmetrical Total Percentage Error of learned weights compared to underlying AR coefficients.
    Computed as the average over snapshots at each batch.
    """

    def __init__(self, ar_params, at_epoch_end=False):
        self.ar_params = ar_params
        self.at_epoch_end = at_epoch_end

    def reset(self):
        self.total, self.count = 0.0, 0
        self.sTPE = None

    def accumulate(self, learn):
        self.sTPE = to_detach(
            compute_sTPE(est=coeff_from_model(model=learn.model, reversed_weights=True), real=self.ar_params)
        )
        self.total += self.sTPE
        self.count += 1

    @property
    def value(self):
        if self.at_epoch_end:
            return self.sTPE
        return self.total / self.count if self.count != 0 else None

    @property
    def name(self):
        return "sTPE of AR coeff"


def init_ar_learner(
    series,
    ar_order,
    n_forecasts=1,
    valid_p=0.1,
    sparsity=None,
    ar_params=None,
    train_bs=32,
    valid_bs=128,
    verbose=False,
):
    if sparsity is not None and sparsity == 1.0:
        sparsity = None
    df_all = tabularize_univariate(series, ar_order, n_forecasts)
    est_noise = estimate_noise(series)

    if verbose:
        print("tabularized df")
        print("df columns", list(df_all.columns))
        print("df shape", df_all.shape)
        # if nested_list: print("x_dim:", len(df_all['x'][0]), "y_dim:", len(df_all['y'][0]))
        # print("df head(3)", df_all.head(3))
        print("estimated noise of series", est_noise)

    ## split
    splits = split_by_p_valid(valid_p, len(df_all), verbose)

    cont_names = [col for col in list(df_all.columns) if "x_" == col[:2]]
    target_names = [col for col in list(df_all.columns) if "y_" == col[:2]]

    ## preprocess?
    # procs = [Normalize]
    procs = []

    tp = TabularPandas(df_all, procs=procs, cat_names=None, cont_names=cont_names, y_names=target_names, splits=splits)
    if verbose:
        print("cont var num", len(tp.cont_names), tp.cont_names)
        # print(tp.iloc[0:5])

    ### next: data loader, learner
    trn_dl = TabDataLoader(tp.train, bs=train_bs, shuffle=True, drop_last=True)
    val_dl = TabDataLoader(tp.valid, bs=valid_bs)
    dls = DataLoaders(trn_dl, val_dl)

    # if verbose:
    #     print("showing batch")
    #     print(dls.show_batch(show=False))

    callbacks = []
    if sparsity is not None:
        callbacks.append(SparsifyAR(sparsity, est_noise))
        if verbose:
            print("reg lam: ", callbacks[0].lam)

    metrics = [mae]
    if ar_params is not None:
        metrics.append(sTPE(ar_params, at_epoch_end=False))

    tm_config = {"use_bn": False, "bn_final": False, "bn_cont": False}
    learn = tabular_learner(
        dls,
        layers=[],  # Note: None defaults to [200, 100]
        config=tm_config,  # None calls tabular_config()
        n_out=len(target_names),  # None calls get_c(dls)
        train_bn=False,  # passed to Learner
        metrics=metrics,  # passed on to TabularLearner, to parent Learner
        loss_func=mse,
        cbs=callbacks,
    )
    if verbose:
        print(learn.model)
    return learn


def get_loss_func(loss_func):
    if type(loss_func) == str:
        if loss_func.lower() == "mse":
            loss_func = fastai.metrics.mse
        elif loss_func.lower() in ["huber", "smooth_l1", "smoothl1"]:
            loss_func = fastai.metrics.huber
        elif loss_func.lower() in ["mae", "l1"]:
            loss_func = fastai.metrics.mae
        else:
            log.error("loss {} not defined".format(loss_func))
            loss_func = None
    return loss_func


@dataclass
class ARNet:
    ar_order: int
    n_forecasts: int = 1
    verbose: bool = False
    log_level: str = None
    est_noise: float = None
    train_bs: int = 32
    valid_bs: int = 1024
    valid_p: float = 0.1
    sparsity: float = None
    ar_params: list = None
    loss_func: str = "huber"
    n_epoch: int = 10
    lr: float = None
    dls: DataLoaders = field(init=False)
    learn: TabularLearner = field(init=False)
    coeff: list = field(init=False)

    def __post_init__(self):
        if self.log_level is not None:
            utils.set_logger_level(log, self.log_level)
        self.loss_func = get_loss_func(self.loss_func)

    def tabularize(self, series):
        if self.est_noise is None:
            self.est_noise = estimate_noise(series)
            log.info("estimated noise of series: {}".format(self.est_noise))

        df_all = tabularize_univariate(series, self.ar_order, self.n_forecasts)

        log.debug("tabularized df")
        log.debug("df columns: {}".format(list(df_all.columns)))
        log.debug("df shape: {}".format(df_all.shape))
        # log.debug("df head(3): {}".format(df_all.head(3)))
        return df_all

    def make_datasets(
        self,
        df_all,
        valid_p=None,
        train_bs=None,
        valid_bs=None,
    ):
        valid_p = self.valid_p if valid_p is None else valid_p
        train_bs = self.train_bs if train_bs is None else train_bs
        valid_bs = self.valid_bs if valid_bs is None else valid_bs

        # preprocess?
        # procs = [Normalize]
        procs = []
        # split
        splits = split_by_p_valid(valid_p, len(df_all))
        cont_names = [col for col in list(df_all.columns) if "x_" == col[:2]]
        target_names = [col for col in list(df_all.columns) if "y_" == col[:2]]
        tp = TabularPandas(
            df_all, procs=procs, cat_names=None, cont_names=cont_names, y_names=target_names, splits=splits
        )
        log.info("cont var num: {}, names: {}".format(len(tp.cont_names), tp.cont_names))
        # log.debug(tp.iloc[0:5])

        # next: data loader, learner
        trn_dl = TabDataLoader(tp.train, bs=train_bs, shuffle=True, drop_last=True)
        val_dl = TabDataLoader(tp.valid, bs=valid_bs)
        self.dls = DataLoaders(trn_dl, val_dl)
        log.debug("showing batch")
        log.debug("{}".format(self.dls.show_batch(show=False)))
        return self

    def create_learner(
        self,
        sparsity=None,
        ar_params=None,
        loss_func=None,
    ):
        sparsity = self.sparsity if sparsity is None else sparsity
        ar_params = self.ar_params if ar_params is None else ar_params
        loss_func = self.loss_func if loss_func is None else get_loss_func(loss_func)
        callbacks = []
        if sparsity is not None:
            callbacks.append(SparsifyAR(sparsity, self.est_noise))
            log.info("reg lam: {}".format(callbacks[0].lam))

        metrics = [fastai.metrics.mse, fastai.metrics.mae]
        if ar_params is not None:
            metrics.append(sTPE(ar_params, at_epoch_end=False))

        tm_config = {"use_bn": False, "bn_final": False, "bn_cont": False}
        self.learn = tabular_learner(
            self.dls,
            layers=[],  # Note: None defaults to [200, 100]
            config=tm_config,  # None calls tabular_config()
            n_out=self.n_forecasts,  # None calls get_c(dls)
            train_bn=False,  # passed to Learner
            metrics=metrics,  # passed on to TabularLearner, to parent Learner
            loss_func=self.loss_func,
            cbs=callbacks,
        )
        log.debug("{}".format(self.learn.model))
        return self

    def find_lr(self, plot=True):
        if self.learn is None:
            raise ValueError("create learner first.")
        lr_at_min, lr_steep = self.learn.lr_find(start_lr=1e-6, end_lr=1, num_it=400, show_plot=plot)
        if plot:
            plt.show()
        log.debug("lr at minimum: {}; (steepest lr: {})".format(lr_at_min, lr_steep))
        lr = lr_at_min / 10
        log.info("Optimal learning rate: {}".format(lr))
        self.lr = lr
        return self

    def fit(self, n_epoch=None, lr=None, plot=True, cycles=2):
        n_epoch = self.n_epoch if n_epoch is None else n_epoch
        lr = self.lr if lr is None else lr
        if lr is None:
            self.find_lr(plot=plot)
            lr = self.lr
        for i in range(0, cycles):
            self.learn.fit_one_cycle(n_epoch=n_epoch, lr_max=lr)
            lr = lr / 10
        if plot:
            self.learn.recorder.plot_loss()
            plt.show()
        # record Coeff
        self.coeff = utils.coeff_from_model(self.learn.model)
        return self

    def fit_with_defaults(self, series):
        self.make_datasets(self.tabularize(series))
        self.create_learner()
        self.fit(plot=False)
        return self

    def plot_weights(self, **kwargs):
        plotting.plot_weights(
            ar_val=self.ar_order,
            weights=self.coeff,
            ar=self.ar_params,
            **kwargs,
        )
