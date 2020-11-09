import os
import torch

## lazy imports ala fastai2 style (for nice print functionality)
from fastai.basics import *
from fastai.tabular.all import *

## explicit imports (for reference)
# from fastai.basics import Callback
# from fastai.data.core import DataLoaders
# from fastai.learner import Metric
# from fastai.metrics import mse, mae
# from fastai.tabular.core import TabularPandas, TabDataLoader
# from fastai.tabular.learner import tabular_learner
# from fastai.torch_core import to_detach
# from fastai.data.transforms import Normalize

## import arnet
from arnet.make_dataset import load_from_file, tabularize_univariate
from arnet.utils import pad_ar_params, estimate_noise, split_by_p_valid, nice_print_list, compute_sTPE, coeff_from_model
from arnet.plotting import plot_weights, plot_prediction_sample, plot_error_scatter


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
    if sparsity is not None and spparsity == 1.0:
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
