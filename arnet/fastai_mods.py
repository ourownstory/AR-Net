import logging
import torch
import torch.nn.functional as F
import fastai
from fastai.torch_core import flatten_check
from fastai.metrics import mse, mae
from fastai.basics import Callback
from fastai.learner import Metric

from arnet import utils

log = logging.getLogger("ARNet.fastai_mods")


def huber(inp, targ):
    """Huber error between `inp` and `targ`."""
    return F.smooth_l1_loss(*flatten_check(inp, targ))


class SparsifyAR(Callback):
    """Callback that adds regularization of first linear layer according to AR-Net paper"""

    def __init__(
        self,
        est_sparsity,
        est_noise=1.0,
        reg_strength=0.02,
        start_pct=0.0,
        full_pct=0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lam_max = 0.0
        if est_sparsity is not None:
            self.lam_max = reg_strength * est_noise * (1.0 / est_sparsity - 1.0)
        self.start_pct = start_pct
        self.full_pct = full_pct
        self.lam = None

    def after_loss(self):
        if not self.training:
            return
        if self.lam_max == 0 or self.lam == 0:
            return
        abs_weights = None
        for layer in self.learn.model.modules():
            if isinstance(layer, torch.nn.Linear):
                abs_weights = torch.abs(layer.weight)
                break
        if abs_weights is None:
            raise NotImplementedError("weight regualarization only implemented for model with Linear layer")
        reg = torch.div(2.0, 1.0 + torch.exp(-3.0 * abs_weights.pow(1.0 / 3.0))) - 1.0

        progress_iter = (1.0 + self.learn.iter) / (1.0 * self.learn.n_iter)
        progress = (progress_iter + self.learn.epoch) / (1.0 * self.learn.n_epoch)
        progress = (progress - self.start_pct) / (self.full_pct - self.start_pct)
        if progress <= 0:
            self.lam = 0.0
        elif progress < 1:
            self.lam = self.lam_max * progress ** 2
        else:
            self.lam = self.lam_max

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
        self.total = 0.0
        self.count = 0
        self.sTPE = None

    def reset(self):
        self.total = 0.0
        self.count = 0
        self.sTPE = None

    def accumulate(self, learn):
        self.sTPE = fastai.torch_core.to_detach(
            utils.compute_sTPE(
                est=utils.coeff_from_model(model=learn.model, reversed_weights=True),
                real=self.ar_params,
            )
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


def get_loss_func(loss_func):
    if type(loss_func) == str:
        if loss_func.lower() == "mse":
            loss_func = mse
        elif loss_func.lower() in ["huber", "smooth_l1", "smoothl1"]:
            loss_func = huber
        elif loss_func.lower() in ["mae", "l1"]:
            loss_func = mae
        else:
            log.error("loss {} not defined".format(loss_func))
            loss_func = None
    return loss_func
