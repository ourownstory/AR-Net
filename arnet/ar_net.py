import os
## lazy imports ala fastai2 style (for nice print functionality)
from fastai2.basics import *
from fastai2.tabular.all import *
## explicit imports (for reference)
# from fastai2.basics import Callback
# from fastai2.data.core import DataLoaders
# from fastai2.learner import Metric
# from fastai2.metrics import mse, mae
# from fastai2.tabular.core import TabularPandas, TabDataLoader
# from fastai2.tabular.learner import tabular_learner
# from fastai2.torch_core import to_detach
# from fastai2.data.transforms import Normalize
## import my own code
from make_dataset import load_from_file, tabularize_univariate
from utils import pad_ar_params, estimate_noise, split_by_p_valid, nice_print_list, compute_sTPE, coeff_from_model
from plotting import plot_weights, plot_prediction_sample, plot_error_scatter


class SparsifyAR(Callback):
    """Callback that adds regularization of first linear layer according to AR-Net paper"""
    def __init__(self, est_sparsity, est_noise=1.0, reg_strength=0.02):
        self.lam = 0.0
        if est_sparsity is not None:
            self.lam = reg_strength * est_noise * (1.0 / est_sparsity - 1.0)
    def after_loss(self):
        if not self.training: return
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
    """"
    Symmetrical Total Percentage Error of learned weights compared to underlying AR coefficients.
    Computed as the average over snapshots at each batch.
    """
    def __init__(self, ar_params, at_epoch_end=False):
        self.ar_params = ar_params
        self.at_epoch_end = at_epoch_end
    def reset(self):
        self.total, self.count = 0., 0
        self.sTPE = None
    def accumulate(self, learn):
        self.sTPE = to_detach(compute_sTPE(
            est=coeff_from_model(model=learn.model, reversed_weights=True),
            real=self.ar_params
        ))
        self.total += self.sTPE
        self.count += 1
    @property
    def value(self):
        if self.at_epoch_end:
            return self.sTPE
        return self.total/self.count if self.count != 0 else None
    @property
    def name(self):  return "sTPE of AR coeff"


def init_ar_learner(series, ar_order, n_forecasts=1, valid_p=0.1, sparsity=None, ar_params=None, verbose=False):
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
    # procs = [Normalize] # Note: not AR if normalized, need to unnormalize then again.
    procs = []

    tp = TabularPandas(
        df_all,
        procs=procs,
        cat_names=None,
        cont_names=cont_names,
        y_names=target_names,
        splits=splits
    )
    if verbose:
        print("cont var num", len(tp.cont_names), tp.cont_names)
        # print(tp.iloc[0:5])

    ### next: data loader, learner
    trn_dl = TabDataLoader(tp.train, bs=32, shuffle=True, drop_last=True)
    val_dl = TabDataLoader(tp.valid, bs=128)
    dls = DataLoaders(trn_dl, val_dl)

    # if verbose:
    #     print("showing batch")
    #     print(dls.show_batch(show=False))

    callbacks = []
    if sparsity is not None:
        callbacks.append(SparsifyAR(sparsity, est_noise))
        if verbose: print("reg lam: ", callbacks[0].lam)

    metrics = [mae]
    if ar_params is not None:
        metrics.append(sTPE(ar_params, at_epoch_end=False))

    learn = tabular_learner(
        dls,
        layers=[],  # Note: None defaults to [200, 100]
        # config = None, # None calls tabular_config()
        n_out=len(target_names),  # None calls get_c(dls)
        use_bn=False,  # passed to TabularModel
        bn_final=False,  # passed to TabularModel
        bn_cont=False,  # passed to TabularModel
        metrics=metrics,  # passed on to TabularLearner, to parent Learner
        loss_func=mse,
        cbs=callbacks
    )
    if verbose:
        print(learn.model)
    return learn


def main(verbose=False):
    """example of how to use AR-Net"""
    save = True
    created_ar_data = True

    data_path = 'ar_data'
    data_name = 'ar_3_ma_0_noise_0.100_len_10000'
    results_path = 'results'

    ## Load data
    if created_ar_data:
        ## if created AR data with create_ar_data, we can use the helper function:
        df, data_config = load_from_file(data_path, data_name, load_config=True, verbose=verbose)
    else:
        ## else we can manually load any file that stores a time series, for example:
        df = pd.read_csv(os.path.join(data_path, data_name + '.csv'), header=None, index_col=False)

    #### Hyperparameters
    n_epoch = 20
    valid_p = 0.5
    n_forecasts = 1  # Note: must have a list of ar_param for each forecast target.
    sparsity = 0.3  # guesstimate
    ar_order = 10  # guesstimate

    ar_params = None
    ## if we know the true AR parameters:
    if created_ar_data:
        ## for normal AR:
        ar_order = int(data_config["ar_order"])
        ## for sparse AR:
        ar_order = int(1 / sparsity * data_config["ar_order"])
        ## to compute stats
        ar_params = pad_ar_params([data_config["ar_params"]], ar_order, n_forecasts)

    #### Init Model
    learn = init_ar_learner(
        series=df,
        ar_order=ar_order,
        n_forecasts=n_forecasts,
        valid_p=valid_p,
        sparsity=sparsity,
        ar_params=ar_params,
        verbose=verbose,
    )

    #### Run Model
    ## find Learning Rate
    lr_at_min, lr_steepest = learn.lr_find(start_lr=1e-6, end_lr=1e2, num_it=400, show_plot=False)
    if verbose: print("lr_at_min", lr_at_min)
    ## if you know the best learning rate:
    # learn.fit(n_epoch, 1e-2)
    ## else use onecycle
    learn.fit_one_cycle(n_epoch=n_epoch, lr_max=lr_at_min/10)

    learn.recorder.plot_loss()

    #### Analysis of results
    coeff = coeff_from_model(learn.model)
    if created_ar_data: print("ar params", nice_print_list(ar_params))
    print("model weights", nice_print_list(coeff))
    # should be [0.20, 0.30, -0.50, ...]

    if created_ar_data:
        plot_weights(
            ar_val=len(ar_params[0]),
            weights=coeff[0],
            ar=ar_params[0],
            save=save,
            savedir=results_path
        )
    preds, y = learn.get_preds()
    plot_prediction_sample(preds, y, num_obs=100, save=save, savedir=results_path)
    plot_error_scatter(preds, y, save=save, savedir=results_path)

    #### Optional: save and create inference learner
    if save:
        learn.freeze()
        if not os.path.exists(results_path): os.makedirs(results_path)
        model_name = "ar{}_sparse_{:.3f}_ahead_{}_epoch_{}.pkl".format(ar_order, sparsity, n_forecasts, n_epoch)
        learn.export(fname=os.path.join(results_path, model_name))
        ## can be loaded like this
        infer = load_learner(fname=os.path.join(results_path, model_name), cpu=True)

    #### Optional: can unfreeze the model and fine_tune
    learn.unfreeze()
    learn.fit_one_cycle(10, lr_at_min/100)
    ## check if improved
    coeff2 = coeff_from_model(learn.model)
    if created_ar_data: print("ar params", nice_print_list(ar_params))
    print("model weights", nice_print_list(coeff))
    print("model weights2", nice_print_list(coeff2))

if __name__ == '__main__':
    main(verbose=True)
