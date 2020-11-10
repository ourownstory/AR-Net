import logging

log = logging.getLogger("ARNet")
log.setLevel("INFO")
# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler("logs.log", "w+")
# c_handler.setLevel("WARNING")
# f_handler.setLevel("INFO")
# Create formatters and add it to handlers
c_format = logging.Formatter("%(levelname)s: %(name)s - %(funcName)s: %(message)s")
f_format = logging.Formatter("%(asctime)s; %(levelname)s; %(name)s; %(funcName)s; %(message)s")
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)
# Add handlers to the logger
log.addHandler(c_handler)
log.addHandler(f_handler)

from fastai.basics import *
from fastai.tabular.all import *

from .ar_net import ARNet

# from .ar_net_legacy import init_ar_learner
from .utils_data import load_from_file, tabularize_univariate, estimate_noise, split_by_p_valid
from .utils import pad_ar_params, nice_print_list, compute_sTPE, coeff_from_model
from .plotting import plot_weights, plot_prediction_sample, plot_error_scatter
