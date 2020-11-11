#!/usr/bin/env python3

import unittest
import os
import pathlib
import shutil
import logging
import random
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", message=".*nonzero.*", category=UserWarning)

import arnet
from arnet.create_ar_data import generate_armaprocess_data
from arnet.create_ar_data import save_to_file, load_from_file

log = logging.getLogger("ARNet.test")
log.setLevel("WARNING")
log.parent.setLevel("WARNING")

DIR = pathlib.Path(__file__).parent.absolute()
data_path = os.path.join(DIR, "ar_data")
results_path = os.path.join(DIR, "results")
EPOCHS = 2


class UnitTests(unittest.TestCase):
    save = True

    def test_create_data_random(self):
        # option 1: Randomly generated AR parameters
        data_config = {
            "samples": 1000,
            "noise_std": 0.1,
            "ar_order": 3,
            "ma_order": 0,
            "params": None,  # for randomly generated AR params
        }
        log.debug("{}".format(data_config))

        # Generate data
        series, data_config["ar_params"], data_config["ma_params"] = generate_armaprocess_data(**data_config)

        if self.save:
            del data_config["params"]
            data_name = save_to_file(data_path, series, data_config)
            # just to test:
            df, data_config2 = load_from_file(data_path, data_name, load_config=True)
            log.debug("loaded from saved files:")
            log.debug("{}".format(data_config2))
            log.debug("{}".format(df.head().to_string))

    def test_create_data_manual(self):
        # option 1: Randomly generated AR parameters
        # option 2: Manually define AR parameters
        data_config = {
            "samples": 1000,
            "noise_std": 0.1,
            "params": ([0.2, 0.3, -0.5], []),
            #     "params": ([0.2, 0, 0.3, 0, 0, 0, 0, 0, 0, -0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], []),
        }
        data_config["ar_order"] = int(sum(np.array(data_config["params"][0]) != 0.0))
        data_config["ma_order"] = int(sum(np.array(data_config["params"][1]) != 0.0))
        log.debug("{}".format(data_config))

        # Generate data
        series, data_config["ar_params"], data_config["ma_params"] = generate_armaprocess_data(**data_config)

        if self.save:
            del data_config["params"]
            data_name = save_to_file(data_path, series, data_config)
            # just to test:
            df, data_config2 = load_from_file(data_path, data_name, load_config=True)
            log.debug("loaded from saved files:")
            log.debug("{}".format(data_config2))
            log.debug("{}".format(df.head().to_string))
