#!/usr/bin/env python3

import unittest
import os
import pathlib
import logging
from arnet import init_ar_learner

log = logging.getLogger("AR-Net.test")
log.setLevel("WARNING")
log.parent.setLevel("WARNING")

DIR = pathlib.Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(DIR, "example_data")
AIR_FILE = os.path.join(DATA_DIR, "air_passengers.csv")
EPOCHS = 5


class IntegrationTests(unittest.TestCase):
    plot = False

    def test_fakie(self):
        pass
