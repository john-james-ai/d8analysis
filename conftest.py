#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Exploratory Data Analysis Framework                                                 #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.10                                                                             #
# Filename   : /conftest.py                                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/d8analysis                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday May 26th 2023 11:12:03 pm                                                    #
# Modified   : Monday August 21st 2023 01:59:12 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os
import pytest
import pandas as pd
import subprocess
from datetime import datetime
import dotenv
import logging
from dataclasses import dataclass

import numpy as np

from d8analysis.container import D8AnalysisContainer
from d8analysis.data.dataclass import DataClass

# ------------------------------------------------------------------------------------------------ #
logging.getLogger("matplotlib").setLevel(logging.WARNING)
# ------------------------------------------------------------------------------------------------ #
DATAFILE = "data/Credit Score Classification Dataset.csv"
RESET_SCRIPT = "tests/scripts/reset.sh"

# ------------------------------------------------------------------------------------------------ #
collect_ignore_glob = []


# ------------------------------------------------------------------------------------------------ #
#                                      DATACLASS                                                   #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class TestDataClass(DataClass):
    name: str = "test"
    size: int = 8329
    length: float = 920932.98
    dt: datetime = datetime.now()
    # somelist: list = field(default_factory=lambda: [2, 3, 6])
    # sometuple: tuple = field(default=lambda: (2, "two", 2.0))
    # somedict: dict = field(default=lambda: {"some": 2, "dict": "yeah"})


# ------------------------------------------------------------------------------------------------ #
#                                   RESET TEST DB                                                  #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=False)
def reset():
    subprocess.run(RESET_SCRIPT, shell=True)


# ------------------------------------------------------------------------------------------------ #
#                                  SET MODE TO TEST                                                #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=True)
def mode():
    dotenv_file = dotenv.find_dotenv()
    dotenv.load_dotenv(dotenv_file)
    prior_mode = os.environ["MODE"]
    os.environ["MODE"] = "test"
    dotenv.set_key(dotenv_file, "MODE", os.environ["MODE"])
    yield
    os.environ["MODE"] = prior_mode


# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=False)
def dataset():
    df = pd.read_csv(DATAFILE, index_col=None)
    df = df.astype(
        {
            "Gender": "category",
            "Age": np.int64,
            "Income": np.int64,
            "Children": np.int64,
            "Marital Status": "category",
            "Credit Rating": "category",
            "Education": "category",
        }
    )
    return df


# ------------------------------------------------------------------------------------------------ #
#                              DEPENDENCY INJECTION                                                #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=True)
def container():
    container = D8AnalysisContainer()
    container.init_resources()
    container.wire(packages=["d8analysis"])


# ------------------------------------------------------------------------------------------------ #
#                                     DATACLASS                                                    #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=False)
def dataklass():
    return TestDataClass()
