#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Explorer                                                                            #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.10                                                                             #
# Filename   : /conftest.py                                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/explorer                                           #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday May 26th 2023 11:12:03 pm                                                    #
# Modified   : Sunday May 28th 2023 05:35:00 pm                                                    #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os
import pytest
import pandas as pd
import subprocess
import dotenv

from explorer.container import ExplorerContainer

# ------------------------------------------------------------------------------------------------ #
DATAFILE = "data/Credit Score Classification Dataset.csv"
RESET_SCRIPT = "tests/scripts/reset.sh"

# ------------------------------------------------------------------------------------------------ #
collect_ignore_glob = ["*_univariate/*.py", "*_bivariate/*.py", "*statistics/*.py"]


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
    os.environ["MODE"] = "test"
    dotenv.set_key(dotenv_file, "MODE", os.environ["MODE"])


# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=False)
def dataset():
    return pd.read_csv(DATAFILE)


# ------------------------------------------------------------------------------------------------ #
#                              DEPENDENCY INJECTION                                                #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=True)
def container():
    container = ExplorerContainer()
    container.init_resources()
    container.wire(packages=["explorer.service"])

    return container
