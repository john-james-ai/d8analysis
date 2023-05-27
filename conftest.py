#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Enter Project Name in Workspace Settings                                            #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.10                                                                             #
# Filename   : /conftest.py                                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : Enter URL in Workspace Settings                                                     #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday May 26th 2023 11:12:03 pm                                                    #
# Modified   : Saturday May 27th 2023 03:33:24 am                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import pytest
import pandas as pd

# ------------------------------------------------------------------------------------------------ #
DATAFILE = "data/Credit Score Classification Dataset.csv"


# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=False)
def dataset():
    return pd.read_csv(DATAFILE)
