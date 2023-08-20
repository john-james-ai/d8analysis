#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Exploratory Data Analysis Framework                                                 #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.12                                                                             #
# Filename   : /d8analysis/quantitative/descriptive/stats.py                                       #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/d8analysis                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday August 20th 2023 05:48:32 am                                                 #
# Modified   : Sunday August 20th 2023 05:49:37 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from dataclasses import dataclass

import pandas as pd


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DescriptiveStats:
    numeric: pd.DataFrame = None
    categorical: pd.DataFrame = None
