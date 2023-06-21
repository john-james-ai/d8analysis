#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Explorer                                                                            #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.11                                                                             #
# Filename   : /explorer/analysis/data.py                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/explorer                                           #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday June 21st 2023 12:31:10 am                                                #
# Modified   : Wednesday June 21st 2023 01:02:32 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Data Module"""
from dataclasses import dataclass

import pandas as pd

from .base import Config


# ------------------------------------------------------------------------------------------------ #
#                                           DATA                                                   #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class Data(Config):
    dataframe: pd.DataFrame
    x: str
    y: str = None
    z: str = None
