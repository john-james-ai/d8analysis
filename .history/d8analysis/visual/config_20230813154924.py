#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Exploratory Data Analysis Framework                                                 #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.11                                                                             #
# Filename   : /d8analysis/visual/config.py                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/d8analysis                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday May 24th 2023 04:11:27 pm                                                 #
# Modified   : Sunday August 13th 2023 03:49:24 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations
from abc import ABC
import math
from dataclasses import dataclass, field
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

from d8analysis import DataClass

# ------------------------------------------------------------------------------------------------ #
plt.rcParams["font.size"] = "10"


# ------------------------------------------------------------------------------------------------ #
#                                           COLORS                                                 #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class Colors(DataClass):
    cool_black: str = "#002B5B"
    police_blue: str = "#2B4865"
    teal_blue: str = "#256D85"
    pale_robin_egg_blue: str = "#8FE3CF"
    russian_violet: str = "#231955"
    dark_cornflower_blue: str = "#1F4690"
    meat_brown: str = "#E8AA42"
    peach: str = "#FFE5B4"
    dark_blue: str = "#002B5B"
    blue: str = "#1F4690"
    orange: str = "#E8AA42"
    crimson: str = "#BA0020"

    def __post_init__(self) -> None:
        return
