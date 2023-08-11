#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Exploratory Data Analysis Framework                                                 #
# Version    : 0.0.9                                                                               #
# Python     : 3.10.11                                                                             #
# Filename   : /d8analysis/stats/profile.py                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/d8analysis                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday May 28th 2023 06:24:28 pm                                                    #
# Modified   : Thursday August 10th 2023 10:27:29 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Profile Module for the Statistical Analytics Package"""
from dataclasses import dataclass
from d8analysis.stats.base import StatTestProfile


# ------------------------------------------------------------------------------------------------ #
@dataclass
class StatTestProfileOne(StatTestProfile):
    X_variable_type: str = None


# ------------------------------------------------------------------------------------------------ #
@dataclass
class StatTestProfileTwo(StatTestProfile):
    X_variable_type: str = None
    Y_variable_type: str = None
