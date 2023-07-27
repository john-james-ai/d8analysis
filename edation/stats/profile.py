#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Enter Project Name in Workspace Settings                                            #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.11                                                                             #
# Filename   : /edation/stats/profile.py                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : Enter URL in Workspace Settings                                                     #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday May 28th 2023 06:24:28 pm                                                    #
# Modified   : Thursday July 27th 2023 08:48:10 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Profile Module for the Statistical Analytics Package"""
from dataclasses import dataclass
from edation.stats.base import StatTestProfile


# ------------------------------------------------------------------------------------------------ #
@dataclass
class StatTestProfileOne(StatTestProfile):
    X_variable_type: str = None


# ------------------------------------------------------------------------------------------------ #
@dataclass
class StatTestProfileTwo(StatTestProfile):
    X_variable_type: str = None
    Y_variable_type: str = None
