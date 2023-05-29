#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Explorer                                                                            #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.11                                                                             #
# Filename   : /explorer/stats/base.py                                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/explorer                                           #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday May 28th 2023 06:24:28 pm                                                    #
# Modified   : Sunday May 28th 2023 06:27:03 pm                                                    #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Stats Base Module"""
from abc import ABC
from dataclasses import dataclass


# ------------------------------------------------------------------------------------------------ #
@dataclass
class StatTestResult(ABC):
    analyzer: str
    test: str
    statistic: float
    pvalue: float
