#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Explorer                                                                            #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.10                                                                             #
# Filename   : /explorer/base.py                                                                   #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/explorer                                           #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday May 26th 2023 06:14:59 pm                                                    #
# Modified   : Saturday May 27th 2023 08:37:15 am                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List
import logging

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from explorer.visual import Canvas

# ------------------------------------------------------------------------------------------------ #
sns.set_style(Canvas.style)
sns.set_palette = sns.dark_palette(Canvas.color, reverse=True, as_cmap=True)


# ------------------------------------------------------------------------------------------------ #
@dataclass
class StatTest(ABC):
    analyzer: str
    test: str
    statistic: float
    pvalue: float


# ------------------------------------------------------------------------------------------------ #
@dataclass
class StatTestOne(StatTest):
    x: str  # Variable name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(test={self.test}, x={self.x}, statistic={self.statistic}, pvalue={self.pvalue})"

    def __str__(self) -> str:
        width = 32
        s = f"\t{'Analyzer:'.rjust(width,' ')} | {self.analyzer}\n"
        s += f"\t{'Test:'.rjust(width,' ')} | {self.test}\n"
        s += f"\t{'Variable:'.rjust(width,' ')} | {self.x}\n"
        s += f"\t{'Statistic:'.rjust(width,' ')} | {self.statistic}\n"
        s += f"\t{'p-Value:'.rjust(width,' ')} | {self.pvalue}\n"
        return s


# ------------------------------------------------------------------------------------------------ #
@dataclass
class StatTestOneGoF(StatTestOne):
    distribution: str
    method: str = "approx"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(test={self.test}, x={self.x}, statistic={self.statistic}, pvalue={self.pvalue})"

    def __str__(self) -> str:
        width = 32
        s = f"\t{'Analyzer:'.rjust(width,' ')} | {self.analyzer}\n"
        s += f"\t{'Test:'.rjust(width,' ')} | {self.test}\n"
        s += f"\t{'Distribution:'.rjust(width,' ')} | {self.distribution}\n"
        s += f"\t{'Method:'.rjust(width,' ')} | {self.method}\n"
        s += f"\t{'Variable:'.rjust(width,' ')} | {self.x}\n"
        s += f"\t{'Statistic:'.rjust(width,' ')} | {self.statistic}\n"
        s += f"\t{'p-Value:'.rjust(width,' ')} | {self.pvalue}\n"
        return s


# ------------------------------------------------------------------------------------------------ #
@dataclass
class StatTestTwo(StatTest):
    a: str
    b: str


# ------------------------------------------------------------------------------------------------ #
@dataclass
class StatTestMV(StatTest):
    vars: List = field(default_factory=lambda: [str])


# ------------------------------------------------------------------------------------------------ #
class Analysis(ABC):
    """Abstract base class for variables; types managed by subclasses."""

    def __init__(self, data: pd.DataFrame) -> None:
        self._data = data
        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    @abstractmethod
    def describe(self, *args, **kwargs) -> pd.DataFrame:
        """Computes descriptive statistics and returns a DataFrame."""

    def _check_name(self, name: str) -> None:
        """Checks name against variables in data"""
        if name not in self._data.columns:
            msg = f"Data has no attribute {name}."
            self._logger.error(msg)
            raise KeyError(msg)

    def _get_axes(self) -> plt.Axes:
        """Returns a matplotlib Axes object"""
        canvas = Canvas()
        return canvas.ax
