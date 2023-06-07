#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Enter Project Name in Workspace Settings                                            #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.10                                                                             #
# Filename   : /explorer/stats/base.py                                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : Enter URL in Workspace Settings                                                     #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday June 5th 2023 12:13:09 am                                                    #
# Modified   : Wednesday June 7th 2023 05:02:53 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass, fields

import matplotlib.pyplot as plt
import numpy as np

from explorer.service.io import IOService
from explorer import IMMUTABLE_TYPES
from explorer.visual.config import Canvas

# ------------------------------------------------------------------------------------------------ #
ANALYSIS_TYPES = {
    "univariate": "Univariate",
    "bivariate": "Bivariate",
    "multivariate": "Multivariate",
}
STAT_CONFIG = "config/stats.yml"


# ------------------------------------------------------------------------------------------------ #
@dataclass
class StatTestProfile(ABC):
    """Abstract base class defining the interface for statistical tests.

    Interface inspired by: https://doc.dataiku.com/dss/latest/statistics/tests.html
    """

    id: str
    name: str = None
    description: str = None
    statistic: str = None
    analysis: str = None  # one of ANALYSIS_TYPES
    hypothesis: str = None  # One of HYPOTHESIS_TYPES
    H0: str = None
    parametric: bool = None
    min_sample_size: int = None
    assumptions: str = None
    use_when: str = None

    def __repr__(self) -> str:
        return "{}({})".format(
            self.__class__.__name__,
            ", ".join("{}={!r}".format(k, v) for k, v in self.__dict__.items()),
        )

    def __str__(self) -> str:
        s = ""
        width = 20
        for k, v in self.__dict__.items():
            s += f"\t{k.rjust(width,' ')} | {v}\n"
        return s

    @classmethod
    def create(cls, id) -> None:
        """Loads the values from the statistical tests file"""
        profiles = IOService.read(STAT_CONFIG)
        profile = profiles[id]
        fieldlist = {f.name for f in fields(cls) if f.init}
        filtered_dict = {k: v for k, v in profile.items() if k in fieldlist}
        filtered_dict["id"] = id
        return cls(**filtered_dict)


# ------------------------------------------------------------------------------------------------ #
@dataclass
class StatTestResult(ABC):
    test: str
    hypothesis: str
    H0: str
    statistic: str
    value: float
    pvalue: float
    inference: str
    alpha: float = 0.05
    result: str = None

    def __repr__(self) -> str:
        return "{}({})".format(
            self.__class__.__name__,
            ", ".join(
                "{}={!r}".format(k, v)
                for k, v in self.__dict__.items()
                if type(v) in IMMUTABLE_TYPES
            ),
        )

    def __str__(self) -> str:
        s = ""
        width = 32
        for k, v in self.__dict__.items():
            if type(v) in IMMUTABLE_TYPES:
                s += f"\t{k.rjust(width,' ')} | {v}\n"
        return s

    def _fill_curve(self, ax: plt.Axes) -> plt.Axes:
        """Fills the area under the curve at the value of the hypothesis test statistic."""

        # Obtain the lines for the x y values from the axes object.
        line = ax.lines[0]
        xdata = line.get_xydata()[:, 0]
        ydata = line.get_xydata()[:, 1]
        # Get index of first value greater than the statistic.
        try:
            idx = np.where(xdata > self.value)[0][0]
            fill_x = xdata[idx:]
            fill_y2 = ydata[idx:]
            ax.fill_between(x=fill_x, y1=0, y2=fill_y2, color=Canvas.colors.orange)
        except IndexError:
            pass
        return ax


# ------------------------------------------------------------------------------------------------ #
class StatisticalTest(ABC):
    def __init__(self, io: IOService = IOService) -> None:
        self._io = io
        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    @property
    @abstractmethod
    def profile(self) -> StatTestProfile:
        """Returns the statistical test profile."""

    @property
    @abstractmethod
    def result(self) -> StatTestResult:
        """Returns a Statistical Test Result object."""

    @abstractmethod
    def __call__(self, *args, **kwargs) -> None:
        """Performs the statistical test and creates a result object."""
