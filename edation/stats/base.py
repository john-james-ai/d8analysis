#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Enter Project Name in Workspace Settings                                            #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.10                                                                             #
# Filename   : /edation/stats/base.py                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : Enter URL in Workspace Settings                                                     #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday June 5th 2023 12:13:09 am                                                    #
# Modified   : Thursday July 27th 2023 01:05:12 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass, fields
from typing import Union

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

from edation.service.io import IOService
from edation import IMMUTABLE_TYPES, SEQUENCE_TYPES
from edation.visual.config import Canvas

# ------------------------------------------------------------------------------------------------ #
sns.set_style(Canvas.style)
# ------------------------------------------------------------------------------------------------ #
ANALYSIS_TYPES = {
    "univariate": "Univariate",
    "bivariate": "Bivariate",
    "multivariate": "Multivariate",
}
STAT_CONFIG = "config/stats.yml"


# ------------------------------------------------------------------------------------------------ #
@dataclass
class Description(ABC):
    """Descriptive statistics for distributions"""

    count: int  # Non-Nulls only
    length: int  # Total length of iterable.
    size: int


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
    interpretation: str = None

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

    def _fill_reject_region(
        self,
        ax: plt.Axes,
        lower: float,
        upper: float,
        lower_critical: float,
        upper_critical: float,
        dof: int,
    ) -> plt.Axes:
        """Fills the area under the curve at the value of the hypothesis test statistic."""

        # Fill lower tail
        xlower = np.arange(lower, lower_critical, 0.001)
        ax.fill_between(x=xlower, y1=0, y2=stats.t.pdf(xlower, dof), color=Canvas.colors.orange)

        # Fill Upper Tail
        xupper = np.arange(upper_critical, upper, 0.001)
        ax.fill_between(x=xupper, y1=0, y2=stats.t.pdf(xupper, dof), color=Canvas.colors.orange)

        # Plot the statistic
        line = ax.lines[0]
        xdata = line.get_xydata()[:, 0]
        ydata = line.get_xydata()[:, 1]
        try:
            idx = np.where(xdata > self.value)[0][0]
            x = xdata[idx]
            y = ydata[idx]
            ax = sns.regplot(
                x=np.array([x]),
                y=np.array([y]),
                scatter=True,
                fit_reg=False,
                marker="o",
                scatter_kws={"s": 100},
                ax=ax,
                color=Canvas.colors.dark_blue,
            )
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

    def _report_pvalue(self, pvalue: float) -> str:
        """Rounds the pvalue in accordance with the APA Style Guide 7th Edition"""
        if pvalue < 0.001:
            return "p<.001"
        else:
            return "P=" + str(round(pvalue, 3))

    def _report_alpha(self) -> str:
        a = int(self._alpha * 100)
        return f"significant at {a}%."


# ------------------------------------------------------------------------------------------------ #
class StatisticalTestTwo(StatisticalTest):
    """Abstract base class for bivariate statistial tests."""

    def _parse_arguments(
        self, data: pd.DataFrame, x: Union[np.ndarray, str], y: Union[np.ndarray, str]
    ) -> None:
        """Parses arguments into a 2 column DataFrame and x,y column names"""
        msg = ""
        if isinstance(data, pd.DataFrame):
            if data.shape[1] == 2:
                x, y = data.columns
                return data, x, y
            elif data.shape[1] > 2:
                if self._is_column(data, x) and self._is_column(data, y) and x != y:
                    return data, x, y
                else:
                    msg += "Arguments are ambiguous. If data contains more than two columns, x and y must reference columns in data."

        elif data is None:
            if self._is_iterable(x) and self._is_iterable(y):
                return self._make_df(x, y)
            else:
                msg += "Arguments are ambiguous. Data must be a dataframe and x,y column names, or x and y must be array-like."
        else:
            msg += "Arguments are ambiguous. Data must be a dataframe and x,y column names, or x and y must be array-like."

        if len(msg) > 0:
            self._logger.error(msg)
            raise ValueError(msg)

    def _make_df(self, x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """Converts two arrays to a DataFrame"""
        d = {"Sample 1": x, "Sample 2": y}
        df = pd.DataFrame(data=d)
        x = "Sample 1"
        y = "Sample 2"
        return df, x, y

    def _is_iterable(self, a) -> bool:
        return type(a) in SEQUENCE_TYPES or isinstance(a, pd.Series)

    def _is_column(self, data: pd.DataFrame, a: str) -> bool:
        if not isinstance(a, str):
            return False
        else:
            return a in data.columns
