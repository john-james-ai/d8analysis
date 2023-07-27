#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Enter Project Name in Workspace Settings                                            #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.11                                                                             #
# Filename   : /edation/domain/base.py                                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : Enter URL in Workspace Settings                                                     #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday June 21st 2023 07:45:48 pm                                                #
# Modified   : Thursday July 27th 2023 08:51:36 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Base Module for the Analysis Package"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from datetime import datetime
import logging

from dependency_injector.wiring import inject, Provide
import pandas as pd


from edation.container import EdationContainer


# ------------------------------------------------------------------------------------------------ #
#                                        DATA                                                      #
# ------------------------------------------------------------------------------------------------ #
class Data(ABC):
    """Encapsulates the data used in an analysis"""

    @property
    @abstractmethod
    def metadata(self) -> dict:
        """Returns metadata that identifies and distinguishes the dataset."""

    @property
    @abstractmethod
    def summary(self) -> pd.DataFrame:
        """Provides summary metrics at the dataset level."""

    @property
    @abstractmethod
    def info(self) -> pd.DataFrame:
        """Immulates the pandas DataFrame info method"""

    @abstractmethod
    def describe(self, include: list = None, exclude: list = None) -> pd.DataFrame:
        """Reports descriptive statistics for the DataFrame.

        Args:
            include (list) Optional list of dtypes to include in the analysis.
            exclude (list) Optional list of dtypes to exclude from the analysis.
        """

    @abstractmethod
    def head(self, n: int = 5) -> pd.DataFrame:
        """Returns the top n rows from the DataFrame."""


# ------------------------------------------------------------------------------------------------ #
#                                        PLOT                                                      #
# ------------------------------------------------------------------------------------------------ #
class Plot(ABC):
    """Abstraction for classes that encapsulate plots."""

    @abstractmethod
    def fit(
        self,
        dataset: pd.DataFrame,
        x: str,
        y: str = None,
        z: str = None,
        *args,
        **kwargs,
    ) -> Plot:
        """Fits the data to the plot

        Args:
            dataset (pd.DataFrame): DataFrame containing the data to be plotted.
            x: (str): Name of the column in the dataset to be plotted along the x-axis.
            y: (str): Optional name of the column in the dataset to be plotted along the y-axis.
            z: (str): Optional name of the column to be used as a third dimension.
            canvas (Canvas): Optional Canvas object upon which the plot will be rendered.
                If a Canvas is not provided, one will be created.
            *args, **kwargs: Positional and keyword arguments of subclasses.

        """

    @abstractmethod
    def show(self) -> None:
        """Renders the plot."""

    @abstractmethod
    def export(self, filepath: str) -> None:
        """Saves plot as an image to file."""


# ------------------------------------------------------------------------------------------------ #
#                                       VISUAL                                                     #
# ------------------------------------------------------------------------------------------------ #
class Visual(ABC):
    """Specifies a visualization object."""

    def __init__(self, data: Data, config: Config) -> None:
        self._data = data
        self._config = config
        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    @abstractmethod
    def add_plot(self, name: str, plot: Plot) -> None:
        """Adds a Plot object to the Visual
        Args:
            name (str): Unique name for plot in the visual
            plot (Plot): A Plot object.
        """

    @abstractmethod
    def show(self) -> None:
        """Renders the visualization"""


# ------------------------------------------------------------------------------------------------ #
#                          STATISTICAL TEST PROFILE                                                #
# ------------------------------------------------------------------------------------------------ #
@inject
@dataclass
class StatTestProfile(ABC):
    """Abstract base class specifying the parameters for the statistical test."""

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

    def __post_init__(self, stats_config=Provide[EdationContainer.stats_config]) -> None:
        self._stats_config = stats_config

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

        profile = cls._stats_config[id]
        fieldlist = {f.name for f in fields(cls) if f.init}
        filtered_dict = {k: v for k, v in profile.items() if k in fieldlist}
        filtered_dict["id"] = id
        return cls(**filtered_dict)


# ------------------------------------------------------------------------------------------------ #
#                              STATISTICAL TEST ABC                                                #
# ------------------------------------------------------------------------------------------------ #
@inject
class StatisticalTest(ABC):
    def __init__(self) -> None:
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
#                               STATISTICAL TEST RESULT                                            #
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
