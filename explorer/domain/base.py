#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Explorer                                                                            #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.11                                                                             #
# Filename   : /explorer/domain/base.py                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/explorer                                           #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday June 21st 2023 07:45:48 pm                                                #
# Modified   : Wednesday June 21st 2023 10:18:57 pm                                                #
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

from explorer import IMMUTABLE_TYPES, SEQUENCE_TYPES
from explorer.container import ExplorerContainer


# ------------------------------------------------------------------------------------------------ #
class Analysis(ABC):
    """Encapsulates the data, visualizations, descriptive and inferential statistics for an analysis."""

    @abstractmethod
    def set_data(self, data: Data) -> Analysis:
        """Sets the data for an analysis object."""

    @abstractmethod
    def add_visual(self, visual: Visual) -> Analysis:
        """Sets the data for an analysis object."""

    @abstractmethod
    def add_test(self, test: StatTestResult) -> Analysis:
        """Sets the data for an analysis object."""


# ------------------------------------------------------------------------------------------------ #
@dataclass
class Data(ABC):
    """Encapsulates the data used in an analysis"""

    dataframe: pd.DataFrame
    x: str
    y: str = None
    z: str = None

    @abstractmethod
    def summary(self) -> pd.DataFrame:
        """Summarizes the data at the dataset level."""

    @abstractmethod
    def info(self) -> pd.DataFrame:
        """Immulates the pandas DataFrame info method"""

    @abstractmethod
    def describe(self) -> pd.DataFrame:
        """Reports descriptive statistics for the DataFrame."""

    @abstractmethod
    def head(self, n: int = 5) -> pd.DataFrame:
        """Returns the top n rows from the DataFrame."""


# ------------------------------------------------------------------------------------------------ #
class Visual(ABC):
    """Specifies a visualization object."""

    @abstractmethod
    def config(self, data: Data, config: Config) -> None:
        """Configures the Visualization"""

    @abstractmethod
    def plot(self) -> None:
        """Renders the visualization"""


# ------------------------------------------------------------------------------------------------ #
#                               CONFIGURATION BASE CLASS                                           #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class Config(ABC):
    """Abstract base class for Configuration data classes."""

    def as_dict(self) -> dict:
        """Returns a dictionary representation of the the Legend object."""
        return {k: self._export_config(v) for k, v in self.__dict__.items()}

    @classmethod
    def _export_config(cls, v):  # pragma: no cover
        """Returns v with Configs converted to dicts, recursively."""
        if isinstance(v, IMMUTABLE_TYPES):
            return v
        elif isinstance(v, SEQUENCE_TYPES):
            return type(v)(map(cls._export_config, v))
        elif isinstance(v, datetime):
            return v
        elif isinstance(v, dict):
            return v
        elif hasattr(v, "as_dict"):
            return v.as_dict()
        else:
            return v


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

    def __post_init__(self, stats_config=Provide[ExplorerContainer.stats_config]) -> None:
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
