#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Exploratory Data Analysis Framework                                                 #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.10                                                                             #
# Filename   : /d8analysis/quantitative/inferential/base.py                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/d8analysis                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday June 5th 2023 12:13:09 am                                                    #
# Modified   : Monday August 14th 2023 02:37:26 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime
from dataclasses import dataclass, fields
import seaborn as sns

import pandas as pd

from d8analysis.service.io import IOService
from d8analysis import IMMUTABLE_TYPES, SEQUENCE_TYPES
from d8analysis.analysis.base import Analysis, Result
from d8analysis.visual.base import Canvas

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
            ", ".join(
                "{}={!r}".format(k, v)
                for k, v in self.__dict__.items()
                if type(v) in IMMUTABLE_TYPES
            ),
        )

    def __str__(self) -> str:
        width = 32
        breadth = width * 2
        s = f"\n\n{self.__class__.__name__.center(breadth, ' ')}"
        d = self.as_dict()
        for k, v in d.items():
            if type(v) in IMMUTABLE_TYPES:
                s += f"\n{k.rjust(width,' ')} | {v}"
        s += "\n\n"
        return s

    def as_dict(self) -> dict:
        """Returns a dictionary representation of the the Config object."""
        return {
            k: self._export_config(v) for k, v in self.__dict__.items() if not k.startswith("_")
        }

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
            """Else nothing. What do you want?"""

    def as_df(self) -> pd.DataFrame:
        """Returns the project in DataFrame format"""
        d = self.as_dict()
        return pd.DataFrame(data=d, index=[0])

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
class StatTestProfileOne(StatTestProfile):
    X_variable_type: str = None


# ------------------------------------------------------------------------------------------------ #
@dataclass
class StatTestProfileTwo(StatTestProfile):
    X_variable_type: str = None
    Y_variable_type: str = None


# ------------------------------------------------------------------------------------------------ #
@dataclass
class StatTestResult(Result):
    test: str = None
    hypothesis: str = None
    H0: str = None
    statistic: str = None
    value: float = 0
    pvalue: float = 0
    inference: str = None
    alpha: float = 0.05
    result: str = None
    interpretation: str = None

    def __post_init__(self, canvas: Canvas) -> None:
        self._canvas = canvas
        sns.set_style(self._canvas.style)
        sns.set_palette(self._canvas.palette)


# ------------------------------------------------------------------------------------------------ #
class StatisticalTest(Analysis):
    """Base class for Statistical Tests"""

    def __init__(self, io: IOService = IOService, *args, **kwargs) -> None:
        super().__init__()
        self._io = io

    @property
    @abstractmethod
    def profile(self) -> StatTestProfile:
        """Returns the statistical test profile."""

    @property
    @abstractmethod
    def result(self) -> StatTestResult:
        """Returns a Statistical Test Result object."""

    @abstractmethod
    def run(self) -> None:
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
