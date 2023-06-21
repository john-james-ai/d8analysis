#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Explorer                                                                            #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.11                                                                             #
# Filename   : /explorer/analysis/base.py                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/explorer                                           #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday June 21st 2023 12:36:17 am                                                #
# Modified   : Wednesday June 21st 2023 02:06:01 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
from datetime import datetime

from explorer import IMMUTABLE_TYPES, SEQUENCE_TYPES


# ------------------------------------------------------------------------------------------------ #
#                           ANALYSIS COMPONENT CONFIGURATION BASE CLASS                            #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class Config(ABC):
    """Abstract base class for analysis component configurations.

    Analysis objects have five primary component types:
    1. id: The name, creation datetime, execution datetime.
    2. Data: The DataFrame and the variable names to include in the analysis
    3. Canvas: The canvas upon which the analysis is built.
    4. Plots: The plots rendered upon matplotlib axes on the canvas
    5. Tables: Tabular data to be rendered on the canvas.

    These components are stored in dataclasses descending from this Config object.

    """

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
#                                   ANALYSIS BUILDER                                               #
# ------------------------------------------------------------------------------------------------ #
class AnalysisBuilder(ABC):
    """Abstract base class defining the interface for analysis objects"""

    @property
    def analysis(self) -> Analysis:
        """Returns the analysis object."""

    @abstractmethod
    def name(self, name: str) -> AnalysisBuilder:
        """Name of the analyis"""

    @abstractmethod
    def set_data(self, data: Config) -> AnalysisBuilder:
        """Data for the analysis"""

    @abstractmethod
    def add_plot(self, plot_config: Config) -> AnalysisBuilder:
        """Adds a plot configuration to the analysis"""

    @abstractmethod
    def add_test_config(self, test_config: Config) -> AnalysisBuilder:
        """Visualization for the analysis. Can be called multiple times to"""

    @abstractmethod
    def validate(self) -> AnalysisBuilder:
        """Validates the analysis"""

    @abstractmethod
    def build(self) -> AnalysisBuilder:
        """Constructs the Analysis object."""


# ------------------------------------------------------------------------------------------------ #
#                                       ANALYSIS                                                   #
# ------------------------------------------------------------------------------------------------ #
class Analysis(ABC):
    """Analysis object"""

    @abstractmethod
    def run(self) -> None:
        """Executes the analysis."""
