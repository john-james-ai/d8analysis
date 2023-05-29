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
# Created    : Sunday May 28th 2023 06:23:03 pm                                                    #
# Modified   : Sunday May 28th 2023 06:30:07 pm                                                    #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
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
# Modified   : Sunday May 28th 2023 10:51:23 am                                                    #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
import logging

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from explorer.service.visual import Canvas

# ------------------------------------------------------------------------------------------------ #
sns.set_style(Canvas.style)
sns.set_palette = sns.dark_palette(Canvas.color, reverse=True, as_cmap=True)


# ------------------------------------------------------------------------------------------------ #
class Analysis(ABC):
    """Abstract base class for variables; types managed by subclasses."""

    def __init__(self, data: pd.DataFrame) -> None:
        self._data = data
        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    @abstractmethod
    def describe(self, *args, **kwargs) -> pd.DataFrame:
        """Computes descriptive statistics and returns a DataFrame."""

    def _is_categorical(self, name) -> bool:
        """Checks whether the variable is categorical."""
        return self._data[name].dtype in (str, object, "category")

    def _is_numeric(self, name) -> bool:
        """Returns True if the named variable is discrete"""
        return self._data[name].dtype in (int, float)

    def _is_discrete(self, name) -> bool:
        """Returns True if the named variable is discrete"""
        return self._data[name].dtype == int

    def _is_continuous(self, name) -> bool:
        """Returns True if the named variable is continuous."""
        return self._data[name].dtype == float

    def _check_name(self, name: str) -> None:
        """Checks name against variables in data"""
        if name not in self._data.columns:
            msg = f"Data has no attribute {name}."
            self._logger.error(msg)
            raise KeyError(msg)

    def _check_dtypes(self, dtypes: dict) -> tuple:
        """Checks data types according to the dtypes mapping."""
        valid_dtype = {
            "categorical": self._is_categorical,
            "numeric": self._is_numeric,
            "discrete": self._is_discrete,
            "continuous": self._is_continuous,
        }
        for name, dtype in dtypes.items():
            if not valid_dtype[dtype](name=name):
                msg = f"Variable {name} must be a {dtype} data type."
                self._logger.error(msg)
                raise TypeError(msg)

    def _wrap_ticklabels(
        self, axis: str, axes: List[plt.Axes], fontsize: int = 8
    ) -> List[plt.Axes]:
        """Wraps long tick labels"""
        if axis.lower() == "x":
            for i, ax in enumerate(axes):
                xlabels = [label.get_text() for label in ax.get_xticklabels()]
                xlabels = [label.replace(" ", "\n") for label in xlabels]
                ax.set_xticklabels(xlabels, fontdict={"fontsize": fontsize})
                ax.tick_params(axis="x", labelsize=fontsize)

        if axis.lower() == "y":
            for i, ax in enumerate(axes):
                ylabels = [label.get_text() for label in ax.get_yticklabels()]
                ylabels = [label.replace(" ", "\n") for label in ylabels]
                ax.set_yticklabels(ylabels, fontdict={"fontsize": fontsize})
                ax.tick_params(axis="y", labelsize=fontsize)

        return axes
