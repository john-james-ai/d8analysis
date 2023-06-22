#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Explorer                                                                            #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.11                                                                             #
# Filename   : /explorer/domain/data.py                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/explorer                                           #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday June 21st 2023 09:43:24 pm                                                #
# Modified   : Wednesday June 21st 2023 11:08:35 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from dataclasses import dataclass
from .base import Data

import pandas as pd


# ------------------------------------------------------------------------------------------------ #
@dataclass
class Dataset(Data):
    """Encapsulates the data used in an analysis"""

    dataframe: pd.DataFrame
    x: str
    y: str = None
    z: str = None

    @property
    def summary(self) -> dict:
        """Summarizes the data at the dataframe level."""
        d = {}
        d["Rows"] = self.dataframe.shape[0]
        d["Cols"] = self.dataframe.shape[1]
        d["Cells"] = d["Rows"] * d["Cols"]
        d["Zeros"] = self.dataframe[self.dataframe == 0].count(axis=0).sum()
        d["Pct Zeros"] = round(
            self.dataframe[self.dataframe == 0].count(axis=0).sum() / d["Cells"] * 100, 2
        )
        d["Non-Nulls"] = self.dataframe.count(axis=0).sum()
        d["Nulls"] = self.dataframe.isna().sum().sum()
        d["Pct Null"] = round(self.dataframe.isna().sum().sum() / d["Cells"] * 100, 2)
        d["Duplicate Rows"] = len(self.dataframe.duplicated())
        d["Size"] = self.dataframe.memory_usage(deep=True)
        return d

    @property
    def info(self) -> pd.DataFrame:
        """Immulates the pandas DataFrame info method"""

        info = self.dataframe.dtypes.to_frame()
        info.columns = ["Dtypes"]
        info["Zeros"] = self.dataframe[self.dataframe == 0].count(axis=0)
        info["Pct Zeros"] = round(
            self.dataframe[self.dataframe == 0].count(axis=0) / self.dataframe.shape[0] * 100, 2
        )
        info["Non-Nulls"] = self.dataframe.count(axis=0)
        info["Nulls"] = self.dataframe.shape[0] - info["Non-Nulls"]
        info["Pct Null"] = round(info["Nulls"] / self.dataframe.shape[0] * 100, 2)
        info["Cardinality"] = self.dataframe.nunique()
        info["Pct Unique"] = round(
            self.dataframe.nunique(axis=0) / self.dataframe.shape[0] * 100, 2
        )
        info["Size"] = self.dataframe.memory_usage(deep=True).sum()
        return info

    @property
    def describe(self) -> pd.DataFrame:
        """Reports descriptive statistics for the DataFrame."""
        return self.dataframe.describe().T

    def head(self, n: int = 5) -> pd.DataFrame:
        """Returns the top n rows from the DataFrame."""
        return self.dataframe.head(n)

    def sample(self, n: int = 5) -> pd.DataFrame:
        """Returns a random sample from the DataFrame of size n or frac * m.

        Args:
            n (int): The number of rows to sample
        """

        return self.dataframe.sample(n=n)
