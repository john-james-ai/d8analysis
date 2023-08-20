#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Exploratory Data Analysis Framework                                                 #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.11                                                                             #
# Filename   : /d8analysis/data/credit.py                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/d8analysis                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday August 10th 2023 08:52:00 pm                                               #
# Modified   : Sunday August 20th 2023 12:02:21 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations
from dataclasses import dataclass

import pandas as pd

from d8analysis.data.dataset import Dataset
from d8analysis.data.entity import Entity


# ------------------------------------------------------------------------------------------------ #
@dataclass
class Consumer(Entity):
    Gender: str
    Age: int
    Children: int
    MaritalStatus: str
    Own: str
    Education: int
    Income: float
    CreditRating: str

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> Consumer:
        return cls(
            Gender=df["Gender"],
            Age=df["Age"],
            Children=df["Children"],
            MaritalStatus=df["Marital Status"],
            Own=df["Own"],
            Education=df["Education"],
            Income=df["Income"],
            CreditRating=df["Credit Rating"],
        )


# ------------------------------------------------------------------------------------------------ #
class CreditScoreDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__(df=df)

    def __getitem__(self, idx: int) -> pd.Series:
        df = self._df.iloc[idx]
        return Consumer.from_df(df=df)

    @property
    def summary(self) -> pd.DataFrame:
        return (
            self._df[["Gender", "Education", "Marital Status", "Own", "Credit Rating"]]
            .value_counts()
            .reset_index()
        )
