#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Data Exploration Framework                                                          #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.11                                                                             #
# Filename   : /edation/data/credit_score.py                                                       #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/edation                                            #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday August 10th 2023 08:52:00 pm                                               #
# Modified   : Thursday August 10th 2023 09:06:38 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import pandas as pd

from edation.data.base import Dataset


# ------------------------------------------------------------------------------------------------ #
class CreditScoreDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__(df=df)

    def __getitem__(self, idx: int) -> pd.Series:
        return self._df.iloc[idx]

    def summary(self) -> pd.DataFrame:
        return (
            self._df[["Gender", "Education", "Marital Status", "Own", "Credit Rating"]]
            .value_counts()
            .reset_index()
        )
