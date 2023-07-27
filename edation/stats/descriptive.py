#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Enter Project Name in Workspace Settings                                            #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.10                                                                             #
# Filename   : /edation/stats/descriptive.py                                                       #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : Enter URL in Workspace Settings                                                     #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday June 8th 2023 02:56:56 am                                                  #
# Modified   : Thursday July 27th 2023 08:50:52 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from abc import ABC
from dataclasses import dataclass
from typing import Union

import pandas as pd
import numpy as np
from scipy import stats

# ------------------------------------------------------------------------------------------------ #


@dataclass
class VarStats(ABC):
    length: int  # total length of variable
    count: int  # Non-null count
    size: int  # Size in memory

    @classmethod
    def compute(cls, x: Union[pd.Series, np.ndarray]) -> None:
        return cls(length=len(x), count=len(list(filter(None, x))), size=x.__sizeof__())


# ------------------------------------------------------------------------------------------------ #
@dataclass
class QuantStats(VarStats):
    mean: float
    std: float
    var: float
    min: float
    q25: float
    median: float
    q75: float
    max: float
    range: float
    skew: float
    kurtosis: float

    @classmethod
    def compute(cls, x: Union[pd.Series, np.ndarray]) -> None:
        return cls(
            length=len(x),
            count=len(list(filter(None, x))),
            size=x.__sizeof__(),
            mean=np.mean(x),
            std=np.std(x),
            var=np.var(x),
            min=np.min(x),
            q25=np.percentile(x, q=25),
            median=np.median(x),
            q75=np.percentile(x, q=75),
            max=np.max(x),
            range=np.max(x) - np.min(x),
            skew=stats.skew(x),
            kurtosis=stats.kurtosis(x, bias=False),
        )


# ------------------------------------------------------------------------------------------------ #
@dataclass
class QualStats(VarStats):
    mode: Union[int, str]
    unique: int
    freq: int

    @classmethod
    def compute(cls, x: Union[pd.Series, np.ndarray]) -> None:
        return cls(
            length=len(x),
            count=len(list(filter(None, x))),
            size=x.__sizeof__(),
            mode=stats.mode(x),
            unique=len(np.unique(x)),
            freq=np.bincount(x).argmax(),
        )
