#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Enter Project Name in Workspace Settings                                            #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.10                                                                             #
# Filename   : /explorer/stats/base.py                                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : Enter URL in Workspace Settings                                                     #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday June 5th 2023 12:13:09 am                                                    #
# Modified   : Monday June 5th 2023 06:14:55 pm                                                    #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations
from abc import ABC
from dataclasses import dataclass, fields

from explorer.service.io import IOService

# ------------------------------------------------------------------------------------------------ #
ANALYSIS_TYPES = {
    "univariate": "Univariate",
    "bivariate": "Bivariate",
    "multivariate": "Multivariate",
}
HYPOTHESIS_TYPES = {
    "ind": "Independence",
    "corr": "Correlation",
    "gof": "Goodness of Fit",
    "centrality": "Central Tendency of Groups",
    "norm": "Normality",
    "var": "Equal Variance",
    "dist": "Equal Distributions",
}
VARIABLE_TYPES = {
    "discrete": "Discrete",
    "continuous": "Continuous",
    "nominal": "Nominal",
    "ordinal": "Ordinal",
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
    atype: str = None  # one of ANALYSIS_TYPES
    htype: str = None  # One of HYPOTHESIS_TYPES
    h0: str = None
    parametric: bool = None
    min_sample_size: int = None
    assumptions: str = None
    use_when: str = None

    @classmethod
    def create(cls, id) -> None:
        """Loads the values from the statistical tests file"""
        profiles = IOService(STAT_CONFIG)
        profile = profiles[id]
        fieldlist = {f.name for f in fields(cls) if f.init}
        filtered_dict = {k: v for k, v in profile.items() if k in fieldlist}
        filtered_dict["id"] = id
        return cls(**filtered_dict)
