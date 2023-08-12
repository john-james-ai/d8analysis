#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Exploratory Data Analysis Framework                                                 #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.11                                                                             #
# Filename   : /d8analysis/quantitative/statistical/distribution/chisquare.py                      #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/d8analysis                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday May 29th 2023 03:00:39 am                                                    #
# Modified   : Saturday August 12th 2023 12:11:09 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from dataclasses import dataclass
from typing import Union

import numpy as np
from scipy import stats

from d8analysis.quantitative.statistical.base import StatTestProfileTwo
from d8analysis.quantitative.statistical.base import (
    StatTestResult,
    StatisticalTest,
    StatTestProfile,
)


# ------------------------------------------------------------------------------------------------ #
#                                     TEST RESULT                                                  #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class ChiSquareResult(StatTestResult):
    dof: int = None
    observed: Union[list, np.ndarray] = None
    expected: Union[list, np.ndarray] = None


# ------------------------------------------------------------------------------------------------ #
#                                          TEST                                                    #
# ------------------------------------------------------------------------------------------------ #
class ChiSquareTest(StatisticalTest):
    """Calculate a one-way chi-square test.

    The chi-square test tests the null hypothesis that the categorical data has the given frequencies.

    Args:
        observed (array-like): Observed frequencies by category
        expected (array-like): Expected frequencies by category. Optional. By default,
            categories are assumed to be equally likely.

    """

    __id = "x2gof"

    def __init__(
        self,
        observed: Union[list, np.ndarray],
        expected: Union[list, np.ndarray] = None,
        alpha: float = 0.05,
    ) -> None:
        super().__init__()
        self._observed = observed
        self._expected = expected
        self._alpha = alpha
        self._profile = StatTestProfileTwo.create(self.__id)
        self._result = None

    @property
    def profile(self) -> StatTestProfile:
        """Returns the statistical test profile."""
        return self._profile

    @property
    def result(self) -> StatTestResult:
        """Returns a Statistical Test Result object."""
        return self._result

    def run(self) -> None:
        """Performs the statistical test and creates a result object."""

        dof = len(self._observed) - 1

        statistic, pvalue = stats.chisquare(f_obs=self._observed, f_exp=self._expected, axis=None)
        if pvalue > self._alpha:
            inference = f"The pvalue {round(pvalue,2)} is greater than level of significance {int(self._alpha*100)}%; therefore, the null hypothesis is not rejected. The evidence against a common distribution was not significant."
        else:
            inference = f"The pvalue {round(pvalue,2)} is less than level of significance {int(self._alpha*100)}%; therefore, the null hypothesis is rejected. The evidence against a common distribution is significant."

        if (
            self._expected is None
        ):  # Add an explainable value to the result object, rather than None.
            self._expected = ("Equal Frequencies among Groups",)

        # Create the result object.
        self._result = ChiSquareResult(
            test=self._profile.name,
            H0=self._profile.H0,
            statistic="X\u00b2",
            hypothesis=self._profile.hypothesis,
            dof=dof,
            value=statistic,
            pvalue=pvalue,
            result=f"X\u00b2({dof}, N={sum(self._observed)})={round(statistic,2)}, {self._report_pvalue(pvalue)} {self._report_alpha()}",
            observed=self._observed,
            expected=self._expected,
            inference=inference,
            alpha=self._alpha,
        )
