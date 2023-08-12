#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Exploratory Data Analysis Framework                                                 #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.10                                                                             #
# Filename   : /d8analysis/quantitative/statistical/centrality/ttest.py                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/d8analysis                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday June 7th 2023 11:41:00 pm                                                 #
# Modified   : Friday August 11th 2023 07:57:46 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from dataclasses import dataclass

import numpy as np
from scipy import stats

from d8analysis.quantitative.statistical.base import (
    StatTestProfileTwo,
    StatTestResult,
    StatisticalTest,
    StatTestProfile,
)
from d8analysis.quantitative.descriptive.summary import QuantStats


# ------------------------------------------------------------------------------------------------ #
#                                     TEST RESULT                                                  #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class TTestResult(StatTestResult):
    dof: int = None
    homoscedastic: bool = None
    x: str = None
    y: str = None
    x_stats: QuantStats = None
    y_stats: QuantStats = None


# ------------------------------------------------------------------------------------------------ #
#                                          TEST                                                    #
# ------------------------------------------------------------------------------------------------ #
class TTest(StatisticalTest):
    """Calculate the T-test for the means of two independent samples of scores.

    This is a test for the null hypothesis that 2 independent samples have identical average
    (expected) values. This test assumes that the populations have identical variances by default.

    Args:
        x: (np.ndarray): An array containing the first of two independent samples.
        y: (np.ndarray): An array containing the second of two independent samples.
        alpha (float): The level of statistical significance for inference.
        homoscedastic (bool): If True, perform a standard independent 2 sample test t
            hat assumes equal population variances. If False, perform Welch’s
            t-test, which does not assume equal population variance.

    """

    __id = "t2"

    def __init__(
        self, x: np.ndarray, y: np.ndarray, alpha: float = 0.05, homoscedastic: bool = False
    ) -> None:
        super().__init__()
        self._x = x
        self._y = y
        self._alpha = alpha
        self._homoscedastic = homoscedastic
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
        """Executes the TTest."""

        statistic, pvalue = stats.ttest_ind(a=self._x, b=self._y, equal_var=self._homoscedastic)

        x_stats = QuantStats.compute(self._x)
        y_stats = QuantStats.compute(self._y)
        dof = x_stats.length + y_stats.length - 2

        result = self._report_results(x_stats, y_stats, dof, statistic, pvalue)

        if pvalue > self._alpha:  # pragma: no cover
            inference = f"The pvalue {round(pvalue,2)} is greater than level of significance {int(self._alpha*100)}%; therefore, the null hypothesis is not rejected. The evidence against identical centers for x and y is not significant."
        else:
            inference = f"The pvalue {round(pvalue,2)} is less than level of significance {int(self._alpha*100)}%; therefore, the null hypothesis is rejected. The evidence against identical centers for x and y is significant."

        # Create the result object.
        self._result = TTestResult(
            test=self._profile.name,
            H0=self._profile.H0,
            statistic=self._profile.statistic,
            hypothesis=self._profile.hypothesis,
            homoscedastic=self._homoscedastic,
            dof=dof,
            value=np.abs(statistic),
            pvalue=pvalue,
            result=result,
            x=self._x,
            y=self._y,
            x_stats=x_stats,
            y_stats=y_stats,
            inference=inference,
            alpha=self._alpha,
        )

    def _report_results(self, x_stats, y_stats, dof, statistic, pvalue) -> str:
        return f"Independent Samples t Test\nX: (N = {x_stats.count}, M = {round(x_stats.mean,2)}, SD = {round(x_stats.std,2)})\nY: (N = {y_stats.count}, M = {round(y_stats.mean,2)}, SD = {round(y_stats.std,2)})\nt({dof}) = {round(statistic,2)}, {self._report_pvalue(pvalue)} {self._report_alpha()}"
