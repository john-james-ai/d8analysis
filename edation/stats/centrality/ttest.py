#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Enter Project Name in Workspace Settings                                            #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.10                                                                             #
# Filename   : /edation/stats/centrality/ttest.py                                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : Enter URL in Workspace Settings                                                     #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday June 7th 2023 11:41:00 pm                                                 #
# Modified   : Thursday July 27th 2023 01:03:50 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from dataclasses import dataclass
import logging

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from edation.stats.profile import StatTestProfileTwo
from edation.stats.base import StatTestResult, StatisticalTestTwo, StatTestProfile
from edation.visual.config import Canvas
from edation.stats.descriptive import QuantStats

# ------------------------------------------------------------------------------------------------ #
sns.set_style(Canvas.style)


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

    def plot(self, varname: str = None, ax: plt.Axes = None) -> plt.Axes:  # pragma: no cover
        canvas = Canvas()
        ax = ax or canvas.ax
        x = np.linspace(stats.t.ppf(0.001, self.dof), stats.t.ppf(0.999, self.dof), 500)
        y = stats.t.pdf(x, self.dof)
        ax = sns.lineplot(x=x, y=y, markers=False, dashes=False, sort=True, ax=ax)

        self._logger = logging.getLogger(f"{self.__class__.__name__}")

        # Compute reject region
        lower = x[0]
        upper = x[-1]
        lower_alpha = self.alpha / 2
        upper_alpha = 1 - (self.alpha / 2)
        lower_critical = stats.t.ppf(lower_alpha, self.dof)
        upper_critical = stats.t.ppf(upper_alpha, self.dof)

        self._logger.info(f"Lower={lower}")
        self._logger.info(f"Upper={upper}")

        # Fill reject region at critical points
        # self._fill_curve(ax, lower, upper)
        self._fill_reject_region(ax, lower, upper, lower_critical, upper_critical, self.dof)

        ax.set_title(
            f"{self.result}",
            fontsize=canvas.fontsize_title,
        )

        # ax.set_xlabel(r"$X^2$")
        ax.set_ylabel("Probability Density")
        plt.tight_layout()
        return ax


# ------------------------------------------------------------------------------------------------ #
#                                          TEST                                                    #
# ------------------------------------------------------------------------------------------------ #
class TTest(StatisticalTestTwo):
    __id = "t2"

    def __init__(self, alpha: float = 0.05) -> None:
        super().__init__()
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

    def __call__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        homoscedastic: bool = False,
    ) -> None:
        """Performs the statistical test and creates a result object.

        Internally, the data is converted into a DataFrame and x and y are strings referencing columns in data.

        Args:
            x: (np.ndarray): An array containing the first of two independent samples.
            y: (np.ndarray): An array containing the second of two independent samples.
            homoscedastic (bool): If True, perform a standard independent 2 sample test that assumes equal
                population variances. If False, perform Welch’s t-test, which does not assume equal
                population variance.

        """

        statistic, pvalue = stats.ttest_ind(a=x, b=y, equal_var=homoscedastic)

        x_stats = QuantStats.compute(x)
        y_stats = QuantStats.compute(y)
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
            homoscedastic=homoscedastic,
            dof=dof,
            value=np.abs(statistic),
            pvalue=pvalue,
            result=result,
            x=x,
            y=y,
            x_stats=x_stats,
            y_stats=y_stats,
            inference=inference,
            alpha=self._alpha,
        )

    def _report_results(self, x_stats, y_stats, dof, statistic, pvalue) -> str:
        return f"Independent Samples t Test\nX: (N = {x_stats.count}, M = {round(x_stats.mean,2)}, SD = {round(x_stats.std,2)})\nY: (N = {y_stats.count}, M = {round(y_stats.mean,2)}, SD = {round(y_stats.std,2)})\nt({dof}) = {round(statistic,2)}, {self._report_pvalue(pvalue)} {self._report_alpha()}"
