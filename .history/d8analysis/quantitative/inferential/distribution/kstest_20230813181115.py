#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Exploratory Data Analysis Framework                                                 #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.10                                                                             #
# Filename   : /d8analysis/quantitative/inferential/distribution/kstest.py                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/d8analysis                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday June 6th 2023 01:45:05 am                                                   #
# Modified   : Sunday August 13th 2023 06:11:14 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from dataclasses import dataclass
from typing import Union, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from dependency_injector.wiring import inject, Provide

from d8analysis.container import D8AnalysisContainer
from d8analysis.visual.base import Canvas
from d8analysis.quantitative.inferential.base import (
    StatTestProfileOne,
    StatTestResult,
    StatisticalTest,
    StatTestProfile,
)


# ------------------------------------------------------------------------------------------------ #
#                                     TEST RESULT                                                  #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class KSTestResult(StatTestResult):
    reference_distribution: str = None
    data: Union[pd.DataFrame, np.ndarray, pd.Series] = None

    @inject
    def plot(self, canvas: Canvas = Provide[D8AnalysisContainer.canvas.seaborn]) -> None:
        """Plots the test statistic and reject region

        Args:
            canvas (Canvas): Visual configuration object.
        """
        self._canvas = canvas()
        _, self._ax = self._canvas.get_figaxes()

        n = len(self.data)

        # Render the probability distribution
        x = np.linspace(stats.kstwo.ppf(0.001, self.dof), stats.kstwo.ppf(0.999, self.dof), 500)
        y = stats.kstwo.pdf(x, n)
        self._ax = sns.lineplot(x=x, y=y, markers=False, dashes=False, sort=True, ax=self._ax)

        # Compute reject region
        lower = x[0]
        upper = x[-1]
        lower_alpha = self.alpha / 2
        upper_alpha = 1 - (self.alpha / 2)
        lower_critical = stats.kstwo.ppf(lower_alpha, n)
        upper_critical = stats.kstwo.ppf(upper_alpha, n)

        self._fill_reject_region(
            n=n,
            lower=lower,
            upper=upper,
            lower_critical=lower_critical,
            upper_critical=upper_critical,
        )

        self._ax.set_title(
            f"{self.result}",
            fontsize=self._canvas.fontsize_title,
        )

        self._ax.set_ylabel("Probability Density")
        plt.tight_layout()

        if self._legend_config is not None:
            self.config_legend()

    def _fill_reject_region(
        self,
        n: int,
        lower: float,
        upper: float,
        lower_critical: float,
        upper_critical: float,
    ) -> None:
        """Fills the area under the curve at the value of the hypothesis test statistic."""

        # Fill lower tail
        xlower = np.arange(lower, lower_critical, 0.001)
        self._ax.fill_between(
            x=xlower,
            y1=0,
            y2=stats.kstwo.pdf(xlower, n),
            color=self._canvas.colors.crimson,
        )

        # Fill Upper Tail
        xupper = np.arange(upper_critical, upper, 0.001)
        self._ax.fill_between(
            x=xupper,
            y1=0,
            y2=stats.kstwo.pdf(xupper, n),
            color=self._canvas.colors.crimson,
        )

        # Plot the statistic
        line = self._ax.lines[0]
        xdata = line.get_xydata()[:, 0]
        ydata = line.get_xydata()[:, 1]
        statistic = round(self.value, 4)
        try:
            idx = np.where(xdata > self.value)[0][0]
            x = xdata[idx]
            y = ydata[idx]
            _ = sns.regplot(
                x=np.array([x]),
                y=np.array([y]),
                scatter=True,
                fit_reg=False,
                marker="o",
                scatter_kws={"s": 100},
                ax=self._ax,
                color=self._canvas.colors.dark_blue,
            )
            self._ax.annotate(
                f"t = {str(statistic)}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

            self._ax.annotate(
                "Critical Value",
                (lower_critical, 0),
                textcoords="offset points",
                xytext=(20, 15),
                ha="left",
                arrowprops={"width": 2, "shrink": 0.05},
            )

            self._ax.annotate(
                "Critical Value",
                (upper_critical, 0),
                xycoords="data",
                textcoords="offset points",
                xytext=(-20, 15),
                ha="right",
                arrowprops={"width": 2, "shrink": 0.05},
            )
        except IndexError:
            pass


# ------------------------------------------------------------------------------------------------ #
#                                          TEST                                                    #
# ------------------------------------------------------------------------------------------------ #
class KSTest(StatisticalTest):
    """Performs the statistical test and creates a result object.

    Args:
        data (np.ndarray): 1D Numpy array of data to be tested.
        reference_distribution (str): A reference distribution from the scipy list
            of Continuous Distributions at https://docs.scipy.org/doc/scipy/reference/stats.html

    """

    __id = "ks1"

    def __init__(self, data: np.ndarray, reference_distribution: str, alpha: float = 0.05) -> None:
        super().__init__()
        self._data = data
        self._reference_distribution = reference_distribution
        self._alpha = alpha
        self._profile = StatTestProfileOne.create(self.__id)
        self._result = None

    @property
    def profile(self) -> StatTestProfile:
        """Returns the statistical test profile."""
        return self._profile

    @property
    def data(self) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Returns the data tested"""
        return self._data

    @property
    def result(self) -> StatTestResult:
        """Returns a Statistical Test Result object."""
        return self._result

    def run(self) -> None:
        """Performs the statistical test and creates a result object."""

        # Conduct the two-sided ks test
        try:
            result = stats.ks_1samp(
                x=self._data, cdf=self._reference_distribution, alternative="two-sided"
            )
        except KeyError as e:
            msg = f"Distribution {self._reference_distribution} is not supported.\n{e}"
            self._logger.error(msg)
            raise

        if result.pvalue > self._alpha:
            inference = f"The pvalue {round(result.pvalue,2)} is greater than level of significance {int(self._alpha*100)}%; therefore, the null hypothesis is not rejected. The evidence against the data being drawn from the {self._reference_distribution} distribution is not significant."
        else:
            inference = f"The pvalue {round(result.pvalue,2)} is less than level of significance {int(self._alpha*100)}%; therefore, the null hypothesis is rejected. The evidence against the data being drawn from the {self._reference_distribution} distribution is significant."

        interpretation = None
        if len(self._data) < 50:
            interpretation = "Note: The Kolmogorov-Smirnov Test requires a sample size N > 50. For smaller sample sizes, the Shapiro-Wilk test should be considered."
        if len(self._data) > 1000:
            interpretation = "Note: The Kolmogorov-Smirnov Test on large sample sizes may lead to rejections of the null hypothesis that are statistically significant, yet practically insignificant."

        # Create the result object.
        self._result = KSTestResult(
            test=self._profile.name,
            H0=self._profile.H0,
            statistic=self._profile.statistic,
            hypothesis=self._profile.hypothesis,
            value=result.statistic,
            pvalue=result.pvalue,
            result=f"(N={len(self._data)})={round(result.statistic,2)}, {self._report_pvalue(result.pvalue)} {self._report_alpha()}",
            data=self._data,
            reference_distribution=self._reference_distribution,
            inference=inference,
            interpretation=interpretation,
            alpha=self._alpha,
        )
