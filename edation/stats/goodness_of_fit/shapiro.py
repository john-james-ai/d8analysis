#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Enter Project Name in Workspace Settings                                            #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.10                                                                             #
# Filename   : /edation/stats/goodness_of_fit/shapiro.py                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : Enter URL in Workspace Settings                                                     #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday June 7th 2023 08:15:08 pm                                                 #
# Modified   : Thursday July 27th 2023 08:48:03 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from edation.stats.profile import StatTestProfileOne
from edation.stats.base import StatTestResult, StatisticalTest
from edation.visual.config import Canvas

# ------------------------------------------------------------------------------------------------ #
sns.set_style(Canvas.style)


# ------------------------------------------------------------------------------------------------ #
#                                     TEST RESULT                                                  #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class ShapiroWilkResult(StatTestResult):
    data: pd.DataFrame = None

    def plot(self, varname: str = None, ax: plt.Axes = None) -> plt.Axes:  # pragma: no cover
        canvas = Canvas()
        ax = ax or canvas.ax

        loc = np.mean(self.data)
        scale = np.std(self.data)

        x = np.linspace(
            stats.norm.ppf(0.01, loc=loc, scale=scale),
            stats.norm.ppf(0.99, loc=loc, scale=scale),
            100,
        )
        y = stats.norm.pdf(x=x, loc=loc, scale=scale)
        ax = sns.lineplot(
            x=x,
            y=y,
            markers=False,
            dashes=False,
            sort=True,
            ax=ax,
            color=canvas.colors.dark_blue,
            label="Normal Distribution",
        )
        ax = sns.kdeplot(
            x=self.data, ax=ax, color=canvas.colors.orange, label="Data Kernel Density Estimate"
        )

        ax.set_title(f"Shapiro-Wilk Normality Test\n{self.result}", fontsize=canvas.fontsize_title)

        ax.set_xlabel("X")
        ax.set_ylabel("Probability Density")
        ax.legend(fontsize=canvas.fontsize)

        return ax


# ------------------------------------------------------------------------------------------------ #
#                                          TEST                                                    #
# ------------------------------------------------------------------------------------------------ #
class ShapiroWilkTest(StatisticalTest):
    __id = "sw"

    def __init__(self, alpha: float = 0.05) -> None:
        super().__init__()
        self._alpha = alpha
        self._profile = StatTestProfileOne.create(self.__id)
        self._result = None

    @property
    def profile(self) -> StatTestProfileOne:
        """Returns the statistical test profile."""
        return self._profile

    @property
    def result(self) -> StatTestResult:
        """Returns a Statistical Test Result object."""
        return self._result

    def __call__(self, data: np.ndarray) -> None:
        """Performs the statistical test and creates a result object.

        Args:
            data (np.ndarray): Array of data to be tested.

        """

        W, pvalue = stats.shapiro(x=data)

        if pvalue > self._alpha:  # pragma: no cover
            inference = f"The pvalue {round(pvalue,2)} is greater than level of significance {int(self._alpha*100)}%; therefore, the null hypothesis is not rejected. The evidence didn't indicate non-normality for the data."
        else:
            inference = f"The pvalue {round(pvalue,2)} is less than level of significance {int(self._alpha*100)}%; therefore, the null hypothesis is rejected. The evidence against normality is significant."

        interpretation = None
        if len(data) > 5000:
            interpretation = (
                "For N > 5000, the W test statistic is accurate, but the p-value may not be."
            )

        # Create the result object.
        self._result = ShapiroWilkResult(
            test=self._profile.name,
            H0=self._profile.H0,
            statistic=self._profile.statistic,
            hypothesis=self._profile.hypothesis,
            value=W,
            pvalue=pvalue,
            result=f"(W={round(W,2)}, {self._report_pvalue(pvalue)}), {self._report_alpha()}",
            data=data,
            inference=inference,
            interpretation=interpretation,
            alpha=self._alpha,
        )
