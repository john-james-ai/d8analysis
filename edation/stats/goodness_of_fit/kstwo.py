#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Enter Project Name in Workspace Settings                                            #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.10                                                                             #
# Filename   : /edation/stats/goodness_of_fit/kstwo.py                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : Enter URL in Workspace Settings                                                     #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday June 6th 2023 01:45:05 am                                                   #
# Modified   : Thursday July 27th 2023 08:48:04 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from edation.stats.profile import StatTestProfileTwo
from edation.stats.base import StatTestResult, StatisticalTest, StatTestProfile
from edation.visual.config import Canvas

# ------------------------------------------------------------------------------------------------ #
sns.set_style(Canvas.style)


# ------------------------------------------------------------------------------------------------ #
#                                     TEST RESULT                                                  #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class KSTwoTestResult(StatTestResult):
    sample1: Union[pd.DataFrame, np.ndarray, pd.Series] = None
    sample2: Union[pd.DataFrame, np.ndarray, pd.Series] = None

    def plot(self, ax: plt.Axes = None) -> plt.Axes:  # pragma: no cover
        """Plots the critical values and shades the area on the KS distribution

        Args:
            ax (plt.Axes): A matplotlib Axes object. Optional
        """
        canvas = Canvas()
        ax = ax or canvas.ax

        # Get the callable for the statistic.
        n = len(self.sample1)

        x = np.linspace(stats.kstwo.ppf(0.01, n), stats.ksone.ppf(0.999, n), 500)
        y = stats.kstwo.pdf(x, n)
        ax = sns.lineplot(
            x=x, y=y, markers=False, dashes=False, sort=True, ax=ax, color=canvas.colors.dark_blue
        )

        # Compute reject region
        upper_alpha = 1 - (self.alpha)
        upper = stats.kstwo.ppf(upper_alpha, n)

        # Fill reject region at critical points
        self._fill_curve(ax, upper=upper)

        ax.set_title(
            f"Goodness of Fit\nDistributions of {self.sample1.name.capitalize()} of {self.sample2.name.capitalize()}\n{self.result}",
            fontsize=canvas.fontsize_title,
        )

        ax.set_xlabel("Value")
        ax.set_ylabel("Probability Density")
        return ax

    def plothist(self, ax: plt.Axes = None) -> plt.Axes:  # pragma: no cover)
        """Plots the data against the theoretical probability distribution function.

        Args:
            ax (plt.Axes): A matplotlib Axes object. Optional
        """
        canvas = Canvas()
        ax = ax or canvas.ax

        ax = sns.histplot(
            x=self.sample1,
            color=canvas.colors.dark_blue,
            multiple="layer",
            label=self.sample1.name,
            ax=ax,
        )
        ax = sns.histplot(
            x=self.sample2,
            color=canvas.colors.orange,
            multiple="layer",
            label=self.sample2.name,
            ax=ax,
        )

        ax.set(xlabel=None)

        title = f"Goodness of Fit\nDistributions of {self.sample1.name.capitalize()} of {self.sample2.name.capitalize()}"
        ax.set_title(title, fontsize=canvas.fontsize_title)
        ax.legend()

        return ax


# ------------------------------------------------------------------------------------------------ #
#                                          TEST                                                    #
# ------------------------------------------------------------------------------------------------ #
class KSTwoTest(StatisticalTest):
    __id = "ks2"

    def __init__(self, alpha: float = 0.05) -> None:
        super().__init__()
        self._alpha = alpha
        self._sample1 = None
        self._sample2 = None
        self._profile = StatTestProfileTwo.create(self.__id)
        self._result = None

    @property
    def sample1(self) -> pd.Series:
        return self._sample1

    @property
    def sample2(self) -> pd.Series:
        return self._sample2

    @property
    def profile(self) -> StatTestProfile:
        """Returns the statistical test profile."""
        return self._profile

    @property
    def result(self) -> StatTestResult:
        """Returns a Statistical Test Result object."""
        return self._result

    def __call__(self, sample1: pd.Series, sample2: pd.Series) -> None:
        """Performs the statistical test and creates a result object.

        Args:
            sample1 (pd.Series): Pandas series containing first sample
            sample2 (pd.Series): Pandas series containing second sample

        """
        self._sample1 = sample1
        self._sample2 = sample2

        # Conduct the two-sided ks test
        result = stats.ks_2samp(
            sample1.values, sample2.values, alternative="two-sided", method="auto"
        )

        if result.pvalue > self._alpha:
            inference = f"The pvalue {round(result.pvalue,2)} is greater than level of significance {int(self._alpha*100)}%; therefore, the null hypothesis is not rejected. The evidence against a common distribution is not significantly."
        else:
            inference = f"The pvalue {round(result.pvalue,2)} is less than level of significance {int(self._alpha*100)}%; therefore, the null hypothesis is rejected. The evidence against a common distribution is significantly different."

        # Create the result object.
        self._result = KSTwoTestResult(
            test=self._profile.name,
            H0=self._profile.H0,
            statistic=self._profile.statistic,
            hypothesis=self._profile.hypothesis,
            value=result.statistic,
            pvalue=result.pvalue,
            result=f"(KS={round(result.statistic,2)}), {self._report_pvalue(result.pvalue)} {self._report_alpha()}",
            sample1=sample1,
            sample2=sample2,
            inference=inference,
            alpha=self._alpha,
        )
