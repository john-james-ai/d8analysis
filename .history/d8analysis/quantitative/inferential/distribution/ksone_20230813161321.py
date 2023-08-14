#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Exploratory Data Analysis Framework                                                 #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.10                                                                             #
# Filename   : /d8analysis/quantitative/inferential/distribution/ksone.py                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/d8analysis                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday June 6th 2023 01:45:05 am                                                   #
# Modified   : Sunday August 13th 2023 04:13:19 pm                                                 #
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

from d8analysis.quantitative.inferential.base import StatTestProfileOne
from d8analysis.quantitative.inferential.base import (
    StatTestResult,
    StatisticalTest,
    StatTestProfile,
)
from d8analysis.visual.seaborn.config import SeabornCanvas
from d8analysis.data.generation import CDF

# ------------------------------------------------------------------------------------------------ #
MC_SAMPLES = 100
# ------------------------------------------------------------------------------------------------ #
sns.set_style(SeabornCanvas.style)


# ------------------------------------------------------------------------------------------------ #
#                                     TEST RESULT                                                  #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class KSOneTestResult(StatTestResult):
    reference_distribution: str = None
    data: Union[pd.DataFrame, np.ndarray, pd.Series] = None

    def plot(self, ax: plt.Axes = None) -> plt.Axes:  # pragma: no cover
        """Plots the critical values and shades the area on the KS distribution

        Args:
            ax (plt.Axes): A matplotlib Axes object. Optional
        """
        canvas = SeabornCanvas()
        ax = ax or canvas.ax

        # Get the callable for the statistic.
        n = len(self.data)

        x = np.linspace(stats.ksone.ppf(0.01, n), stats.ksone.ppf(0.999, n), 500)
        y = stats.ksone.pdf(x, n)
        ax = sns.lineplot(
            x=x, y=y, markers=False, dashes=False, sort=True, ax=ax, color=canvas.colors.dark_blue
        )

        # Compute reject region
        upper_alpha = 1 - (self.alpha)
        upper = stats.ksone.ppf(upper_alpha, n)

        # Fill reject region at critical points
        self._fill_curve(ax, upper=upper)

        ax.set_title(
            f"Goodness of Fit\n{self.reference_distribution.capitalize()} Distribution\n{self.result}",
            fontsize=canvas.fontsize_title,
        )

        ax.set_xlabel("Value")
        ax.set_ylabel("Probability Density")
        return ax

    def plotpdf(self, ax: plt.Axes = None) -> plt.Axes:  # pragma: no cover)
        """Plots the data against the theoretical probability distribution function.

        Args:
            ax (plt.Axes): A matplotlib Axes object. Optional
        """
        dist = RVSDistribution()
        dist(data=self.data, distribution=self.reference_distribution)
        return dist.histpdfplot()

    def plotcdf(self, ax: plt.Axes = None) -> plt.Axes:  # pragma: no cover)
        """Plots the data against the theoretical cumulative distribution function.

        Args:
            ax (plt.Axes): A matplotlib Axes object. Optional
        """
        dist = RVSDistribution()
        dist(data=self.data, distribution=self.reference_distribution)
        return dist.ecdfplot()


# ------------------------------------------------------------------------------------------------ #
#                                          TEST                                                    #
# ------------------------------------------------------------------------------------------------ #
class KSOneTest(StatisticalTest):
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
                x=self._data, cdf=CDF[self._reference_distribution], alternative="two-sided"
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
        self._result = KSOneTestResult(
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
