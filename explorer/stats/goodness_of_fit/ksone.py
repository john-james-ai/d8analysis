#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Enter Project Name in Workspace Settings                                            #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.10                                                                             #
# Filename   : /explorer/stats/goodness_of_fit/ksone.py                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : Enter URL in Workspace Settings                                                     #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday June 6th 2023 01:45:05 am                                                   #
# Modified   : Wednesday June 7th 2023 03:59:06 am                                                 #
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

from explorer.stats.profile import StatTestProfileOne
from explorer.stats.base import StatTestResult, StatisticalTest, StatTestProfile
from explorer.visual.config import Canvas
from explorer.stats.distribution import DISTRIBUTIONS
from explorer.stats.distribution import RVSDistribution

# ------------------------------------------------------------------------------------------------ #
MC_SAMPLES = 100


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
        canvas = Canvas()
        ax = ax or canvas.ax

        # Get the callable for the statistic.
        n = len(self.data)

        x = np.linspace(stats.ksone.ppf(0.01, n), stats.ksone.ppf(0.999, n), 100)
        y = stats.ksone.pdf(x, n)
        ax = sns.lineplot(
            x=x, y=y, markers=False, dashes=False, sort=True, ax=ax, color=canvas.colors.dark_blue
        )
        line = ax.lines[0]
        xdata = line.get_xydata()[:, 0]
        ydata = line.get_xydata()[:, 1]
        # Get index of first value greater than the statistic.
        try:
            idx = np.where(xdata > self.value)[0][0]
            fill_x = xdata[idx:]
            fill_y2 = ydata[idx:]
            ax.fill_between(x=fill_x, y1=0, y2=fill_y2, color=canvas.colors.orange)
        except IndexError:
            pass
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
    __id = "ks1"

    def __init__(self, alpha: float = 0.05) -> None:
        super().__init__()
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

    def __call__(self, data: np.ndarray, reference_distribution: str) -> None:
        """Performs the statistical test and creates a result object.

        Args:
            data (np.ndarray): 1D Numpy array of data to be tested.
            reference_distribution (str): A reference distribution from the scipy list
                of Continuous Distributions at https://docs.scipy.org/doc/scipy/reference/stats.html

        """
        self._data = data

        # Conduct the two-sided ks test
        try:
            result = stats.goodness_of_fit(
                dist=DISTRIBUTIONS[reference_distribution],
                data=data,
                statistic="ks",
                n_mc_samples=MC_SAMPLES,
            )
        except KeyError as e:
            msg = f"Distribution {reference_distribution} is not supported.\n{e}"
            self._logger.error(msg)
            raise

        self._logger.debug(
            f"\n\nType Pvalue: {type(result.pvalue)}\nType Statistic{type(result.statistic)}"
        )

        if result.pvalue > self._alpha:
            gtlt = ">"
            inference = f"The pvalue {round(result.pvalue,2)} is greater than level of significance {self._alpha}; therefore, the null hypothesis is not rejected. The data were drawn from the reference distribution."
        else:
            gtlt = "<"
            inference = f"The pvalue {round(result.pvalue,2)} is less than level of significance {self._alpha}; therefore, the null hypothesis is rejected. The data were not drawn from the reference distribution."

        # Create the result object.
        self._result = KSOneTestResult(
            test=self._profile.name,
            H0=self._profile.H0,
            statistic=self._profile.statistic,
            hypothesis=self._profile.hypothesis,
            value=result.statistic,
            pvalue=result.pvalue,
            result=f"(N={len(data)})={round(result.statistic,2)}, p{gtlt}{self._alpha}",
            data=data,
            reference_distribution=reference_distribution,
            inference=inference,
            alpha=self._alpha,
        )
