#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Enter Project Name in Workspace Settings                                            #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.10                                                                             #
# Filename   : /explorer/stats/goodness_of_fit/kstest.py                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : Enter URL in Workspace Settings                                                     #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday June 6th 2023 01:45:05 am                                                   #
# Modified   : Tuesday June 6th 2023 02:24:12 am                                                   #
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


# ------------------------------------------------------------------------------------------------ #
#                                     TEST RESULT                                                  #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class KSTestOneResult(StatTestResult):
    statistic_location: float = None
    statistic_sign: int = None
    reference_distribution: str = None
    reference_data: Union[pd.DataFrame, np.ndarray, pd.Series] = None
    data: Union[pd.DataFrame, np.ndarray, pd.Series] = None

    def plot(self, ax: plt.Axes = None) -> plt.Axes:  # pragma: no cover
        ax = ax or Canvas().ax
        x = np.linspace(stats.kstwobign.ppf(0.01), stats.kstwobign.ppf(0.99), 100)
        y = stats.kstwobign.pdf(x)
        sns.lineplot(x=x, y=y, markers=False, dashes=False, sort=True, ax=ax)
        line = ax.lines[0]
        fill_x = line.get_xydata()[int(self.value) :, 0]  # noqa: E203
        fill_y2 = line.get_xydata()[int(self.value) :, 1]  # noqa: E203
        ax.fill_between(x=fill_x, y1=0, y2=fill_y2, color="red")
        ax.set_title(f"Kolmogorov-Smirnov Goodness of Fit\nTest Result\n{self.result}")

        ax.set_xlabel("Value")
        ax.set_ylabel("Probability Density")
        return ax


# ------------------------------------------------------------------------------------------------ #
#                                          TEST                                                    #
# ------------------------------------------------------------------------------------------------ #
class KSTestOne(StatisticalTest):
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
        statistic, pvalue, location, sign = stats.kstest(
            rvs=data, cdf=reference_distribution, alternative="two-sided", method="auto"
        )

        # Generate a sample from the reference distribution

        # Extract observed frequencies sorted by category in lexical order
        observed = data.value_counts(sort=True, ascending=False).to_frame().sort_index().values
        # Extract expected frequencies (if provided) similarly
        if expected is not None:
            expected = pd.DataFrame.from_dict(data=expected, orient="index").sort_index().values

        dof = len(observed) - 1

        statistic, pvalue = stats.chisquare(f_obs=observed, f_exp=expected, axis=None)
        if pvalue > self._alpha:
            gtlt = ">"
            inference = f"The pvalue {round(pvalue,2)} is greater than level of significance {self._alpha}; therefore, the null hypothesis is not rejected. The data have the expected frequencies."
        else:
            gtlt = "<"
            inference = f"The pvalue {round(pvalue,2)} is less than level of significance {self._alpha}; therefore, the null hypothesis is rejected. The data do not have the expected frequencies."

        if expected is None:  # Add an explainable value to the result object, rather than None.
            expected = ("Equal Frequencies among Groups",)

        # Create the result object.
        self._result = KSTestOneResult(
            test=self._profile.name,
            H0=self._profile.H0,
            statistic=self._profile.statistic,
            hypothesis=self._profile.hypothesis,
            dof=dof,
            value=statistic,
            pvalue=pvalue,
            result=f"X\u00b2({dof}, N={len(data)})={round(statistic,2)}, p{gtlt}{self._alpha}",
            data=data,
            observed=observed,
            expected=expected,
            inference=inference,
            alpha=self._alpha,
        )
