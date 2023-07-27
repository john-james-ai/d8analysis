#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Enter Project Name in Workspace Settings                                            #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.11                                                                             #
# Filename   : /edation/stats/goodness_of_fit/chisquare.py                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : Enter URL in Workspace Settings                                                     #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday May 29th 2023 03:00:39 am                                                    #
# Modified   : Thursday July 27th 2023 08:48:06 am                                                 #
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

from edation.stats.profile import StatTestProfileTwo
from edation.stats.base import StatTestResult, StatisticalTest, StatTestProfile
from edation.visual.config import Canvas

# ------------------------------------------------------------------------------------------------ #
sns.set_style(Canvas.style)


# ------------------------------------------------------------------------------------------------ #
#                                     TEST RESULT                                                  #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class ChiSquareGOFResult(StatTestResult):
    dof: int = None
    data: Union[pd.DataFrame, np.ndarray, pd.Series] = None
    observed: Union[pd.DataFrame, np.ndarray, pd.Series] = None
    expected: Union[pd.DataFrame, np.ndarray, pd.Series] = None

    def plot(self, varname: str = None, ax: plt.Axes = None) -> plt.Axes:  # pragma: no cover
        canvas = Canvas()
        ax = ax or canvas.ax
        x = np.linspace(stats.chi2.ppf(0.01, self.dof), stats.chi2.ppf(0.99, self.dof), 100)
        y = stats.chi2.pdf(x, self.dof)
        ax = sns.lineplot(x=x, y=y, markers=False, dashes=False, sort=True, ax=ax)

        # Compute reject region
        upper_alpha = 1 - (self.alpha / 2)
        upper = stats.chi2.ppf(upper_alpha, self.dof)
        ax = self._fill_curve(ax, upper=upper)

        ax.set_title(
            f"X\u00b2 Goodness of Fit\nTest Result\n{self.result}", fontsize=canvas.fontsize_title
        )

        ax.set_xlabel(r"$X^2$")
        ax.set_ylabel("Probability Density")
        return ax


# ------------------------------------------------------------------------------------------------ #
#                                          TEST                                                    #
# ------------------------------------------------------------------------------------------------ #
class ChiSquareGOFTest(StatisticalTest):
    __id = "x2gof"

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
    def data(self) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Returns the data tested"""
        return self._data

    @property
    def result(self) -> StatTestResult:
        """Returns a Statistical Test Result object."""
        return self._result

    def __call__(self, data: Union[pd.Series, pd.DataFrame], expected: dict = None) -> None:
        """Performs the statistical test and creates a result object.

        Args:
            data (Union[pd.Series,np.ndarray]) A pandas series or a one-column dataframe containing the
                nominal / categorical data to be tested.
            expected (dict): Dictionary in which the keys are categories and the values
                are the expected frequencies.

        """
        self._data = data

        # Extract observed frequencies sorted by category in lexical order
        observed = data.value_counts(sort=True, ascending=False).to_frame().sort_index().values
        # Extract expected frequencies (if provided) similarly
        if expected is not None:
            expected = pd.DataFrame.from_dict(data=expected, orient="index").sort_index().values

        dof = len(observed) - 1

        statistic, pvalue = stats.chisquare(f_obs=observed, f_exp=expected, axis=None)
        if pvalue > self._alpha:
            inference = f"The pvalue {round(pvalue,2)} is greater than level of significance {int(self._alpha*100)}%; therefore, the null hypothesis is not rejected. The evidence against a common distribution was not significant."
        else:
            inference = f"The pvalue {round(pvalue,2)} is less than level of significance {int(self._alpha*100)}%; therefore, the null hypothesis is rejected. The evidence against a common distribution is significant."

        if expected is None:  # Add an explainable value to the result object, rather than None.
            expected = ("Equal Frequencies among Groups",)

        # Create the result object.
        self._result = ChiSquareGOFResult(
            test=self._profile.name,
            H0=self._profile.H0,
            statistic="X\u00b2",
            hypothesis=self._profile.hypothesis,
            dof=dof,
            value=statistic,
            pvalue=pvalue,
            result=f"X\u00b2({dof}, N={len(data)})={round(statistic,2)}, {self._report_pvalue(pvalue)} {self._report_alpha()}",
            data=data,
            observed=observed,
            expected=expected,
            inference=inference,
            alpha=self._alpha,
        )
