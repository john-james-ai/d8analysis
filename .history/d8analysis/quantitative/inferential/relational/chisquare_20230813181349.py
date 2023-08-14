#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Exploratory Data Analysis Framework                                                 #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.11                                                                             #
# Filename   : /d8analysis/quantitative/inferential/relational/chisquare.py                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/d8analysis                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday May 29th 2023 03:00:39 am                                                    #
# Modified   : Sunday August 13th 2023 06:13:49 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from dataclasses import dataclass
from typing import Union

import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from dependency_injector.wiring import inject, Provide

from d8analysis.visual.base import Canvas
from d8analysis.container import D8AnalysisContainer
from d8analysis.quantitative.inferential.base import StatTestProfileTwo
from d8analysis.quantitative.inferential.base import (
    StatTestResult,
    StatisticalTest,
    StatTestProfile,
)


# ------------------------------------------------------------------------------------------------ #
sns.set_style(SeabornCanvas.style)


# ------------------------------------------------------------------------------------------------ #
#                                     TEST RESULT                                                  #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class ChiSquareIndependenceResult(StatTestResult):
    dof: int = None
    data: pd.DataFrame = None
    x: str = None
    y: str = None

    @inject
    def plot(self, canvas: Canvas = Provide[D8AnalysisContainer.canvas.seaborn]) -> None:
        self._canvas = canvas()
        __, self._ax = self._canvas.get_figaxes()

        # Render the probability distribution
        x = np.linspace(stats.chi2.ppf(0.01, self.dof), stats.chi2.ppf(0.99, self.dof), 100)
        y = stats.chi2.pdf(x, self.dof)
        self._ax = sns.lineplot(x=x, y=y, markers=False, dashes=False, sort=True, ax=self._ax)

        # Compute reject region
        upper = x[-1]
        upper_alpha = 1 - self.alpha
        critical = stats.chi2.ppf(upper_alpha, self.dof)
        self._fill_curve(critical=critical, upper=upper)

        self._axes.set_title(
            f"X\u00b2Test Result\n{self.result}",
            fontsize=self._canvas.fontsize_title,
        )

        self._ax.set_xlabel(r"$X^2$")
        self._ax.set_ylabel("Probability Density")
        plt.tight_layout()

    def _fill_curve(self, critical: float, upper: float) -> None:
        """Fills the area under the curve at the value of the hypothesis test statistic."""

        # Fill Upper Tail
        x = np.arange(critical, upper, 0.001)
        self._ax.fill_between(
            x=x,
            y1=0,
            y2=stats.chi2.pdf(x, self._result.dof),
            color=self._canvas.colors.crimson,
        )

        # Plot the statistic
        line = self._ax.lines[0]
        xdata = line.get_xydata()[:, 0]
        ydata = line.get_xydata()[:, 1]
        statistic = round(self._result.value, 4)
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
                rf"$X^2$ = {str(statistic)}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 20),
                ha="center",
            )

            self._ax.annotate(
                "Critical Value",
                (critical, 0),
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
class ChiSquareIndependenceTest(StatisticalTest):
    __id = "x2ind"

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
        data: pd.DataFrame = None,
        x: Union[np.ndarray, str] = None,
        y: Union[np.ndarray, str] = None,
    ) -> None:
        """Performs the statistical test and creates a result object.

        Internally, the data is converted into a DataFrame and x and y are strings referencing columns in data.

        Args:
            data (pd.DataFrame) A pandas dataframe containing the two nominal/categorical
                variable columns to be tested. Optional.
            x: (Union[np.ndarray,str]): An array or string key referencing a column data, if data is provided.
            y: (Union[np.ndarray,str]): An array or string key referencing a column data, if data is provided.

        """

        obs = stats.contingency.crosstab(data[x], data[y])

        statistic, pvalue, dof, exp = stats.chi2_contingency(obs.count)

        if pvalue > self._alpha:  # pragma: no cover
            inference = f"The pvalue {round(pvalue,2)} is greater than level of significance {int(self._alpha*100)}%; therefore, the null hypothesis is not rejected. The evidence against independence of {x} and {y} is not significant."
        else:
            inference = f"The pvalue {round(pvalue,2)} is less than level of significance {int(self._alpha*100)}%; therefore, the null hypothesis is rejected. The evidence against independence of {x} and {y} is significant."

        # Create the result object.
        self._result = ChiSquareIndependenceResult(
            test=self._profile.name,
            H0=self._profile.H0,
            statistic="X\u00b2",
            hypothesis=self._profile.hypothesis,
            dof=dof,
            value=statistic,
            pvalue=pvalue,
            result=f"X\u00b2({dof}, N={len(data)})={round(statistic,2)}, {self._report_pvalue(pvalue)} {self._report_alpha()}",
            data=data,
            x=x,
            y=y,
            observed=obs,
            expected=exp,
            inference=inference,
            alpha=self._alpha,
        )
