#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Exploratory Data Analysis Framework                                                 #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.10                                                                             #
# Filename   : /d8analysis/quantitative/inferential/relational/spearman.py                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/d8analysis                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday June 7th 2023 08:15:08 pm                                                 #
# Modified   : Sunday August 13th 2023 10:07:34 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from dataclasses import dataclass
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
#                                     TEST RESULT                                                  #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class SpearmanCorrelationResult(StatTestResult):
    data: pd.DataFrame = None
    x: str = None
    y: str = None
    dof: float = None

    @inject
    def plot(self, canvas: Canvas = Provide[D8AnalysisContainer.canvas.seaborn]) -> None:
        self._canvas = canvas()
        __, self._ax = self._canvas.get_figaxes()

        # Render probability distribution
        dist = stats.t(df=self.dof)
        x = np.linspace(-5, 5, 100)
        y = dist.pdf(x)
        self._ax = sns.lineplot(x=x, y=y, markers=False, dashes=False, sort=True, ax=self._ax)

        # Transform the r statistic and pvalue as per https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html#scipy.stats.spearmanr
        critical_value = self.value * np.sqrt(self.dof / ((self.value + 1.0) * (1.0 - self.value)))
        pvalue = dist.cdf(-critical_value) + dist.sf(critical_value)
        annotation = f"p-value={round(pvalue,4)}\n(shaded area)"

        # Compute reject region.
        upper = x >= critical_value
        lower = x <= -critical_value

        # Plot statistic
        idx = np.where(x[upper])[0]
        a = x[idx]
        b = y[idx]
        _ = sns.regplot(
            x=np.array([a]),
            y=np.array([b]),
            scatter=True,
            fit_reg=False,
            marker="o",
            scatter_kws={"s": 100},
            ax=self._ax,
            color=self._canvas.colors.dark_blue,
        )
        self._ax.annotate(
            f"r({self.dof})={self.value}, ",
            (a, b),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

        # Fill Lower Tail
        self._ax.fill_between(
            x[lower],
            y1=0,
            y2=y[lower],
            color=self._canvas.colors.crimson,
        )

        # Fill Upper Tail
        self._ax.fill_between(
            x[upper],
            y1=0,
            y2=y[upper],
            color=self._canvas.colors.crimson,
        )

        # Create annotations
        self._ax.annotate(
            "Critical Value",
            (-critical_value, 0),
            textcoords="offset points",
            xytext=(20, 15),
            ha="left",
            arrowprops={"width": 2, "shrink": 0.05},
        )

        self._ax.annotate(
            "Critical Value",
            (critical_value, 0),
            xycoords="data",
            textcoords="offset points",
            xytext=(-20, 15),
            ha="right",
            arrowprops={"width": 2, "shrink": 0.05},
        )


# ------------------------------------------------------------------------------------------------ #
#                                          TEST                                                    #
# ------------------------------------------------------------------------------------------------ #
class SpearmanCorrelationTest(StatisticalTest):
    __id = "spearman"

    def __init__(self, data: pd.DataFrame, a=str, b=str, alpha: float = 0.05) -> None:
        super().__init__()
        self._data = data
        self._a = a
        self._b = b
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

    def run(self) -> None:
        """Performs the statistical test and creates a result object."""

        r, pvalue = stats.spearmanr(
            x=self._data[self._a].values,
            y=self._data[self._b].values,
            alternative="two-sided",
            nan_policy="omit",
        )

        dof = len(self._data) - 2

        result = self._report_results(r=r, pvalue=pvalue, dof=dof)

        if pvalue > self._alpha:  # pragma: no cover
            inference = f"The two variables had {self._interpret_r(r)}, r({dof})={round(r,2)}, {self._report_pvalue(pvalue)}.\nHowever, the pvalue, {round(pvalue,2)} is greater than level of significance {int(self._alpha*100)}% indicating that the correlation coefficient is not statistically significant."
        else:
            inference = f"The two variables had {self._interpret_r(r)}, r({dof})={round(r,2)}, {self._report_pvalue(pvalue)}.\nHowever, the pvalue, {round(pvalue,2)} is lower than level of significance {int(self._alpha*100)}% indicating that the correlation coefficient is statistically significant."

        # Create the result object.
        self._result = SpearmanCorrelationResult(
            test=self._profile.name,
            H0=self._profile.H0,
            statistic=self._profile.statistic,
            hypothesis=self._profile.hypothesis,
            value=r,
            pvalue=pvalue,
            dof=dof,
            result=result,
            data=self._data,
            a=self._a,
            b=self._b,
            inference=inference,
            alpha=self._alpha,
        )

    def _report_results(self, r: float, pvalue: float, dof: float) -> str:
        return f"Spearman Correlation Test\nThe two variables had {self._interpret_r(r)}.\nr({dof})={round(r,3)}, {self._report_pvalue(pvalue)}."

    def _interpret_r(self, r: float) -> str:
        """Interprets the value of the correlation[1]_

        .. [1] Mukaka MM. Statistics corner: A guide to appropriate use of correlation coefficient in medical research. Malawi Med J. 2012 Sep;24(3):69-71. PMID: 23638278; PMCID: PMC3576830.


        """

        if r < 0:
            direction = "negative"
        else:
            direction = "positive"

        r = abs(r)
        if r >= 0.9:
            return f"very high {direction} correlation."
        elif r >= 0.70:
            return f"high {direction} correlation."
        elif r >= 0.5:
            return f"moderate {direction} correlation."
        elif r >= 0.3:
            return f"low {direction} correlation."
        else:
            return "negligible correlation."
