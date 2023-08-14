#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Exploratory Data Analysis Framework                                                 #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.10                                                                             #
# Filename   : /d8analysis/quantitative/inferential/relational/pearson.py                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/d8analysis                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday June 7th 2023 08:15:08 pm                                                 #
# Modified   : Sunday August 13th 2023 09:07:36 pm                                                 #
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
class PearsonCorrelationResult(StatTestResult):
    data: pd.DataFrame = None
    a: str = None
    b: str = None

    @inject
    def plot(self, canvas: Canvas = Provide[D8AnalysisContainer.canvas.seaborn]) -> None:
        self._canvas = canvas()
        __, self._ax = self._canvas.get_figaxes()

        # Compute the pooled skew in the data
        pooled = np.concatenate((self.data[self.a].values, self.data[self.b].values), axis=None)
        skew = stats.skew(pooled)

        # Render the probability distribution
        x = np.linspace(stats.pearson3.ppf(0.01, skew), stats.pearson3.ppf(0.99, skew), 100)
        y = stats.pearson3.pdf(x, skew)
        self._ax = sns.lineplot(x=x, y=y, markers=False, dashes=False, sort=True, ax=self._ax)

        # Compute reject region
        lower = x[0]
        upper = x[-1]
        lower_alpha = self.alpha / 2
        upper_alpha = 1 - (self.alpha / 2)
        lower_critical = stats.pearson3.ppf(lower_alpha, skew)
        upper_critical = stats.pearson3.ppf(upper_alpha, skew)

        self._fill_reject_region(
            skew=skew,
            lower=lower,
            upper=upper,
            lower_critical=lower_critical,
            upper_critical=upper_critical,
        )

        self._ax.set_title(
            f"{self.result}",
            fontsize=self._canvas.fontsize_title,
        )

        # ax.set_xlabel(r"$X^2$")
        self._ax.set_ylabel("Probability Density")
        plt.tight_layout()

        if self._legend_config is not None:
            self.config_legend()

    def _fill_reject_region(
        self,
        skew: float,
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
            y2=stats.pearson3.pdf(xlower, skew),
            color=self._canvas.colors.crimson,
        )

        # Fill Upper Tail
        xupper = np.arange(upper_critical, upper, 0.001)
        self._ax.fill_between(
            x=xupper,
            y1=0,
            y2=stats.pearson3.pdf(xupper, skew),
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
class PearsonCorrelationTest(StatisticalTest):
    __id = "pearson"

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

        r, pvalue = stats.pearsonr(
            x=self._data[self._a].values, y=self._data[self._b].values, alternative="two-sided"
        )

        dof = len(self._data) - 2

        result = self._report_results(r=r, pvalue=pvalue, dof=dof)

        if pvalue > self._alpha:  # pragma: no cover
            inference = f"The two variables had {self._interpret_r(r)}, r({dof})={round(r,2)}, {self._report_pvalue(pvalue)}.\nHowever, the pvalue, {round(pvalue,2)} is greater than level of significance {int(self._alpha*100)}% indicating that the correlation coefficient is not statistically significant."
        else:
            inference = f"The two variables had {self._interpret_r(r)}, r({dof})={round(r,2)}, {self._report_pvalue(pvalue)}.\nHowever, the pvalue, {round(pvalue,2)} is lower than level of significance {int(self._alpha*100)}% indicating that the correlation coefficient is statistically significant."

        # Create the result object.
        self._result = PearsonCorrelationResult(
            test=self._profile.name,
            H0=self._profile.H0,
            statistic=self._profile.statistic,
            hypothesis=self._profile.hypothesis,
            value=r,
            pvalue=pvalue,
            result=result,
            data=self._data,
            a=self._a,
            b=self._b,
            inference=inference,
            alpha=self._alpha,
        )

    def _report_results(self, r: float, pvalue: float, dof: float) -> str:
        return f"Pearson Correlation Test\nThe two variables had {self._interpret_r(r)}.\nr({dof})={round(r,2)}, {self._report_pvalue(pvalue)}."

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
