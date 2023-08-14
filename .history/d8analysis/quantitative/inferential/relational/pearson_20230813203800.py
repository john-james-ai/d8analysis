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
# Modified   : Sunday August 13th 2023 08:37:59 pm                                                 #
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

from d8analysis.quantitative.inferential.base import StatTestProfileTwo
from d8analysis.quantitative.inferential.base import (
    StatTestResult,
    StatisticalTest,
    StatTestProfile,
)
from d8analysis.visual.seaborn.config import SeabornCanvas

# ------------------------------------------------------------------------------------------------ #
sns.set_style(SeabornCanvas.style)


# ------------------------------------------------------------------------------------------------ #
#                                     TEST RESULT                                                  #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class PearsonCorrelationResult(StatTestResult):
    data: pd.DataFrame = None
    x: str = None
    y: str = None

    def plot(self, varname: str = None, ax: plt.Axes = None) -> plt.Axes:  # pragma: no cover
        canvas = SeabornCanvas()
        ax = ax or canvas.ax
        ax = sns.scatterplot(data=self.data, x=self.x, y=self.y, ax=ax)

        ax.set_title(
            f"Pearson's Test for Correlation\n{self.result}",
            fontsize=canvas.fontsize_title,
        )

        ax.set_xlabel(self.x)
        ax.set_ylabel(self.y)
        plt.tight_layout()
        return ax


# ------------------------------------------------------------------------------------------------ #
#                                          TEST                                                    #
# ------------------------------------------------------------------------------------------------ #
class PearsonCorrelationTest(StatisticalTest):
    __id = "pearson"

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
        data, x, y = self._parse_arguments(data=data, x=x, y=y)

        r, pvalue = stats.pearsonr(data[x], data[y])

        if pvalue > self._alpha:  # pragma: no cover
            inference = f"The pvalue {round(pvalue,2)} is greater than level of significance {int(self._alpha*100)}%; therefore, the null hypothesis is not rejected. Correlation between {x} and {y} is not significant."
        else:
            inference = f"The pvalue {round(pvalue,2)} is less than level of significance {int(self._alpha*100)}%; therefore, the null hypothesis is rejected. Correlation between {x} and {y} is significant."

        # Create the result object.
        self._result = PearsonCorrelationResult(
            test=self._profile.name,
            H0=self._profile.H0,
            statistic=self._profile.statistic,
            hypothesis=self._profile.hypothesis,
            value=r,
            pvalue=pvalue,
            result=f"r({len(data)})={round(r,2)}, {self._report_pvalue(pvalue)} {self._report_alpha()}",
            data=data,
            x=x,
            y=y,
            inference=inference,
            alpha=self._alpha,
        )

    def _report_results(self, statistic: float, pvalue: float, dof: float, n: int) -> str:
        return f"Pearson Correlation Test\na: X\u00b2({dof}, N={n})={statistic}, p={self._report_pvalue(pvalue)}."

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
            return f"Very high {direction} correlation."
        elif r >= 0.70:
            return f"High {direction} correlation."
        elif r >= 0.5:
            return f"Moderate {direction} correlation."
        elif r >= 0.3:
            return f"Low {direction} correlation."
        else:
            return f"Negligible correlation."
