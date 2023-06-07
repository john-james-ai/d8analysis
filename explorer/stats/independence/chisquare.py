#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Enter Project Name in Workspace Settings                                            #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.11                                                                             #
# Filename   : /explorer/stats/independence/chisquare.py                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : Enter URL in Workspace Settings                                                     #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday May 29th 2023 03:00:39 am                                                    #
# Modified   : Wednesday June 7th 2023 05:03:06 pm                                                 #
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

from explorer.stats.profile import StatTestProfileTwo
from explorer.stats.base import StatTestResult, StatisticalTest, StatTestProfile
from explorer.visual.config import Canvas


# ------------------------------------------------------------------------------------------------ #
#                                     TEST RESULT                                                  #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class ChiSquareIndependenceResult(StatTestResult):
    dof: int = None
    data: Union[pd.DataFrame, np.ndarray, pd.Series] = None
    observed: Union[pd.DataFrame, np.ndarray, pd.Series] = None
    expected: Union[pd.DataFrame, np.ndarray, pd.Series] = None

    def plot(self, varname: str = None, ax: plt.Axes = None) -> plt.Axes:  # pragma: no cover
        canvas = Canvas()
        ax = ax or canvas.ax
        x = np.linspace(stats.chi2.ppf(0.01, self.dof), stats.chi2.ppf(0.999, self.dof), 100)
        y = stats.chi2.pdf(x, self.dof)
        ax = sns.lineplot(x=x, y=y, markers=False, dashes=False, sort=True, ax=ax)

        self._fill_curve(ax)

        ax.set_title(
            f"X\u00b2 Test of Independence\n{self.result}",
            fontsize=canvas.fontsize_title,
        )

        ax.set_xlabel(r"$X^2$")
        ax.set_ylabel("Probability Density")
        plt.tight_layout()
        return ax

    def plot_obs_exp(self) -> pd.DataFrame:  # pragma: no cover
        """Plots observed vs expected frequencies and returns the data in DataFrame format."""
        dfc = self._combine_contingency_tables()

        dfo = dfc[dfc["Dataset"] == "Observed"]
        dfe = dfc[dfc["Dataset"] == "Expected"]

        canvas = Canvas(nrows=2, ncols=1)
        (ax1, ax2) = canvas.axs
        fig = canvas.fig

        ax1 = sns.barplot(
            data=dfo,
            x=dfc.columns[1],
            y="Count",
            hue=dfc.columns[2],
            palette=canvas.palette,
            ax=ax1,
        )
        ax2 = sns.barplot(
            data=dfe,
            x=dfc.columns[1],
            y="Count",
            hue=dfc.columns[2],
            palette=canvas.palette,
            ax=ax2,
        )

        title = f"X\u00b2 Test of Independence\n{self.result}\nObserved vs Expected Frequencies"
        fig.suptitle(title, fontsize=canvas.fontsize_title)
        ax1.set_title("Observed", fontsize=canvas.fontsize_title)
        ax2.set_title("Expected", fontsize=canvas.fontsize_title)
        plt.tight_layout()

        dfo = dfo.drop(columns=["Dataset"])
        dfe = dfe.drop(columns=["Dataset"])

        dfo.rename(columns={"Count": "Observed"}, inplace=True)
        dfe.rename(columns={"Count": "Expected"}, inplace=True)

        dfo.loc[:, "Expected"] = dfe["Expected"].values

        return dfo

    def _combine_contingency_tables(self):
        """Combines the observed and expected frequencies into a dataframe that can be plotted.

        Returns:
            A DataFrame containing the original dataframe, with two columns added:
                dataset: Designating whether the data is 'observed', or 'expected'.
                count: The observed and expected frequencies.

        """
        df = self.data
        obs = self.observed
        exp = self.expected

        d = {}
        for col, data in zip(obs.elements[0], obs.count):
            d[col] = data
        dfo = pd.DataFrame(d)
        dfo.index = obs.elements[1]
        dfo.reset_index(inplace=True, names=[df.columns[1]])
        dfo["Dataset"] = "Observed"

        d = {}
        for col, data in zip(obs.elements[0], exp):
            d[col] = data
        dfe = pd.DataFrame(d)
        dfe.index = obs.elements[1]
        dfe.reset_index(inplace=True, names=[df.columns[1]])
        dfe["Dataset"] = "Expected"

        dfc = pd.concat([dfo, dfe], axis=0, ignore_index=True)
        print(dfc)
        dfc = pd.melt(
            dfc,
            id_vars=["Dataset", df.columns[1]],
            value_vars=list(obs.elements[0]),
            var_name=df.columns[0],
            value_name="Count",
        )
        return dfc


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
    def data(self) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Returns the data tested"""
        return self._data

    @property
    def result(self) -> StatTestResult:
        """Returns a Statistical Test Result object."""
        return self._result

    def __call__(self, data: pd.DataFrame) -> None:
        """Performs the statistical test and creates a result object.

        Args:
            data (pd.DataFrame) A pandas dataframe containing the two nominal/categorical
                variable columns to be tested.

        """
        self._data = data
        s1 = data.iloc[:, 0]
        s2 = data.iloc[:, 1]
        obs = stats.contingency.crosstab(s1, s2)
        statistic, pvalue, dof, exp = stats.chi2_contingency(obs.count)

        if pvalue > self._alpha:  # pragma: no cover
            gtlt = ">"
            inference = f"The pvalue {round(pvalue,2)} is greater than level of significance {self._alpha}; therefore, the null hypothesis is not rejected. The data have the expected frequencies."
        else:
            gtlt = "<"
            inference = f"The pvalue {round(pvalue,2)} is less than level of significance {self._alpha}; therefore, the null hypothesis is rejected. The data do not have the expected frequencies."

        # Create the result object.
        self._result = ChiSquareIndependenceResult(
            test=self._profile.name,
            H0=self._profile.H0,
            statistic="X\u00b2",
            hypothesis=self._profile.hypothesis,
            dof=dof,
            value=statistic,
            pvalue=pvalue,
            result=f"X\u00b2({dof}, N={len(data)})={round(statistic,2)}, p{gtlt}{self._alpha}",
            data=data,
            observed=obs,
            expected=exp,
            inference=inference,
            alpha=self._alpha,
        )
