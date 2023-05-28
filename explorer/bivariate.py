#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Explorer                                                                            #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.10                                                                             #
# Filename   : /explorer/bivariate.py                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/explorer                                           #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday May 26th 2023 06:22:28 pm                                                    #
# Modified   : Saturday May 27th 2023 09:49:22 am                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Univariate Analysis Module"""
from scipy.stats import chisquare
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from explorer.base import Analysis, StatTestOne, Canvas


# ------------------------------------------------------------------------------------------------ #
class CategoricalTwo(Analysis):
    """Bivariate Analysis class for categorical variables.

    Args:
        data (pd.DataFrame): A DataFrame containing one or more categorical variables.
        name (str): Name of the dataset
    """

    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__(data=data)

    def describe(self, a: str, b: str) -> pd.DataFrame:
        """Returns descriptive statistics for two categorical variables.

        This method assumes nominal variables, unless indicated as ordinal.

        Args:
            name (str): Name of variable. Optional
        """
        self._check_name(name=a)
        self._check_name(name=b)
        return self._data[[a, b]].describe()

    def frequency(self, a: str, b: str) -> pd.DataFrame:
        """Frequency Analysis for two categorical variables.

        Method provides analysis of frequencies for levels of two categorical
        variables.

        Args:
            a (str): Name of a categorical variable.
            b (str): Name of
        """
        self._check_name(name=a)
        self._check_name(name=b)
        return pd.crosstab(index=self._data[a], columns=self._data[b])

    def test_independence(self, a: str, b: str) -> StatTestOne:
        """Tests whether the observed frequency distribution varies significantly from theoretical expectations.

        Args:
            name (str): Name of variable.
            expected (pd.Series): Series containing expected frequencies, where the index contains the
                associated category levels.
        """
        self._check_name(name=name)

        # Get observed frequencies
        observed = self._data[name].value_counts(sort=False)
        # if expected distribution provided, validate and sort
        if expected is not None:
            observed, expected = self._valsort(observed=observed, expected=expected)
            chisq, p = chisquare(f_obs=observed.values, f_exp=expected.values)
        else:
            chisq, p = chisquare(f_obs=observed.values)
        return StatTestOne(
            analyzer=self.__class__.__name__,
            test="Chi-Square Goodness of Fit",
            x=name,
            statistic=chisq,
            pvalue=p,
        )

    def plot_distribution(
        self, name: str, title: str = None, ax: plt.Axes = None, **kwargs
    ) -> None:  # pragma: no cover
        """Renders a Count Plot of Frequencies by Categorical Level

        Args:
            name (str): Variable name.
            title (str): Title for the plot. Optional.
            ax (plt.Axes): An axes object. Optional.
            kwargs (dict): Keyword arguments to be passed to the underlying countplot.
        """
        self._check_name(name=name)

        ax = ax or self._get_axes()

        ax = sns.countplot(data=self._data, x=name, ax=ax, palette=Canvas.palette, **kwargs)

        if title is not None:
            ax.set_title(title)

        # Add counts to plot
        ax.bar_label(ax.containers[0])

        plt.tight_layout()

    def _valsort(self, observed: pd.Series, expected: pd.Series) -> pd.Series:
        """Validates and sorts observed and expected frequences."""
        category_mismatch = "Categories in observed and expected do not match."

        if not isinstance(expected, pd.Series):
            expected = pd.Series(expected)

        observed_categories = observed.index.sort_values()
        expected_categories = expected.index.sort_values()

        if not observed_categories.equals(expected_categories):
            self._logger.error(category_mismatch)
            raise ValueError(category_mismatch)

        observed.sort_index(ascending=True, inplace=True)
        expected.sort_index(ascending=True, inplace=True)
        return observed, expected
