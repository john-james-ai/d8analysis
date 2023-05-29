#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Explorer                                                                            #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.10                                                                             #
# Filename   : /explorer/analysis/univariate.py                                                    #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/explorer                                           #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday May 26th 2023 06:22:28 pm                                                    #
# Modified   : Sunday May 28th 2023 04:01:04 pm                                                    #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Univariate Analysis Module"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats

from explorer.base import Analysis, StatTestOne, StatTestOneGoF, Canvas
from explorer.stats.distribution import DISTRIBUTIONS, DistGen


# ------------------------------------------------------------------------------------------------ #
class QualitativeOne(Analysis):
    """Univariate Analysis class for qualitative variables.

    Args:
        data (pd.DataFrame): A DataFrame containing one or more qualitative variables.
        name (str): Name of the dataset
    """

    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__(data=data)

    def describe(self, name: str) -> pd.DataFrame:
        """Returns descriptive statistics for a qualitative variable.

        This method assumes nominal variables, unless indicated as ordinal.

        Args:
            name (str): Name of variable. Optional
        """
        self._check_name(name=name)
        return self._data[name].describe().to_frame()

    def frequency(self, name: str) -> pd.DataFrame:
        """Frequency Analysis

        Method provides analysis of frequencies for levels of a qualitative
        variable in terms of count, percent, and cumulative percent.

        Args:
            name (str): Name of variable.
        """
        self._check_name(name=name)

        counts = self._data[name].value_counts().to_frame()
        pct = counts / counts.sum() * 100
        cumpct = counts.cumsum() / counts.sum() * 100
        result = pd.concat([counts, pct, cumpct], axis=1)
        result.columns = ["Count", "%", "Cum %"]
        return result

    def test_distribution(self, name: str, expected: pd.Series = None) -> StatTestOne:
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
            chisq, p = stats.chisquare(f_obs=observed.values, f_exp=expected.values)
        else:
            chisq, p = stats.chisquare(f_obs=observed.values)
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
        """Renders a Count Plot of Frequencies by qualitative Level

        Args:
            name (str): Variable name.
            title (str): Title for the plot. Optional.
            ax (plt.Axes): An axes object. Optional.
            kwargs (dict): Keyword arguments to be passed to the underlying countplot.
        """
        self._check_name(name=name)

        ax = ax or Canvas().ax

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


# ------------------------------------------------------------------------------------------------ #
class QuantitativeOne(Analysis):
    """Univariate Analysis class for quantitative variables.

    Args:
        data (pd.DataFrame): A DataFrame containing one or more quantitative variables.
        name (str): Name of the dataset
    """

    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__(data=data)

    def describe(self, name: str) -> pd.DataFrame:
        """Returns descriptive statistics for a quantitative variable.

        This method assumes nominal variables, unless indicated as ordinal.

        Args:
            name (str): Name of variable. Optional
        """
        self._check_name(name=name)
        return self._data[name].describe().to_frame().T

    def test_distribution(
        self,
        name: str,
        distribution: str = "normal",
        approx: bool = False,
        discrete: bool = False,
        **kwargs,
    ) -> StatTestOne:
        """Tests whether the observed frequency distribution varies significantly from theoretical expectations.

        Args:
            name (str): Name of variable.
            distribution (pd.Series): Expected distribution.
            approx (bool): Method for Kolmogorov-Smirnof Goodness of Fit
            discrete (bool): True if data are discrete.
            kwargs (dict): Keyword arguments for parameters of the scipy.stats
                distribution distribution random number generators.
        """
        self._check_name(name=name)
        if distribution == "normal" and self._data.shape[0] < 50:
            return self._shapiro_wilk(name=name)
        elif discrete:
            return self._chisq(name=name, distribution=distribution)
        else:
            return self._kstest(name=name, distribution=distribution, approx=approx)

    def plot_distribution(
        self,
        name: str,
        distribution: str = "normal",
        title: str = None,
        ax: plt.Axes = None,
        **kwargs,
    ) -> None:  # pragma: no cover
        """Renders a Count Plot of Frequencies by quantitative Level

        Args:
            name (str): Variable name.
            distribution (str): The expected distribution
            title (str): Title for the plot. Optional.
            ax (plt.Axes): An axes object. Optional.
            kwargs (dict): Keyword arguments to be passed to the underlying countplot.
        """
        self._check_name(name=name)

        ax = ax or Canvas().ax
        self.plot_pdf(name=name, distribution=distribution, ax=ax)

        if title is not None:
            ax.set_title(title)

        plt.tight_layout()

    def gofplot(self, name: str, distribution: str, ax: plt.Axes = None) -> None:
        """Plots empirical vis-a-vis theoretical probability distribution

        Args:
           name (str): Variable name.
           distribution (str): The expected distribution
           ax (plt.Axes): An axes object. Optional
        """
        ax = ax or Canvas().ax

        # Create Observed Distribution Dataset
        observed = self._data[name].to_frame()
        observed["Distribution"] = "Sample"

        # Create Theoretical Distribution
        dg = DistGen()
        rvs = dg(data=self._data[name].values, distribution=distribution)
        d = {name: rvs}
        theoretical = pd.DataFrame(data=d)
        theoretical["Distribution"] = "Theoretical"

        df = pd.concat([observed, theoretical], axis=0)

        ax = sns.histplot(data=df, x=name, hue="Distribution", kde=True, ax=ax, stat="probability")
        ax.set_title(
            f"{name}  Distribution Goodness of Fit to the {distribution.capitalize()} Disribution"
        )

    def qqplot(self, name: str, distribution: str = "normal", ax: plt.Axes = None) -> None:
        """Plots empirical and theoretical quantiles

        Args:
           name (str): Variable name.
           distribution (str): The expected distribution
           ax (plt.Axes): An axes object. Optional
        """

        ax = ax or Canvas().ax

        # Generate theoretical distribution
        dg = DistGen()
        rvs = dg(data=self._data[name].values, distribution=distribution)

        # Compute quantiles
        percentiles = np.linspace(0, 100, len(rvs) + 1)
        q_observed = np.percentile(a=self._data[name].values, q=percentiles)
        q_theoretical = np.percentile(a=rvs, q=percentiles)

        d = {"Sample Quantiles": q_observed, "Theoretical Quantiles": q_theoretical}
        df = pd.DataFrame(data=d)

        ax = sns.scatterplot(data=df, x="Theoretical Quantiles", y="Sample Quantiles")

        title = f"{name} QQ Plot\n{distribution.capitalize()} Distribution"
        ax.set_title(title)

        # Compute line
        line = np.linspace(
            np.min((q_observed.min(), q_theoretical.min())),
            np.max((q_observed.max(), q_theoretical.max())),
        )
        ax = sns.lineplot(x=line, y=line, ax=ax)

    def cdfplot(self, name: str, distribution: str, ax: plt.Axes = None) -> None:
        """Plots the empirical cdf  a histogram of the variable of interest."""

        ax = ax or Canvas().ax

        # Create Observed Distribution Dataset
        observed = self._data[name].to_frame()
        observed["Distribution"] = "Sample"

        # Create Theoretical Distribution
        dg = DistGen()
        rvs = dg(data=self._data[name].values, distribution=distribution)
        d = {name: rvs}
        theoretical = pd.DataFrame(data=d)
        theoretical["Distribution"] = "Theoretical"

        df = pd.concat([observed, theoretical], axis=0)

        ax = sns.kdeplot(data=df, x=name, hue="Distribution", cumulative=True, ax=ax)
        ax.set_title(
            f"{name} Empirical vs Theoretical {distribution.capitalize()} Cumulative Distribution"
        )

    def boxplot(self, name: str, distribution: str, ax: plt.Axes = None) -> None:
        """Renders a boxplot of both the sample and theoretical distribution."""
        ax = ax or Canvas().ax

        # Create Observed Distribution Dataset
        observed = self._data[name].to_frame()
        observed["Distribution"] = "Sample"

        # Create Theoretical Distribution
        dg = DistGen()
        rvs = dg(data=self._data[name].values, distribution=distribution)
        d = {name: rvs}
        theoretical = pd.DataFrame(data=d)
        theoretical["Distribution"] = "Theoretical"

        df = pd.concat([observed, theoretical], axis=0)

        ax = sns.boxplot(data=df, x=name, y="Distribution", ax=ax)
        ax.set_title(f"Sample and Theoretical {distribution.capitalize()} Distribution of {name}")

    def _chisq(self, name: str, distribution: str, kwargs) -> StatTestOneGoF:
        """Performs a Chi-Square goodness of fit for a discrete random variable."""
        # Generate distribution distribution
        distribution_values = DISTRIBUTIONS[distribution].rvs(**kwargs)
        statistic, pvalue = stats.chisquare(f_obs=self._data[name], f_exp=distribution_values)
        return StatTestOneGoF(
            analyzer=self.__class__.__name__,
            test="Chi-Square Goodness of Fit",
            distribution=distribution,
            statistic=statistic,
            pvalue=pvalue,
            x=name,
        )

    def _normaltest(self, name: str) -> StatTestOneGoF:
        """Computes a ShapiroWilk test for normality"""
        statistic, pvalue = stats.normaltest(a=self._data[name])
        return StatTestOneGoF(
            analyzer=self.__class__.__name__,
            test="D'Agostino and Pearson's Test for Normality",
            distribution="normal",
            method="exact",
            statistic=statistic,
            pvalue=pvalue,
            x=name,
        )

    def _shapiro_wilk(self, name: str) -> StatTestOne:
        """Computes a ShapiroWilk test for normality"""
        statistic, pvalue = stats.shapiro(x=self._data[name])
        return StatTestOneGoF(
            analyzer=self.__class__.__name__,
            test="Shapiro-Wilk Test for Normality",
            distribution="normal",
            statistic=statistic,
            pvalue=pvalue,
            x=name,
        )

    def _kstest(self, name: str, distribution: str, approx: bool = False) -> StatTestOne:
        """Computes a Kolmogorov-Smirnov test for goodness of fit for continuous variables"""

        distribution = DISTRIBUTIONS[distribution]

        method = "approx" if approx is True else "exact"

        statistic, pvalue = stats.kstest(
            rvs=self._data[name], cdf=distribution.cdf, alternative="two-sided", method=method
        )
        self._logger.debug(f"\nKSTest\nStatistic: {statistic}\nPvalue: {pvalue}")

        return StatTestOneGoF(
            analyzer=self.__class__.__name__,
            test="Kolmogorov-Smirnov Goodness-of-Fit",
            distribution=distribution,
            method=method,
            statistic=statistic,
            pvalue=pvalue,
            x=name,
        )
