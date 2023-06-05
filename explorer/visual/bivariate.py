#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Enter Project Name in Workspace Settings                                            #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.10                                                                             #
# Filename   : /explorer/visual/bivariate.py                                                       #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : Enter URL in Workspace Settings                                                     #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday May 26th 2023 06:22:28 pm                                                    #
# Modified   : Monday June 5th 2023 06:20:30 pm                                                    #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Univariate Analysis Module"""
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from explorer.analysis.base import Analysis
from explorer.stats.result import StatTestResultTwo
from explorer.visual.config import Canvas, PlotConfig


# ------------------------------------------------------------------------------------------------ #
#                                  QUALITATIVE TWO                                                 #
# ------------------------------------------------------------------------------------------------ #
class QualitativeTwo(Analysis):
    """Bivariate Analysis in which at least one variables is qualitative.

    Args:
        data (pd.DataFrame): A DataFrame containing one or more categorical variables.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__(data=data)

    def describe(self, a: str, b: str) -> pd.DataFrame:
        """Returns descriptive statistics for two variables.

        Args:
            a (str): Name of a categorical variable in the dataset
            b (str): Name of a categorical or numeric variable in the dataset.
        """
        self._check_name(name=a)
        self._check_name(name=b)
        dtypes = {a: "categorical"}
        self._check_dtypes(dtypes)
        return self._data[[a, b]].groupby(by=a).describe()

    def frequency(self, a: str, b: str) -> pd.DataFrame:
        """Frequency Analysis for two categorical variables.

        Method provides analysis of frequencies for levels of two categorical
        variables.

        Args:
            a (str): Name of a categorical variable.
            b (str): Name of a categorical variable
        """
        self._check_name(name=a)
        self._check_name(name=b)
        dtypes = {a: "categorical", b: "categorical"}
        self._check_dtypes(dtypes)

        if self._is_categorical(name=a) and self._is_categorical(name=b):
            return pd.crosstab(index=self._data[a], columns=self._data[b])
        else:  # pragma: no cover
            msg = "One or more variables are not categorical. Both a, and b, must name categorical data."
            self._logger.error(msg)
            raise TypeError(msg)

    def test_independence(self, a: str, b: str) -> StatTestResultTwo:
        """Tests whether the observed frequency distribution varies significantly from theoretical expectations.

        Args:
            a (str): Name of a categorical variable.
            b (str): Name of a categorical or numeric variable
        """
        self._check_name(name=a)
        self._check_name(name=b)
        dtypes = {a: "categorical"}
        self._check_dtypes(dtypes)

        if self._is_categorical(a) and self._is_categorical(b):
            obs = stats.contingency.crosstab(self._data[a].values, self._data[b].values)
            res = stats.chi2_contingency(observed=obs.count)
            return StatTestResultTwo(
                analyzer=self.__class__.__name__,
                test="Chi-Square Test of Independence",
                a=a,
                b=b,
                statistic=res.statistic,
                pvalue=res.pvalue,
            )
        else:
            groups = self._data.groupby(a)[b].apply(np.array).values
            res = stats.f_oneway(*groups)
            return StatTestResultTwo(
                analyzer=self.__class__.__name__,
                test="One-Way ANOVA Test of Independence",
                a=a,
                b=b,
                statistic=res.statistic,
                pvalue=res.pvalue,
            )

    def plot_distribution(
        self,
        a: str,
        b: str,
        title: str = None,
        wrap_xticklabels: bool = True,
        wrap_yticklabels: bool = False,
        barplot_config: PlotConfig = None,
        boxplot_config: PlotConfig = None,
        kdeplot_config: PlotConfig = None,
        pointplot_config: PlotConfig = None,
        legend_config: PlotConfig = None,
    ) -> None:  # pragma: no cover
        """Renders a Count Plot of Frequencies by Categorical Level

        Args:
            a (str): Name of a categorical variable.
            b (str): Name of a numeric variable
            title (str): Title for the plot. Optional.

        """
        self._check_name(a)
        self._check_name(b)

        dtypes = {a: "categorical", b: "numeric"}
        self._check_dtypes(dtypes)

        canvas = Canvas(figsize=(6, 3), nrows=2, ncols=2)
        fig = canvas.fig
        axes = canvas.axs.flat

        if title:
            fig.suptitle(title)

        self.barplot(
            a=a,
            b=b,
            ax=axes[0],
            config={} if not barplot_config else barplot_config.as_dict(),
            legend_config={} if not legend_config else legend_config.as_dict(),
        )
        self.boxplot(
            a=a,
            b=b,
            ax=axes[1],
            config={} if not boxplot_config else boxplot_config.as_dict(),
            legend_config={} if not legend_config else legend_config.as_dict(),
        )
        self.kdeplot(
            a=a,
            b=b,
            ax=axes[2],
            config={} if not kdeplot_config else kdeplot_config.as_dict(),
            legend_config={} if not legend_config else legend_config.as_dict(),
        )
        self.pointplot(
            a=a,
            b=b,
            ax=axes[3],
            config={} if not pointplot_config else pointplot_config.as_dict(),
            legend_config={} if not legend_config else legend_config.as_dict(),
        )

        if wrap_xticklabels:
            axes = self._wrap_ticklabels(axis="x", axes=axes)
        if wrap_yticklabels:
            axes = self._wrap_ticklabels(axis="y", axes=axes)

        plt.tight_layout()

    def countplot(
        self,
        a: str,
        b: str,
        ax: plt.Axes = None,
        config: dict = {},
        legend_config: dict = (),
    ) -> None:  # pragma: no cover
        """Plots counts of observations for two categories.

        Args:
            a (str): Name of a categorical variable.
            b (str): Name of a categorical variable
            ax (plt.Axes): A matplotlib Axes object. Optional
            legend_config (dict): Parameters to configure the Axes legend
            config (dict): Parameters for the plot method
        """
        self._check_name(a)
        self._check_name(b)
        dtypes = {a: "categorical", b: "categorical"}
        self._check_dtypes(dtypes)

        ax = ax or Canvas().ax

        ax = sns.countplot(
            data=self._data,
            x=a,
            hue=b,
            ax=ax,
            **config,
        )

        if config.get("dodge", None):
            for container in ax.containers:
                ax.bar_label(container)

        title = f"Frequency Distribution\n{a} and {b}."
        ax.set_title(title)
        plt.tight_layout()

    def barplot(
        self,
        a: str,
        b: str,
        ax: plt.Axes = None,
        config: dict = {},
        legend_config: dict = [],
    ) -> None:  # pragma: no cover
        """Produces a stacked bar plot for two categorical variables

        Args:
            a (str): Name of a categorical variable.
            b (str): Name of a categorical or numeric variable
            ax (plt.Axes): A matplotlib Axes object. Optional
            legend_config (dict): Parameters to configure the Axes legend
            config (dict): Parameters for the plot method
        """
        self._logger.debug(ax)
        ax = ax or Canvas().ax

        self._check_name(a)
        self._check_name(b)
        dtypes = {a: "categorical", b: "numeric"}
        self._check_dtypes(dtypes)

        ax = sns.barplot(
            data=self._data,
            x=a,
            y=b,
            ax=ax,
            **config,
        )
        title = f"Distribution of {b} by {a}"

        ax.set_title(title)
        plt.tight_layout()

    def boxplot(
        self,
        a: str,
        b: str,
        ax: plt.Axes = None,
        config: dict = {},
        legend_config: dict = [],
    ) -> None:  # pragma: no cover
        """Renders a boxplot of the categorical, numerical distribution.
        Args:
            a (str): Name of a categorical variable.
            b (str): Name of a numeric variable
            ax (plt.Axes): A matplotlib Axes object. Optional
            config (dict): Parameters for the plot method
        """
        ax = ax or Canvas().ax

        self._check_name(a)
        self._check_name(b)
        dtypes = {a: "categorical", b: "numeric"}
        self._check_dtypes(dtypes)

        ax = sns.boxplot(
            data=self._data,
            x=b,
            y=a,
            ax=ax,
            **config,
        )

        title = f"Distribution of {b} by {a}"
        ax.set_title(title)
        plt.tight_layout()

    def pointplot(
        self,
        a: str,
        b: str,
        ax: plt.Axes = None,
        config: dict = [],
        legend_config: dict = [],
    ) -> None:  # pragma: no cover
        """Renders a boxplot of the categorical, numerical distribution.
        Args:
            a (str): Name of a categorical variable.
            b (str): Name of a numeric variable
            ax (plt.Axes): A matplotlib Axes object. Optional
        """
        ax = ax or Canvas().ax

        self._check_name(a)
        self._check_name(b)
        dtypes = {a: "categorical", b: "numeric"}
        self._check_dtypes(dtypes)

        ax = sns.pointplot(data=self._data, x=a, y=b, ax=ax, **config)

        title = f"Central Tendency of {b} by {a}"
        ax.set_title(title)
        plt.tight_layout()

    def histplot(
        self,
        a: str,
        b: str,
        ax: plt.Axes = None,
        config: dict = {},
        legend_config: dict = [],
    ) -> None:  # pragma: no cover
        """Renders a boxplot of the categorical, numerical distribution.
        Args:
            a (str): Name of a categorical variable.
            b (str): Name of a numeric variable
            ax (plt.Axes): A matplotlib Axes object. Optional
            legend_config (dict): Dictionary containing parameters for the legend.
            kwargs (dict): Keyword arguments passed to underlying plotting method.
        """
        ax = ax or Canvas().ax

        self._check_name(a)
        self._check_name(b)
        dtypes = {a: "categorical", b: "numeric"}
        self._check_dtypes(dtypes)

        ax = sns.histplot(
            data=self._data,
            x=b,
            hue=a,
            ax=ax,
            **config,
        )

        title = f"Histogram of {b} by {a}"
        ax.set_title(title)
        plt.tight_layout()

    def kdeplot(
        self,
        a: str,
        b: str,
        ax: plt.Axes = None,
        config: dict = {},
        legend_config: dict = [],
    ) -> None:  # pragma: no cover
        """Renders a boxplot of the categorical, numerical distribution.
        Args:
            a (str): Name of a categorical variable.
            b (str): Name of a numeric variable
            ax (plt.Axes): A matplotlib Axes object. Optional
            legend_config (dict): Dictionary containing parameters for the legend.
            kwargs (dict): Keyword arguments passed to underlying plotting method.
        """
        ax = ax or Canvas().ax

        self._check_name(a)
        self._check_name(b)
        dtypes = {a: "categorical", b: "numeric"}
        self._check_dtypes(dtypes)

        ax = sns.kdeplot(
            data=self._data,
            x=b,
            hue=a,
            ax=ax,
            **config,
        )

        plt.legend(**legend_config)
        title = f"Kernel Density Estimate (KDE) of {b} by {a}"
        ax.set_title(title)
        plt.tight_layout()


# ------------------------------------------------------------------------------------------------ #
#                                  QUALITATIVE TWO                                                 #
# ------------------------------------------------------------------------------------------------ #
class QuantitativeTwo(Analysis):
    """Bivariate of numeric variables.

    Args:
        data (pd.DataFrame): A DataFrame containing one or more categorical variables.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__(data=data)

    def describe(self, a: str, b: str) -> pd.DataFrame:
        """Returns descriptive statistics for two variables.

        Args:
            a (str): Name of a categorical variable in the dataset
            b (str): Name of a categorical or numeric variable in the dataset.
        """
        self._check_name(name=a)
        self._check_name(name=b)
        dtypes = {a: "numeric", b: "numeric"}
        self._check_dtypes(dtypes)
        return self._data[[a, b]].describe()

    def test_association(self, a: str, b: str) -> StatTestResultTwo:
        """Tests association between two continuous variables.

        Args:
            a (str): Name of a numeric variable in the dataset
            b (str): Name of a numeric variable in the dataset
        """
        self._check_name(name=a)
        self._check_name(name=b)
        dtypes = {a: "numeric", b: "numeric"}
        self._check_dtypes(dtypes)

    def plot_distribution(
        self,
        a: str,
        b: str,
        title: str = None,
        wrap_xticklabels: bool = True,
        wrap_yticklabels: bool = False,
        barplot_config: PlotConfig = None,
        boxplot_config: PlotConfig = None,
        kdeplot_config: PlotConfig = None,
        pointplot_config: PlotConfig = None,
        legend_config: PlotConfig = None,
    ) -> None:  # pragma: no cover
        """Renders a Count Plot of Frequencies by Categorical Level

        Args:
            a (str): Name of a categorical variable.
            b (str): Name of a numeric variable
            title (str): Title for the plot. Optional.

        """
        self._check_name(a)
        self._check_name(b)

        dtypes = {a: "categorical", b: "numeric"}
        self._check_dtypes(dtypes)

        canvas = Canvas(figsize=(6, 3), nrows=2, ncols=2)
        fig = canvas.fig
        axes = canvas.axs.flat

        if title:
            fig.suptitle(title)

        self.barplot(
            a=a,
            b=b,
            ax=axes[0],
            config={} if not barplot_config else barplot_config.as_dict(),
            legend_config={} if not legend_config else legend_config.as_dict(),
        )
        self.boxplot(
            a=a,
            b=b,
            ax=axes[1],
            config={} if not boxplot_config else boxplot_config.as_dict(),
            legend_config={} if not legend_config else legend_config.as_dict(),
        )
        self.kdeplot(
            a=a,
            b=b,
            ax=axes[2],
            config={} if not kdeplot_config else kdeplot_config.as_dict(),
            legend_config={} if not legend_config else legend_config.as_dict(),
        )
        self.pointplot(
            a=a,
            b=b,
            ax=axes[3],
            config={} if not pointplot_config else pointplot_config.as_dict(),
            legend_config={} if not legend_config else legend_config.as_dict(),
        )

        if wrap_xticklabels:
            axes = self._wrap_ticklabels(axis="x", axes=axes)
        if wrap_yticklabels:
            axes = self._wrap_ticklabels(axis="y", axes=axes)

        plt.tight_layout()

    def countplot(
        self,
        a: str,
        b: str,
        ax: plt.Axes = None,
        config: dict = {},
        legend_config: dict = (),
    ) -> None:  # pragma: no cover
        """Plots counts of observations for two categories.

        Args:
            a (str): Name of a categorical variable.
            b (str): Name of a categorical variable
            ax (plt.Axes): A matplotlib Axes object. Optional
            legend_config (dict): Parameters to configure the Axes legend
            config (dict): Parameters for the plot method
        """
        self._check_name(a)
        self._check_name(b)
        dtypes = {a: "categorical", b: "categorical"}
        self._check_dtypes(dtypes)

        ax = ax or Canvas().ax

        ax = sns.countplot(
            data=self._data,
            x=a,
            hue=b,
            ax=ax,
            **config,
        )

        if config.get("dodge", None):
            for container in ax.containers:
                ax.bar_label(container)

        title = f"Frequency Distribution\n{a} and {b}."
        ax.set_title(title)
        plt.tight_layout()

    def barplot(
        self,
        a: str,
        b: str,
        ax: plt.Axes = None,
        config: dict = {},
        legend_config: dict = [],
    ) -> None:  # pragma: no cover
        """Produces a stacked bar plot for two categorical variables

        Args:
            a (str): Name of a categorical variable.
            b (str): Name of a categorical or numeric variable
            ax (plt.Axes): A matplotlib Axes object. Optional
            legend_config (dict): Parameters to configure the Axes legend
            config (dict): Parameters for the plot method
        """
        self._logger.debug(ax)
        ax = ax or Canvas().ax

        self._check_name(a)
        self._check_name(b)
        dtypes = {a: "categorical", b: "numeric"}
        self._check_dtypes(dtypes)

        ax = sns.barplot(
            data=self._data,
            x=a,
            y=b,
            ax=ax,
            **config,
        )
        title = f"Distribution of {b} by {a}"

        ax.set_title(title)
        plt.tight_layout()

    def boxplot(
        self,
        a: str,
        b: str,
        ax: plt.Axes = None,
        config: dict = {},
        legend_config: dict = [],
    ) -> None:  # pragma: no cover
        """Renders a boxplot of the categorical, numerical distribution.
        Args:
            a (str): Name of a categorical variable.
            b (str): Name of a numeric variable
            ax (plt.Axes): A matplotlib Axes object. Optional
            config (dict): Parameters for the plot method
        """
        ax = ax or Canvas().ax

        self._check_name(a)
        self._check_name(b)
        dtypes = {a: "categorical", b: "numeric"}
        self._check_dtypes(dtypes)

        ax = sns.boxplot(
            data=self._data,
            x=b,
            y=a,
            ax=ax,
            **config,
        )

        title = f"Distribution of {b} by {a}"
        ax.set_title(title)
        plt.tight_layout()

    def pointplot(
        self,
        a: str,
        b: str,
        ax: plt.Axes = None,
        config: dict = [],
        legend_config: dict = [],
    ) -> None:  # pragma: no cover
        """Renders a boxplot of the categorical, numerical distribution.
        Args:
            a (str): Name of a categorical variable.
            b (str): Name of a numeric variable
            ax (plt.Axes): A matplotlib Axes object. Optional
        """
        ax = ax or Canvas().ax

        self._check_name(a)
        self._check_name(b)
        dtypes = {a: "categorical", b: "numeric"}
        self._check_dtypes(dtypes)

        ax = sns.pointplot(data=self._data, x=a, y=b, ax=ax, **config)

        title = f"Central Tendency of {b} by {a}"
        ax.set_title(title)
        plt.tight_layout()

    def histplot(
        self,
        a: str,
        b: str,
        ax: plt.Axes = None,
        config: dict = {},
        legend_config: dict = [],
    ) -> None:  # pragma: no cover
        """Renders a boxplot of the categorical, numerical distribution.
        Args:
            a (str): Name of a categorical variable.
            b (str): Name of a numeric variable
            ax (plt.Axes): A matplotlib Axes object. Optional
            legend_config (dict): Dictionary containing parameters for the legend.
            kwargs (dict): Keyword arguments passed to underlying plotting method.
        """
        ax = ax or Canvas().ax

        self._check_name(a)
        self._check_name(b)
        dtypes = {a: "categorical", b: "numeric"}
        self._check_dtypes(dtypes)

        ax = sns.histplot(
            data=self._data,
            x=b,
            hue=a,
            ax=ax,
            **config,
        )

        title = f"Histogram of {b} by {a}"
        ax.set_title(title)
        plt.tight_layout()

    def kdeplot(
        self,
        a: str,
        b: str,
        ax: plt.Axes = None,
        config: dict = {},
        legend_config: dict = [],
    ) -> None:  # pragma: no cover
        """Renders a boxplot of the categorical, numerical distribution.
        Args:
            a (str): Name of a categorical variable.
            b (str): Name of a numeric variable
            ax (plt.Axes): A matplotlib Axes object. Optional
            legend_config (dict): Dictionary containing parameters for the legend.
            kwargs (dict): Keyword arguments passed to underlying plotting method.
        """
        ax = ax or Canvas().ax

        self._check_name(a)
        self._check_name(b)
        dtypes = {a: "categorical", b: "numeric"}
        self._check_dtypes(dtypes)

        ax = sns.kdeplot(
            data=self._data,
            x=b,
            hue=a,
            ax=ax,
            **config,
        )

        plt.legend(**legend_config)
        title = f"Kernel Density Estimate (KDE) of {b} by {a}"
        ax.set_title(title)
        plt.tight_layout()
