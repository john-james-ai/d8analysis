#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Exploratory Data Analysis Framework                                                 #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.11                                                                             #
# Filename   : /d8analysis/visual/distribution.py                                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/d8analysis                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday June 18th 2023 01:41:15 am                                                   #
# Modified   : Saturday August 12th 2023 08:33:15 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Plotizations Revealing the Distribution of Data"""
import pandas as pd

import seaborn as sns
from dependency_injector.wiring import inject, Provide

from d8analysis.visual.base import Plot
from d8analysis.visual.config import Canvas
from d8analysis.container import d8analysisContainer
from d8analysis.data.generation import Distribution


# ------------------------------------------------------------------------------------------------ #
#                                     HISTOGRAM                                                    #
# ------------------------------------------------------------------------------------------------ #
class Histogram(Plot):  # pragma: no cover
    """Plot univariate or bivariate histograms to show distributions of datasets.


    Args:
        data (pd.DataFrame): Input data
        x (str): The variables that specify positions in the x axis
        y (str): The variables that specify positions in the y axis
        hue (str): Variable that determines the colors of plot elements.
        ax (plt.Axes): A matplotlib Axes object. Optional. If not none, the ax parameter
            overrides the default set in the base class.
        title (str): The visualization title. Optional
        canvas (Canvas): A dataclass containing the configuration of the canvas
            for the visualization. Optional. Default is set in the base class.
        args and kwargs passed to the underlying seaborn histplot method.
            See https://seaborn.pydata.org/generated/seaborn.histplot.html for a
            complete list of parameters.
    """

    @inject
    def __init__(
        self,
        data: pd.DataFrame,
        x: str,
        y: str = None,
        hue: str = None,
        title: str = None,
        canvas: type[Canvas] = Provide[d8analysisContainer.canvas],
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self._data = data
        self._x = x
        self._y = y
        self._hue = hue
        self._title = title
        self._args = args
        self._kwargs = kwargs

    def plot(self) -> None:
        ax = self._canvas.get_figaxes().axes

        args = self._args
        kwargs = self._kwargs

        sns.histplot(
            data=self._data,
            x=self._x,
            y=self._y,
            hue=self._hue,
            stat="density",
            element="bars",
            fill=True,
            ax=ax,
            color=self._canvas.colors.dark_blue,
            *args,
            **kwargs,
        )
        if self._title:
            ax.set_title(self._title)


# ------------------------------------------------------------------------------------------------ #
#                            PROBABILITY DENSITY FUNCTION                                          #
# ------------------------------------------------------------------------------------------------ #
class PDF(Plot):  # pragma: no cover
    """Plots a probability density function (pdf)


    Args:
        data (pd.DataFrame): Input data
        x (str): The variables that specify positions in the x axis
        y (str): The variables that specify positions in the y axis
        hue (str): Variable that determines the colors of plot elements.
        ax (plt.Axes): A matplotlib Axes object. Optional. If not none, the ax parameter
            overrides the default set in the base class.
        title (str): The visualization title. Optional
        canvas (Canvas): A dataclass containing the configuration of the canvas
            for the visualization. Optional.
        args and kwargs passed to the underlying seaborn histplot method.
            See https://seaborn.pydata.org/generated/seaborn.histplot.html for a
            complete list of parameters.
    """

    @inject
    def __init__(
        self,
        data: pd.DataFrame,
        x: str,
        y: str = None,
        hue: str = None,
        title: str = None,
        canvas: type[Canvas] = Provide[d8analysisContainer.canvas],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(canvas=canvas)
        self._data = data
        self._x = x
        self._y = y
        self._hue = hue
        self._title = title
        self._args = args
        self._kwargs = kwargs

    def plot(self) -> None:
        ax = self._canvas.get_figaxes().axes

        args = self._args
        kwargs = self._kwargs

        sns.kdeplot(
            data=self._data,
            x=self._x,
            y=self._y,
            hue=self._hue,
            ax=ax,
            palette=self._canvas.palette,
            *args,
            **kwargs,
        )
        if self._title:
            ax.set_title(self._title)


# ------------------------------------------------------------------------------------------------ #
#                            CUMULATIVE DISTRIBUTION FUNCTION                                      #
# ------------------------------------------------------------------------------------------------ #
class CDF(Plot):  # pragma: no cover
    """Plots a cumulative distribution function (cdf)


    Args:
        data (pd.DataFrame): Input data
        x (str): The variables that specify positions in the x axis
        y (str): The variables that specify positions in the y axis
        hue (str): Variable that determines the colors of plot elements.
        ax (plt.Axes): A matplotlib Axes object. Optional. If not none, the ax parameter
            overrides the default set in the base class.
        title (str): The visualization title. Optional
        canvas (Canvas): A dataclass containing the configuration of the canvas
            for the visualization. Optional.
        args and kwargs passed to the underlying seaborn histplot method.
            See https://seaborn.pydata.org/generated/seaborn.histplot.html for a
            complete list of parameters.
    """

    @inject
    def __init__(
        self,
        data: pd.DataFrame,
        x: str,
        y: str = None,
        hue: str = None,
        title: str = None,
        canvas: type[Canvas] = Provide[d8analysisContainer.canvas],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(canvas=canvas)
        self._data = data
        self._x = x
        self._y = y
        self._hue = hue
        self._title = title
        self._args = args
        self._kwargs = kwargs

    def plot(self) -> None:
        ax = self._canvas.get_figaxes().axes

        args = self._args
        kwargs = self._kwargs

        _ = sns.ecdfplot(
            data=self._data,
            x=self._x,
            y=self._y,
            hue=self._hue,
            ax=ax,
            palette=self._canvas.palette,
            *args,
            **kwargs,
        )
        if self._title:
            ax.set_title(self._title)


# ------------------------------------------------------------------------------------------------ #
#                                        BOXPLOT                                                   #
# ------------------------------------------------------------------------------------------------ #
class BoxPlot(Plot):  # pragma: no cover
    """Draw a box plot to show distributions with or without respect to categories.


    Args:
        data (pd.DataFrame): Input data
        x (str): The variables that specify positions in the x axis
        y (str): The variables that specify positions in the y axis
        hue (str): Variable that determines the colors of plot elements.
        ax (plt.Axes): A matplotlib Axes object. Optional. If not none, the ax parameter
            overrides the default set in the base class.
        title (str): The visualization title. Optional
        canvas (Canvas): A dataclass containing the configuration of the canvas
            for the visualization. Optional.
        args and kwargs passed to the underlying seaborn histplot method.
            See https://seaborn.pydata.org/generated/seaborn.histplot.html for a
            complete list of parameters.
    """

    @inject
    def __init__(
        self,
        data: pd.DataFrame,
        x: str,
        y: str = None,
        hue: str = None,
        title: str = None,
        canvas: type[Canvas] = Provide[d8analysisContainer.canvas],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(canvas=canvas)
        self._data = data
        self._x = x
        self._y = y
        self._hue = hue
        self._title = title
        self._args = args
        self._kwargs = kwargs

    def plot(self) -> None:
        ax = self._canvas.get_figaxes().axes

        args = self._args
        kwargs = self._kwargs

        _ = sns.boxplot(
            data=self._data,
            x=self._x,
            y=self._y,
            hue=self._hue,
            ax=ax,
            palette=self._canvas.palette,
            *args,
            **kwargs,
        )
        if self._title:
            ax.set_title(self._title)


# ------------------------------------------------------------------------------------------------ #
#                                      VIOLIN PLOT                                                 #
# ------------------------------------------------------------------------------------------------ #
class ViolinPlot(Plot):  # pragma: no cover
    """Draw a violin plot, as a combination of boxplot and kernel density estimate.


    Args:
        data (pd.DataFrame): Input data
        x (str): The variables that specify positions in the x axis
        y (str): The variables that specify positions in the y axis
        hue (str): Variable that determines the colors of plot elements.
        ax (plt.Axes): A matplotlib Axes object. Optional. If not none, the ax parameter
            overrides the default set in the base class.
        title (str): The visualization title. Optional
        canvas (Canvas): A dataclass containing the configuration of the canvas
            for the visualization. Optional.
        args and kwargs passed to the underlying seaborn histplot method.
            See https://seaborn.pydata.org/generated/seaborn.histplot.html for a
            complete list of parameters.
    """

    @inject
    def __init__(
        self,
        data: pd.DataFrame,
        x: str,
        y: str = None,
        hue: str = None,
        title: str = None,
        canvas: type[Canvas] = Provide[d8analysisContainer.canvas],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(canvas=canvas)
        self._data = data
        self._x = x
        self._y = y
        self._hue = hue
        self._title = title
        self._args = args
        self._kwargs = kwargs

    def plot(self) -> None:
        ax = self._canvas.get_figaxes().axes

        args = self._args
        kwargs = self._kwargs

        _ = sns.violinplot(
            data=self._data,
            x=self._x,
            y=self._y,
            hue=self._hue,
            ax=ax,
            palette=self._canvas.palette,
            *args,
            **kwargs,
        )
        if self._title:
            ax.set_title(self._title)


# ------------------------------------------------------------------------------------------------ #
#                                      HISTOGRAM PDF PLOT                                          #
# ------------------------------------------------------------------------------------------------ #
class HistPDFPlot(Plot):  # pragma: no cover
    """Plots a univariate histogram and a probability density function.


    Args:
        data (pd.DataFrame): Input data
        x (str): The variables that specify positions in the x axis
        y (str): The variables that specify positions in the y axis
        hue (str): Variable that determines the colors of plot elements.
        ax (plt.Axes): A matplotlib Axes object. Optional. If not none, the ax parameter
            overrides the default set in the base class.
        title (str): The visualization title. Optional
        canvas (Canvas): A dataclass containing the configuration of the canvas
            for the visualization. Optional. Default is set in the base class.
        args and kwargs passed to the underlying seaborn histplot method.
            See https://seaborn.pydata.org/generated/seaborn.histplot.html for a
            complete list of parameters.
    """

    @inject
    def __init__(
        self,
        data: pd.DataFrame,
        x: str = None,
        y: str = None,
        hue: str = None,
        title: str = None,
        canvas: type[Canvas] = Provide[d8analysisContainer.canvas],
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self._data = data
        self._pdf = pdf
        self._hue = hue
        self._title = title
        self._canvas = canvas
        self._args = args
        self._kwargs = kwargs

    def plot(self) -> None:
        ax1 = self._canvas.get_figaxes().axes
        fig = self._canvas.fig

        ax1 = sns.histplot(
            x=self._data,
            stat="count",
            element="bars",
            fill=True,
            color=self._canvas.colors.dark_blue,
            ax=ax1,
            label="Empirical Distribution",
            legend=True,
        )
        ax2 = ax1.twinx()
        ax2 = sns.lineplot(
            x=self._pdf.x,
            y=self._pdf.y,
            ax=ax2,
            color=self._canvas.colors.orange,
            label="Probability Distribution Function",
        )
        title = f"{self._pdf.name}\nEmpirical Distribution vis-a-vis Probability Density Function\n{self._pdf.params}"
        ax2.get_legend().remove()
        fig.legend()
        fig.suptitle(title, fontsize=self._canvas.fontsize_title)
        fig.tight_layout()
        return fig


# ------------------------------------------------------------------------------------------------ #
#              PROBABILITY DENSITY FUNCTION / CUMULATIVE DISTRIBUTION FUNCTION PLOT                #
# ------------------------------------------------------------------------------------------------ #
class PdfCdf(Plot):  # pragma: no cover
    """Plots a probability density function and cumulative distribution function.

    Args:
        data (pd.DataFrame): Input data
        x (str): The variables that specify positions in the x axis
        y (str): The variables that specify positions in the y axis
        hue (str): Variable that determines the colors of plot elements.
        ax (plt.Axes): A matplotlib Axes object. Optional. If not none, the ax parameter
            overrides the default set in the base class.
        title (str): The visualization title. Optional
        canvas (Canvas): A dataclass containing the configuration of the canvas
            for the visualization. Optional. Default is set in the base class.
        args and kwargs passed to the underlying seaborn histplot method.
            See https://seaborn.pydata.org/generated/seaborn.histplot.html for a
            complete list of parameters.
    """

    @inject
    def __init__(
        self,
        data: pd.DataFrame,
        x: str = None,
        y: str = None,
        hue: str = None,
        title: str = None,
        canvas: type[Canvas] = Provide[d8analysisContainer.canvas],
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self._pdf = pdf
        self._cdf = cdf
        self._hue = hue
        self._title = title
        self._canvas = canvas
        self._args = args
        self._kwargs = kwargs

    def plot(self) -> None:
        ax1 = self._canvas.get_figaxes().axes
        fig = self._canvas.fig

        ax1 = sns.lineplot(
            x=self._cdf.x,
            y=self._cdf.y,
            ax=ax1,
            color=self._canvas.colors.dark_blue,
            label="Cumulative Distribution Function",
        )
        ax2 = ax1.twinx()
        ax2 = sns.lineplot(
            x=self._pdf.x,
            y=self._pdf.y,
            ax=ax2,
            color=self._canvas.colors.orange,
            label="Probability Distribution Function",
        )
        title = f"{self._cdf.name}\n{self._cdf.label}-{self._pdf.label}\n{self._pdf.params}"
        ax1.get_legend().remove()
        ax2.get_legend().remove()
        fig.legend()
        fig.suptitle(title, fontsize=self._canvas.fontsize_title)
        fig.tight_layout()
        return fig
