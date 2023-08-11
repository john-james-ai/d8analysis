#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Data Exploration Framework                                                          #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.11                                                                             #
# Filename   : /edation/visual/distribution.py                                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/edation                                            #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday June 18th 2023 01:41:15 am                                                   #
# Modified   : Thursday August 10th 2023 06:08:56 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Plotizations Revealing the Distribution of Data"""
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from edation.visual.base import Plot
from edation.visual.config import Canvas


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

    def __init__(
        self,
        data: pd.DataFrame,
        x: str,
        y: str = None,
        hue: str = None,
        ax: plt.Axes = None,
        title: str = None,
        canvas: Canvas = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(canvas=canvas)
        self._data = data
        self._x = x
        self._y = y
        self._hue = hue
        self._ax = ax
        self._title = title
        self._args = args
        self._kwargs = kwargs

        self._legend_config = None

    def visualize(self) -> None:
        super().visualize()

        args = self._args
        kwargs = self._kwargs

        sns.histplot(
            data=self._data,
            x=self._x,
            y=self._y,
            hue=self._hue,
            ax=self._ax,
            palette=self._canvas.palette,
            *args,
            **kwargs,
        )
        if self._title:
            self._ax.set_title(self._title)

        if self._legend_config is not None:
            self.config_legend()


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

    def __init__(
        self,
        data: pd.DataFrame,
        x: str,
        y: str = None,
        hue: str = None,
        ax: plt.Axes = None,
        title: str = None,
        canvas: Canvas = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(canvas=canvas)
        self._data = data
        self._x = x
        self._y = y
        self._hue = hue
        self._ax = ax
        self._title = title
        self._args = args
        self._kwargs = kwargs

        self._legend_config = None

    def visualize(self) -> None:
        super().visualize()

        args = self._args
        kwargs = self._kwargs

        sns.kdeplot(
            data=self._data,
            x=self._x,
            y=self._y,
            hue=self._hue,
            ax=self._ax,
            palette=self._canvas.palette,
            *args,
            **kwargs,
        )
        if self._title:
            self._ax.set_title(self._title)

        if self._legend_config is not None:
            self.config_legend()


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

    def __init__(
        self,
        data: pd.DataFrame,
        x: str,
        y: str = None,
        hue: str = None,
        ax: plt.Axes = None,
        title: str = None,
        canvas: Canvas = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(canvas=canvas)
        self._data = data
        self._x = x
        self._y = y
        self._hue = hue
        self._ax = ax
        self._title = title
        self._args = args
        self._kwargs = kwargs

        self._legend_config = None

    def visualize(self) -> None:
        super().visualize()

        args = self._args
        kwargs = self._kwargs

        _ = sns.ecdfplot(
            data=self._data,
            x=self._x,
            y=self._y,
            hue=self._hue,
            ax=self._ax,
            palette=self._canvas.palette,
            *args,
            **kwargs,
        )
        if self._title:
            self._ax.set_title(self._title)

        if self._legend_config is not None:
            self.config_legend()


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

    def __init__(
        self,
        data: pd.DataFrame,
        x: str,
        y: str = None,
        hue: str = None,
        ax: plt.Axes = None,
        title: str = None,
        canvas: Canvas = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(canvas=canvas)
        self._data = data
        self._x = x
        self._y = y
        self._hue = hue
        self._ax = ax
        self._title = title
        self._args = args
        self._kwargs = kwargs

        self._legend_config = None

    def visualize(self) -> None:
        super().visualize()

        args = self._args
        kwargs = self._kwargs

        _ = sns.boxplot(
            data=self._data,
            x=self._x,
            y=self._y,
            hue=self._hue,
            ax=self._ax,
            palette=self._canvas.palette,
            *args,
            **kwargs,
        )
        if self._title:
            self._ax.set_title(self._title)

        if self._legend_config is not None:
            self.config_legend()


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

    def __init__(
        self,
        data: pd.DataFrame,
        x: str,
        y: str = None,
        hue: str = None,
        ax: plt.Axes = None,
        title: str = None,
        canvas: Canvas = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(canvas=canvas)
        self._data = data
        self._x = x
        self._y = y
        self._hue = hue
        self._ax = ax
        self._title = title
        self._args = args
        self._kwargs = kwargs

        self._legend_config = None

    def visualize(self) -> None:
        super().visualize()

        args = self._args
        kwargs = self._kwargs

        _ = sns.violinplot(
            data=self._data,
            x=self._x,
            y=self._y,
            hue=self._hue,
            ax=self._ax,
            palette=self._canvas.palette,
            *args,
            **kwargs,
        )
        if self._title:
            self._ax.set_title(self._title)

        if self._legend_config is not None:
            self.config_legend()
