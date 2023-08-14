#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Exploratory Data Analysis Framework                                                 #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.11                                                                             #
# Filename   : /d8analysis/visual/association.py                                                   #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/d8analysis                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday June 20th 2023 07:57:56 pm                                                  #
# Modified   : Sunday August 13th 2023 09:13:08 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Plotizations that Reveal Associations between Variables."""
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from d8analysis.visual.base import Plot, Figure
from d8analysis.visual.config import SeabornCanvas

# ------------------------------------------------------------------------------------------------ #
#                                        SCATTERPLOT                                               #
# ------------------------------------------------------------------------------------------------ #


class ScatterPlot(Plot):  # pragma: no cover
    """Draw a scatter plot with possibility of several semantic groupings.


    Args:
        data (pd.DataFrame): Input data
        x (str): The variables that specify positions in the x axis
        y (str): The variables that specify positions in the y axis
        hue (str): Variable that determines the colors of plot elements.
        ax (plt.Axes): A matplotlib Axes object. Optional. If not none, the ax parameter
            overrides the default set in the base class.
        title (str): Title for the visualization.
        canvas (SeabornCanvas): A dataclass containing the configuration of the canvas
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
        canvas: type[SeabornCanvas] = SeabornCanvas,
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

    def plot(self) -> None:
        super().visualize()

        args = self._args
        kwargs = self._kwargs

        sns.scatterplot(
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
#                                        LINE PLOT                                                 #
# ------------------------------------------------------------------------------------------------ #


class LinePlot(Plot):  # pragma: no cover
    """Renders a lineplot

    Args:
        data (pd.DataFrame): Input data
        x (str): The variables that specify positions in the x axis
        y (str): The variables that specify positions in the y axis
        hue (str): Variable that determines the colors of plot elements.
        ax (plt.Axes): A matplotlib Axes object. Optional. If not none, the ax parameter
            overrides the default set in the base class.
        title (str): Title for the visualization.
        canvas (SeabornCanvas): A dataclass containing the configuration of the canvas
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
        canvas: type[SeabornCanvas] = SeabornCanvas,
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

    def plot(self) -> None:
        super().visualize()

        args = self._args
        kwargs = self._kwargs

        sns.lineplot(
            data=self._data,
            x=self._x,
            y=self._y,
            hue=self._hue,
            ax=self._ax,
            palette=self._canvas.palette,
            *args,
            **kwargs,
        )
        if self._title is not None:
            self._ax.set_title(self._title)

        if self._legend_config is not None:
            self.config_legend()


# ------------------------------------------------------------------------------------------------ #
#                                        PAIR PLOT                                                 #
# ------------------------------------------------------------------------------------------------ #


class PairPlot(Figure):  # pragma: no cover
    """Plot pairwise relationships in a dataset. This is a figure level plot showing a grid of axes

    Args:
        data (pd.DataFrame): Input data
        vars (list): List of variable names from data to use. If None, all variables will be used.
        hue (str): Variable that determines the colors of plot elements.
        title (str): Title for the visualization.
        canvas (SeabornCanvas): A dataclass containing the configuration of the canvas
            for the visualization. Optional. Default is set in the base class.
        args and kwargs passed to the underlying seaborn histplot method.
            See https://seaborn.pydata.org/generated/seaborn.histplot.html for a
            complete list of parameters.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        vars: list = None,
        hue: str = None,
        title: str = None,
        canvas: type[SeabornCanvas] = SeabornCanvas,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(canvas=canvas)
        self._data = data
        self._vars = vars
        self._hue = hue
        self._title = title
        self._args = args
        self._kwargs = kwargs

        self._legend_config = None

    def plot(self) -> None:
        args = self._args
        kwargs = self._kwargs
        g = sns.pairplot(
            data=self._data,
            hue=self._hue,
            vars=self._vars,
            palette=self._canvas.palette,
            *args,
            **kwargs,
        )
        if self._title is not None:
            g.fig.suptitle(self._title)
        g.tight_layout()

        if self._legend_config is not None:
            self.config_legend()


# ------------------------------------------------------------------------------------------------ #
#                                       JOINT PLOT                                                 #
# ------------------------------------------------------------------------------------------------ #


class JointPlot(Figure):  # pragma: no cover
    """Draw a plot of two variables with bivariate and univariate graphs.

    Args:
        data (pd.DataFrame): Input data
        x (str): The variables that specify positions in the x axis
        y (str): The variables that specify positions in the y axis
        hue (str): Variable that determines the colors of plot elements.
        title (str): Title for the visualization.
        canvas (SeabornCanvas): A dataclass containing the configuration of the canvas
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
        title: str = None,
        canvas: type[SeabornCanvas] = SeabornCanvas,
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

        self._legend_config = None

    def plot(self) -> None:
        args = self._args
        kwargs = self._kwargs

        g = sns.jointplot(
            data=self._data,
            x=self._x,
            y=self._y,
            hue=self._hue,
            palette=self._canvas.palette,
            *args,
            **kwargs,
        )
        if self._title is not None:
            g.fig.suptitle(self._title)
            g.fig.tight_layout()

        if self._legend_config is not None:
            self.config_legend()
