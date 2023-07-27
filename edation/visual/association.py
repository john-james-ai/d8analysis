#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Enter Project Name in Workspace Settings                                            #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.11                                                                             #
# Filename   : /edation/visual/association.py                                                      #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : Enter URL in Workspace Settings                                                     #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday June 20th 2023 07:57:56 pm                                                  #
# Modified   : Thursday July 27th 2023 03:44:42 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Visualizations that Reveal Associations between Variables."""
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from .base import Visual
from .config import Canvas

# ------------------------------------------------------------------------------------------------ #
#                                        SCATTERPLOT                                               #
# ------------------------------------------------------------------------------------------------ #


class ScatterPlot(Visual):  # pragma: no cover
    """Draw a scatter plot with possibility of several semantic groupings."""

    def __call__(
        self,
        data: pd.DataFrame,
        x: str,
        y: str = None,
        hue: str = None,
        ax: plt.Axes = None,
        title: str = None,
        style: str = "whitegrid",
        palette: str = "colorblind",
        *args,
        **kwargs,
    ) -> plt.Axes:
        """Plots the histogram

        Args:
            data (pd.DataFrame): Input data
            x (str): The variables that specify positions in the x axis
            y (str): The variables that specify positions in the y axis
            hue (str): Variable that determines the colors of plot elements.
            ax (plt.Axes): A matplotlib Axes object. Optional.
            style (str): A string representing a seaborn style. Optional. Defaults to 'whitegrid'
            palette (str): A string indicating one of the supported palettes. See docstring for
                Palette class in config module. Defaults to 'winter_blue'.
            args and kwargs passed to the underlying seaborn histplot method.
                See https://seaborn.pydata.org/generated/seaborn.histplot.html for a
                complete list of parameters.
        """

        ax = ax or Canvas().ax

        ax = sns.scatterplot(
            data=data,
            x=x,
            y=y,
            hue=hue,
            ax=ax,
            *args,
            **kwargs,
        )
        if title:
            ax.set_title(title)


# ------------------------------------------------------------------------------------------------ #
#                                        LINE PLOT                                                 #
# ------------------------------------------------------------------------------------------------ #


class LinePlot(Visual):  # pragma: no cover
    """Draw a scatter plot with possibility of several semantic groupings."""

    def __call__(
        self,
        data: pd.DataFrame,
        x: str,
        y: str = None,
        hue: str = None,
        ax: plt.Axes = None,
        title: str = None,
        style: str = "whitegrid",
        palette: str = "colorblind",
        *args,
        **kwargs,
    ) -> plt.Axes:
        """Plots the histogram

        Args:
            data (pd.DataFrame): Input data
            x (str): The variables that specify positions in the x axis
            y (str): The variables that specify positions in the y axis
            hue (str): Variable that determines the colors of plot elements.
            ax (plt.Axes): A matplotlib Axes object. Optional.
            style (str): A string representing a seaborn style. Optional. Defaults to 'whitegrid'
            palette (str): A string indicating one of the supported palettes. See docstring for
                Palette class in config module. Defaults to 'winter_blue'.
            args and kwargs passed to the underlying seaborn histplot method.
                See https://seaborn.pydata.org/generated/seaborn.histplot.html for a
                complete list of parameters.
        """

        ax = ax or Canvas().ax

        ax = sns.lineplot(
            data=data,
            x=x,
            y=y,
            hue=hue,
            ax=ax,
            *args,
            **kwargs,
        )
        if title is not None:
            ax.set_title(title)


# ------------------------------------------------------------------------------------------------ #
#                                        PAIR PLOT                                                 #
# ------------------------------------------------------------------------------------------------ #


class PairPlot(Visual):  # pragma: no cover
    """Plot pairwise relationships in a dataset. This is a figure level plot showing a grid of axes."""

    def __call__(
        self,
        data: pd.DataFrame,
        hue: str = None,
        vars: list = None,
        x_vars: list = None,
        y_vars: list = None,
        title: str = None,
        style: str = "whitegrid",
        palette: str = "colorblind",
        *args,
        **kwargs,
    ) -> plt.Axes:
        """Plots the histogram

        Args:
            data (pd.DataFrame): Input data
            hue (str): Variable that determines the colors of plot elements.
            vars (list): List of variable names from data to use. If None, all variables will be used.
            x_vars (list): A list of variable names to use for each row of the grid.
            y_vars (list): A list of variable names to use for each column of the grid.
            style (str): A string representing a seaborn style. Optional. Defaults to 'whitegrid'
            palette (str): A string indicating one of the supported palettes. See docstring for
                Palette class in config module. Defaults to 'winter_blue'.
            args and kwargs passed to the underlying seaborn histplot method.
                See https://seaborn.pydata.org/generated/seaborn.histplot.html for a
                complete list of parameters.
        """

        g = sns.pairplot(
            data=data,
            hue=hue,
            vars=vars,
            x_vars=x_vars,
            y_vars=y_vars,
            *args,
            **kwargs,
        )
        if title is not None:
            g.fig.suptitle(title)
        g.tight_layout()


# ------------------------------------------------------------------------------------------------ #
#                                       JOINT PLOT                                                 #
# ------------------------------------------------------------------------------------------------ #


class JointPlot(Visual):  # pragma: no cover
    """Draw a plot of two variables with bivariate and univariate graphs."""

    def __call__(
        self,
        data: pd.DataFrame,
        x: str,
        y: str = None,
        hue: str = None,
        ax: plt.Axes = None,
        title: str = None,
        style: str = "whitegrid",
        palette: str = "colorblind",
        *args,
        **kwargs,
    ) -> plt.Axes:
        """Plots the histogram

        Args:
            data (pd.DataFrame): Input data
            x (str): The variables that specify positions in the x axis
            y (str): The variables that specify positions in the y axis
            hue (str): Variable that determines the colors of plot elements.
            ax (plt.Axes): A matplotlib Axes object. Optional.
            style (str): A string representing a seaborn style. Optional. Defaults to 'whitegrid'
            palette (str): A string indicating one of the supported palettes. See docstring for
                Palette class in config module. Defaults to 'winter_blue'.
            args and kwargs passed to the underlying seaborn histplot method.
                See https://seaborn.pydata.org/generated/seaborn.histplot.html for a
                complete list of parameters.
        """

        g = sns.jointplot(
            data=data,
            x=x,
            y=y,
            hue=hue,
            ax=ax,
            *args,
            **kwargs,
        )
        if title is not None:
            g.fig.suptitle(title)
            g.fig.tight_layout()
