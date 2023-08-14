#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Exploratory Data Analysis Framework                                                 #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.12                                                                             #
# Filename   : /d8analysis/visual/seaborn/plot.py                                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/d8analysis                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday August 13th 2023 08:23:33 am                                                 #
# Modified   : Sunday August 13th 2023 10:34:46 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Plotizations that Reveal Associations between Variables."""
from typing import List, Union

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from dependency_injector.wiring import inject, Provide

from d8analysis.visual.base import Visualizer
from d8analysis.container import D8AnalysisContainer
from d8analysis.visual.seaborn.config import SeabornCanvas

# ------------------------------------------------------------------------------------------------ #
class SeabornVisualizer(Visualizer):
    def __init__(self, canvas: SeabornCanvas = Provide[D8AnalysisContainer.canvas.seaborn]):
        super().__init__(canvas)
        self._canvas = canvas

    def lineplot(self, data: Union[pd.DataFrame, np.ndarray], x: str = None, y: str = None, hue: str = None, title: str = None, ax: plt.Axes = None,  *args, **kwargs) -> None:
        """Draw a line plot with possibility of several semantic groupings.

        The relationship between x and y can be shown for different subsets of the data using the hue, size,
        and style parameters. These parameters control what visual semantics are used to identify the different
        subsets. It is possible to show up to three dimensions independently by using all three semantic
        types, but this style of plot can be hard to interpret and is often ineffective.
        Using redundant semantics (i.e. both hue and style for the same variable) can be helpful
        for making graphics more accessible.

        Args:
            data (Union[pd.DataFrame, np.ndarray]): Input data structure. Either a long-form collection of
                vectors that can be assigned to named variables or a wide-form dataset that will be internally reshaped

            x,y (str): Keys in data.
            hue (str): Grouping variable that will produce lines with different colors. Can be either categorical or numeric,
                although color mapping will behave differently in latter case.
                title (str): Title for the plot. Optional
            ax: (plt.Axes): A matplotlib Axes object. Optional. If not provide, one will be obtained from the canvas.


        """
        if ax is None:
            fig, ax = self._canvas.get_figaxes()

        sns.lineplot(
                data=data,
                x=x,
                y=y,
                hue=hue,
                ax=ax,
                palette=self._canvas.palette,
                *args,
                **kwargs,
            )
        if title is not None:
            ax.set_title(title)

    def scatterplot(self, data: Union[pd.DataFrame, np.ndarray], x: str = None, y: str = None, hue: str = None, title: str = None, ax: plt.Axes = None,  *args, **kwargs) -> None:
        """Draw a scatter plot with possibility of several semantic groupings.

        The relationship between x and y can be shown for different subsets of the data
        using the hue, size, and style parameters. These parameters control what visual
        semantics are used to identify the different subsets. It is possible to show up to
        three dimensions independently by using all three semantic types, but this style of
        plot can be hard to interpret and is often ineffective. Using redundant
        semantics (i.e. both hue and style for the same variable) can be helpful for making graphics
        more accessible.

        Args:
            data (Union[pd.DataFrame, np.ndarray]): Input data structure. Either a long-form collection of
                vectors that can be assigned to named variables or a wide-form dataset that will be internally reshaped

            x,y (str): Keys in data.
            hue (str): Grouping variable that will produce lines with different colors. Can be either categorical or numeric,
                although color mapping will behave differently in latter case.
                title (str): Title for the plot. Optional
            ax: (plt.Axes): A matplotlib Axes object. Optional. If not provide, one will be obtained from the canvas.


        """
        if ax is None:
            fig, ax = self._canvas.get_figaxes()

        sns.scatterplot(
                data=data,
                x=x,
                y=y,
                hue=hue,
                ax=ax,
                palette=self._canvas.palette,
                *args,
                **kwargs,
            )
        if title is not None:
            ax.set_title(title)

    def histogram(self, data: Union[pd.DataFrame, np.ndarray], x: str = None, y: str = None, hue: str = None, stat: str = 'density', element: str = 'bars', fill: bool = True, title: str = None, ax: plt.Axes = None,  *args, **kwargs) -> None:
        """Draw a scatter plot with possibility of several semantic groupings.

        The relationship between x and y can be shown for different subsets of the data
        using the hue, size, and style parameters. These parameters control what visual
        semantics are used to identify the different subsets. It is possible to show up to
        three dimensions independently by using all three semantic types, but this style of
        plot can be hard to interpret and is often ineffective. Using redundant
        semantics (i.e. both hue and style for the same variable) can be helpful for making graphics
        more accessible.

        Args:
            data (Union[pd.DataFrame, np.ndarray]): Input data structure. Either a long-form collection of
                vectors that can be assigned to named variables or a wide-form dataset that will be internally reshaped

            x,y (str): Keys in data.
            hue (str): Grouping variable that will produce lines with different colors. Can be either categorical or numeric,
                although color mapping will behave differently in latter case.
                title (str):  Title for the plot. Optional
            stat (str): Aggregate statistics for each bin. Optional. Default is 'density'.
                See https://seaborn.pydata.org/generated/seaborn.histplot.html for valid values.
            element (str): Visual representation of the histogram statistic. Only relevant with univariate data. Optional. Default is 'bars'.
            fill (bool): If True, fill in the space under the histogram. Only relevant with univariate data.
            ax: (plt.Axes): A matplotlib Axes object. Optional. If not provide, one will be obtained from the canvas.


        """
        if ax is None:
            fig, ax = self._canvas.get_figaxes()

        sns.histplot(
                data=data,
                x=x,
                y=y,
                hue=hue,
                stat=stat,
                element=element,
                fill=fill,
                ax=ax,
                palette=self._canvas.palette,
                *args,
                **kwargs,
            )
        if title is not None:
            ax.set_title(title)

    def boxplot(self, data: Union[pd.DataFrame, np.ndarray], x: str = None, y: str = None, hue: str = None, title: str = None, ax: plt.Axes = None,  *args, **kwargs) -> None:
        """Draw a box plot to show distributions with respect to categories.

        A box plot (or box-and-whisker plot) shows the distribution of
        quantitative data in a way that facilitates comparisons between
        variables or across levels of a categorical variable. The box
        shows the quartiles of the dataset while the whiskers extend
        to show the rest of the distribution, except for points that
        are determined to be “outliers” using a method that is a
        function of the inter-quartile range.

        Args:
            data (Union[pd.DataFrame, np.ndarray]): Input data structure.
                Either a long-form collection of vectors that can be assigned to
                named variables or a wide-form dataset that will be internally
                reshaped

            x,y (str): Keys in data.
            hue (str): Grouping variable that will produce lines with different colors. Can be either categorical or numeric,
                although color mapping will behave differently in latter case.
            title (str): Title for the plot. Optional
            ax: (plt.Axes): A matplotlib Axes object. Optional. If not provide, one will be obtained from the canvas.


        """
        if ax is None:
            fig, ax = self._canvas.get_figaxes()

        sns.boxplot(
                data=data,
                x=x,
                y=y,
                hue=hue,
                ax=ax,
                palette=self._canvas.palette,
                *args,
                **kwargs,
            )
        if title is not None:
            ax.set_title(title)


    def kdeplot(self, data: Union[pd.DataFrame, np.ndarray], x: str = None, y: str = None, hue: str = None, title: str = None, ax: plt.Axes = None,  *args, **kwargs) -> None:
        """Plot univariate or bivariate distributions using kernel density estimation.

        A kernel density estimate (KDE) plot is a method for visualizing the distribution of observations in a dataset, analogous to a histogram. KDE represents the data using a continuous probability density curve in one or more dimensions.

        Args:
            data (Union[pd.DataFrame, np.ndarray]): Input data structure.
                Either a long-form collection of vectors that can be assigned to
                named variables or a wide-form dataset that will be internally
                reshaped

            x,y (str): Keys in data.
            hue (str): Grouping variable that will produce lines with different colors. Can be either categorical or numeric,
                although color mapping will behave differently in latter case.
            title (str): Title for the plot. Optional
            ax: (plt.Axes): A matplotlib Axes object. Optional. If not provide, one will be obtained from the canvas.


        """
        if ax is None:
            fig, ax = self._canvas.get_figaxes()

        sns.kdeplot(
                data=data,
                x=x,
                y=y,
                hue=hue,
                ax=ax,
                palette=self._canvas.palette,
                *args,
                **kwargs,
            )
        if title is not None:
            ax.set_title(title)


    def kdeplot(self, data: Union[pd.DataFrame, np.ndarray], x: str = None, y: str = None, hue: str = None, title: str = None, ax: plt.Axes = None,  *args, **kwargs) -> None:
        """Plot empirical cumulative distribution functions.

        An ECDF represents the proportion or count of observations falling below each unique value
        in a dataset. Compared to a histogram or density plot, it has the advantage that each
        observation is visualized directly, meaning that there are no binning or smoothing
        parameters that need to be adjusted. It also aids direct comparisons between multiple
        distributions. A downside is that the relationship between the appearance of the plot and
        the basic properties of the distribution (such as its central tendency, variance, and the
        presence of any bimodality) may not be as intuitive.

        """
        if ax is None:
            fig, ax = self._canvas.get_figaxes()

        sns.ecdfplot(
                data=data,
                x=x,
                y=y,
                hue=hue,
                ax=ax,
                palette=self._canvas.palette,
                *args,
                **kwargs,
            )
        if title is not None:
            ax.set_title(title)

    def _wrap_ticklabels(self, axis: str, axes: List[plt.Axes], fontsize: int = 8) -> List[plt.Axes]:
        """Wraps long tick labels"""
        if axis.lower() == "x":
            for i, ax in enumerate(axes):
                xlabels = [label.get_text() for label in ax.get_xticklabels()]
                xlabels = [label.replace(" ", "\n") for label in xlabels]
                ax.set_xticklabels(xlabels, fontdict={"fontsize": fontsize})
                ax.tick_params(axis="x", labelsize=fontsize)

        if axis.lower() == "y":
            for i, ax in enumerate(axes):
                ylabels = [label.get_text() for label in ax.get_yticklabels()]
                ylabels = [label.replace(" ", "\n") for label in ylabels]
                ax.set_yticklabels(ylabels, fontdict={"fontsize": fontsize})
                ax.tick_params(axis="y", labelsize=fontsize)

        return axes


# ------------------------------------------------------------------------------------------------ #
#                                        SCATTERPLOT                                               #
# ------------------------------------------------------------------------------------------------ #


class Visualizer(SeabornVisual):  # pragma: no cover
    """Wrapper for Seaborn visualizations.


    Args:
        canvas (SeabornCanvas): A dataclass containing the configuration of the canvas
            for the visualization. Optional. Default is set in the base class.
    """
    @inject
    def __init__(
        self,
        canvas: Canvas = Provide[D8AnalysisContainer.canvas.seaborn]
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
