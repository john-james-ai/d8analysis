#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Enter Project Name in Workspace Settings                                            #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.11                                                                             #
# Filename   : /edation/visual/distribution.py                                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : Enter URL in Workspace Settings                                                     #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday June 18th 2023 01:41:15 am                                                   #
# Modified   : Thursday July 27th 2023 08:50:52 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Visualizations Revealing the Distribution of Data"""
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from .base import Visual

# ------------------------------------------------------------------------------------------------ #
#                                       HISTOGRAM                                                  #
# ------------------------------------------------------------------------------------------------ #


class Histogram(Visual):  # pragma: no cover
    """Plot univariate or bivariate histograms to show distributions of datasets."""

    def __init__(self) -> None:
        super().__init__()

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
            kde (bool): If True, compute a kernel density estimate to smooth the distribution.
            ax (plt.Axes): A matplotlib Axes object. Optional.
            style (str): A string representing a seaborn style. Optional. Defaults to 'whitegrid'
            palette (str): A string indicating one of the supported palettes. See docstring for
                Palette class in config module. Defaults to 'winter_blue'.
            args and kwargs passed to the underlying seaborn histplot method.
                See https://seaborn.pydata.org/generated/seaborn.histplot.html for a
                complete list of parameters.
        """

        ax = self.get_or_create_ax(ax)
        style = self.get_set_styling(style=style, palette=palette)

        ax = sns.histplot(
            data=data,
            x=x,
            y=y,
            hue=hue,
            ax=ax,
            legend=True,
            pmax=style.saturation,
            *args,
            **kwargs,
        )
        if title:
            ax.set_title(title)

        for container in ax.containers:
            ax.bar_label(container)


# ------------------------------------------------------------------------------------------------ #
#                                       KDE PLOT                                                   #
# ------------------------------------------------------------------------------------------------ #


class KDEPlot(Visual):  # pragma: no cover
    """Plot univariate or bivariate histograms to show distributions of datasets."""

    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self,
        data: pd.DataFrame,
        x: str,
        y: str = None,
        hue: str = None,
        ax: plt.Axes = None,
        title: str = None,
        legend: bool = True,
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
            title (str): The title for the plot
            legend (bool): Whether to show render a legend on the plot.
            style (str): A string representing a seaborn style. Optional. Defaults to 'whitegrid'
            palette (str): A string indicating one of the supported palettes. See docstring for
                Palette class in config module. Defaults to 'winter_blue'.
            args and kwargs passed to the underlying seaborn histplot method.
                See https://seaborn.pydata.org/generated/seaborn.histplot.html for a
                complete list of parameters.
        """

        ax = self.get_or_create_ax(ax)
        style = self.get_set_styling(style=style, palette=palette)

        ax = sns.kdeplot(
            data=data,
            x=x,
            y=y,
            hue=hue,
            legend=legend,
            ax=ax,
            *args,
            **kwargs,
        )
        if title:
            ax.set_title(title)


# ------------------------------------------------------------------------------------------------ #
#                                        BOX PLOT                                                  #
# ------------------------------------------------------------------------------------------------ #


class BoxPlot(Visual):  # pragma: no cover
    """Draw a box plot to show distributions with respect to categories or groups."""

    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self,
        data: pd.DataFrame,
        x: str,
        y: str = None,
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
            ax (plt.Axes): A matplotlib Axes object. Optional.
            style (str): A string representing a seaborn style. Optional. Defaults to 'whitegrid'
            palette (str): A string indicating one of the supported palettes. See docstring for
                Palette class in config module. Defaults to 'winter_blue'.
            args and kwargs passed to the underlying seaborn histplot method.
                See https://seaborn.pydata.org/generated/seaborn.histplot.html for a
                complete list of parameters.
        """

        ax = self.get_or_create_ax(ax)
        style = self.get_set_styling(style=style, palette=palette)

        ax = sns.boxplot(
            data=data,
            x=x,
            y=y,
            ax=ax,
            *args,
            **kwargs,
        )
        if title:
            ax.set_title(title)


# ------------------------------------------------------------------------------------------------ #
#                                     VIOLIN PLOT                                                  #
# ------------------------------------------------------------------------------------------------ #


class ViolinPlot(Visual):  # pragma: no cover
    """Draw a combination of boxplot and kernel density estimate."""

    def __init__(self) -> None:
        super().__init__()

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

        ax = self.get_or_create_ax(ax)
        style = self.get_set_styling(style=style, palette=palette)

        ax = sns.violinplot(
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
#                           EMPIRICAL CUMULATIVE DISTRIBUTION PLOT                                 #
# ------------------------------------------------------------------------------------------------ #


class ECDFPlot(Visual):  # pragma: no cover
    """Draw a combination of boxplot and kernel density estimate."""

    def __init__(self) -> None:
        super().__init__()

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

        ax = self.get_or_create_ax(ax)
        style = self.get_set_styling(style=style, palette=palette)

        ax = sns.ecdfplot(
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
