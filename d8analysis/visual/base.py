#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Exploratory Data Analysis Framework                                                 #
# Version    : 0.0.9                                                                               #
# Python     : 3.10.10                                                                             #
# Filename   : /d8analysis/visual/base.py                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/d8analysis                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday May 28th 2023 06:23:03 pm                                                    #
# Modified   : Thursday August 10th 2023 10:27:12 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List

import matplotlib.pyplot as plt
from d8analysis.visual.config import LegendConfig, Canvas


# ------------------------------------------------------------------------------------------------ #
class Figure(ABC):
    """Abstract base class for figure-based visualization classes.

    Subclasses can call the constructor to obtain a default canvas.

    Args:
        canvas (Canvas): Plot configuration object.
    """

    def __init__(self, canvas: Canvas = None) -> None:
        self._canvas = canvas or Canvas()

    @abstractmethod
    def visualize(self) -> None:
        """Renders the visualization"""

        if self._ax is None:
            _, self._ax = plt.subplots(figsize=(self._canvas.width, self._canvas.height))

    def _wrap_ticklabels(
        self, axis: str, axes: List[plt.Axes], fontsize: int = 8
    ) -> List[plt.Axes]:
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
class Plot(Figure):
    """Abstract base class for axis-based plot visualization classes.

    Subclasses can call the constructor to obtain a default canvas and axes object.

    Args:
        canvas (Canvas): Plot configuration object.
    """

    def __init__(self, canvas: Canvas = None) -> None:
        self._canvas = canvas or Canvas()

    @property
    def ax(self) -> plt.Axes:
        return self._ax

    @ax.setter
    def ax(self, ax: plt.Axes) -> None:
        self._ax = ax

    @property
    def legend_config(self) -> LegendConfig:
        return self._legend_config

    @legend_config.setter
    def legend_config(self, legend_config: LegendConfig) -> None:
        self._legend_config = legend_config

    @abstractmethod
    def visualize(self) -> None:
        """Creates an axes if needed and renders the visualization"""
        if self._ax is None:
            _, self._ax = plt.subplots(figsize=(self._canvas.width, self._canvas.height))

    def config_legend(self) -> None:
        """Configures the plot legend"""
        plt.legend(**self._legend_config.as_dict())
