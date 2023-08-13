#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Exploratory Data Analysis Framework                                                 #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.10                                                                             #
# Filename   : /d8analysis/visual/base.py                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/d8analysis                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday May 28th 2023 06:23:03 pm                                                    #
# Modified   : Saturday August 12th 2023 06:13:36 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List

import matplotlib.pyplot as plt
from d8analysis.visual.config import Canvas


# ------------------------------------------------------------------------------------------------ #
class Plot(ABC):
    """Abstract base class for axis-based plot visualization classes.

    Subclasses can call the constructor to obtain a default canvas and axes object.

    Args:
        canvas (Canvas): Plot configuration object.
    """

    def __init__(self, canvas: Canvas, *args, **kwargs) -> None:
        self._canvas = canvas

    @property
    def axes(self) -> plt.Axes:
        return self._axes

    @axes.setter
    def axes(self, axes: plt.Axes) -> None:
        self._axes = axes

    @property
    def fig(self) -> plt.Figure:
        return self._fig

    @fig.setter
    def fig(self, fig: plt.Figure) -> None:
        self._fig = fig

    @abstractmethod
    def plot(self) -> None:
        """Creates an axes if needed and renders the visualization"""

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
