#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Enter Project Name in Workspace Settings                                            #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.10                                                                             #
# Filename   : /edation/visual/base.py                                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : Enter URL in Workspace Settings                                                     #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday May 28th 2023 06:23:03 pm                                                    #
# Modified   : Thursday July 27th 2023 02:21:38 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
import logging

# import seaborn as sns
import matplotlib.pyplot as plt

from .config import Canvas, Styling


# ------------------------------------------------------------------------------------------------ #
class Visual(ABC):
    """Abstract base class for plot classes."""

    __default_palette = "blues_r"

    def __init__(self) -> None:
        # self.get_set_styling()
        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    @abstractmethod
    def __call__(self, *args, **kwargs) -> plt.Axes:
        """Presents the visualization."""

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

    def get_canvas(
        self,
        nrows: int = 1,
        ncols: int = 1,
        figsize: tuple = (12, 4),
    ) -> Canvas:
        """Sets the palette and returns a canvas containing figures, axes

        Args:
            nrows (int): Optional number or rows in the canvas. Defaults to 1.
            ncols (int): Optional number or columns in the canvas. Defaults to 1.
            figsize (tuple): Figure size. Defaults to (12,4)
            style (str): The seaborn style. Defaults to 'whitegrid'
            saturation (float): The level of color saturation. Defaults to 1.
            fontsize (int): The size of fonts for legends, axis labels, and text.
            title_fontsize (int): The size of fonts for title.
        """

        return Canvas(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
        )

    def get_set_styling(self, style: str = "whitegrid", palette: str = "blues_r") -> Styling:
        """Sets the palette and returns a canvas containing figures, axes

        Args:
            style (str): The seaborn style. Defaults to 'whitegrid'
            saturation (float): The level of color saturation. Defaults to 1.
            fontsize (int): The size of fonts for legends, axis labels, and text.
            title_fontsize (int): The size of fonts for title.
        """
        return Styling(style=style, palette=palette)

    def get_or_create_ax(self, ax: plt.Axes = None) -> plt.Axes:
        """Returns a valid matplotlib axes object

        Args:
            ax (plt.Axes): Optional matplotlib axes object.
        """
        ax = ax or Canvas().ax
        return ax
