#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Exploratory Data Analysis Framework                                                 #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.12                                                                             #
# Filename   : /d8analysis/visual/seaborn/distribution.py                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/d8analysis                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday August 13th 2023 11:30:40 pm                                                 #
# Modified   : Sunday August 13th 2023 11:31:11 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dependency_injector.wiring import inject, Provide

from d8analysis.container import D8AnalysisContainer
from d8analysis.visual.seaborn.base import SeabornVisual
from d8analysis.visual.seaborn.plot import SeabornVisualizer


# ------------------------------------------------------------------------------------------------ #
class Histogram(SeabornVisual):
    """Wrapper for the lineplot method in SeabornVisualizer."""

    @inject
    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        x: str = None,
        y: str = None,
        hue: str = None,
        stat: str = "density",
        element: str = "bars",
        fill: bool = True,
        title: str = None,
        ax: plt.Axes = None,
        visualizer: SeabornVisualizer = Provide[D8AnalysisContainer.visualizer.seaborn],
        *args,
        **kwargs,
    ) -> None:  # pragma: no cover
        self._visualizer = visualizer
        self._data = data
        self._x = x
        self._y = y
        self._hue = hue
        self._stat = stat
        self._element = element
        self._fill = fill
        self._title = title
        self._ax = ax
        self._args = args
        self._kwargs = kwargs

    def plot(self) -> None:
        """Renders the plot"""
        self._visualizer.lineplot(
            data=self._data,
            x=self._x,
            y=self._y,
            hue=self._hue,
            stat=self._stat,
            element=self._element,
            fill=self._fill,
            title=self._title,
            ax=self._ax,
            args=self._args,
            kwargs=self._kwargs,
        )
