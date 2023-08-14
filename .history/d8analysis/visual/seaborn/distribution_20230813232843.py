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
# Created    : Sunday August 13th 2023 11:25:29 pm                                                 #
# Modified   : Sunday August 13th 2023 11:28:43 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Plotizations that Reveal Associations between Variables."""
from typing import Union

import pandas as pd
import numpy as np
from dependency_injector.wiring import inject, Provide

from d8analysis.container import D8AnalysisContainer
from d8analysis.visual.seaborn.base import SeabornVisual
from d8analysis.visual.seaborn.plot import SeabornVisualizer


# ------------------------------------------------------------------------------------------------ #
class PairPlot(SeabornVisual):  # pragma: no cover
    """Wrapper for the lineplot method in SeabornVisualizer."""

    @inject
    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        vars: list = None,
        hue: str = None,
        title: str = None,
        visualizer: SeabornVisualizer = Provide[D8AnalysisContainer.visualizer.seaborn],
        *args,
        **kwargs,
    ) -> None:  # pragma: no cover
        self._visualizer = visualizer
        self._data = data
        self._vars = vars
        self._hue = hue
        self._title = title
        self._args = args
        self._kwargs = kwargs

    def plot(self) -> None:
        """Renders the plot"""
        self._visualizer.pairplot(
            data=self._data,
            vars=self._vars,
            hue=self._hue,
            title=self._title,
            ax=self._ax,
            args=self._args,
            kwargs=self._kwargs,
        )
