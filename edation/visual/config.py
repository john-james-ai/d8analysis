#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Enter Project Name in Workspace Settings                                            #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.11                                                                             #
# Filename   : /edation/visual/config.py                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : Enter URL in Workspace Settings                                                     #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday May 24th 2023 04:11:27 pm                                                 #
# Modified   : Thursday July 27th 2023 03:36:09 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns

from edation import IMMUTABLE_TYPES, SEQUENCE_TYPES

# ------------------------------------------------------------------------------------------------ #
plt.rcParams["font.size"] = "10"


# ================================================================================================ #
#                                 PLOTTING PARAMETER OBJECTS                                       #
# ================================================================================================ #
# Parameter objects to create, organize and propagate plot configurations
@dataclass
class PlotConfig(ABC):
    """Abstract base class for plot configurations."""

    def as_dict(self) -> dict:
        """Returns a dictionary representation of the the Legend object."""
        return {k: self._export_config(v) for k, v in self.__dict__.items()}

    @classmethod
    def _export_config(cls, v):  # pragma: no cover
        """Returns v with Configs converted to dicts, recursively."""
        if isinstance(v, IMMUTABLE_TYPES):
            return v
        elif isinstance(v, SEQUENCE_TYPES):
            return type(v)(map(cls._export_config, v))
        elif isinstance(v, datetime):
            return v
        elif isinstance(v, dict):
            return v
        elif hasattr(v, "as_dict"):
            return v.as_dict()
        else:
            return v


# ------------------------------------------------------------------------------------------------ #
#                                           COLORS                                                 #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class Colors(PlotConfig):
    cool_black: str = "#002B5B"
    police_blue: str = "#2B4865"
    teal_blue: str = "#256D85"
    pale_robin_egg_blue: str = "#8FE3CF"
    russian_violet: str = "#231955"
    dark_cornflower_blue: str = "#1F4690"
    meat_brown: str = "#E8AA42"
    peach: str = "#FFE5B4"
    dark_blue: str = "#002B5B"
    blue: str = "#1F4690"
    orange: str = "#E8AA42"

    def __post_init__(self) -> None:
        return


# ------------------------------------------------------------------------------------------------ #
#                                            Palettes                                              #
# ------------------------------------------------------------------------------------------------ #
PALETTES = {
    "blues": "Blues",
    "blues_r": "Blues_r",
    "darkblue": sns.dark_palette("#69d", reverse=False, as_cmap=False),
    "darkblue_r": sns.dark_palette("#69d", reverse=True, as_cmap=True),
    "mako": "mako",
    "bluegreen": "crest",
    "paired": "Paired",
    "dark": "dark",
    "colorblind": "colorblind",
    "winter_blue": sns.color_palette(
        [Colors.cool_black, Colors.police_blue, Colors.teal_blue, Colors.pale_robin_egg_blue],
        as_cmap=True,
    ),
    "blue_orange": sns.color_palette(
        [Colors.russian_violet, Colors.dark_cornflower_blue, Colors.meat_brown, Colors.peach],
        as_cmap=True,
    ),
}


# ------------------------------------------------------------------------------------------------ #
#                                            CANVAS                                                #
# ------------------------------------------------------------------------------------------------ #


@dataclass
class Canvas(PlotConfig):
    """Canvas class encapsulating the figure and axes objects."""

    style: str = "whitegrid"
    figsize: tuple = (12, 4)
    nrows: int = 1
    ncols: int = 1
    saturation: float = 0.5
    fontsize: int = 10
    fontsize_title: int = 10
    fig: plt.figure = None
    ax: plt.axes = None
    axs: List = field(default_factory=lambda: [plt.axes])
    palette: str = "Blues_r"

    def __post_init__(self) -> None:
        if self.nrows > 1 or self.ncols > 1:
            figsize = []
            figsize.append(self.figsize[0] * self.ncols)
            figsize.append(self.figsize[1] * self.nrows)
            self.fig, self.axs = plt.subplots(nrows=self.nrows, ncols=self.ncols, figsize=figsize)
        else:
            self.fig, self.ax = plt.subplots(
                nrows=self.nrows, ncols=self.ncols, figsize=self.figsize
            )


# ------------------------------------------------------------------------------------------------ #
#                                            LEGEND                                                #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class LegendConfig(PlotConfig):
    loc: str = "best"
    ncols: int = 1
    fontsize: int = 8
    markerfirst: bool = True
    reverse: bool = False
    frameon: bool = False
    fancybox: bool = True
    framealpha: float = 0.3
    mode: str = None
    title: str = None
    title_fontsize: int = 8
    alignment: str = "left"


# ------------------------------------------------------------------------------------------------ #
#                                       HISTPLOT CONFIG                                            #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class HistplotConfig(PlotConfig):
    stat: str = "count"  # Most used values are ['count', 'probability', 'percent', 'density']
    discrete: bool = False
    cumulative: bool = False
    multiple: str = "layer"  # Valid values ['layer','dodge','stack','fill']
    element: str = "bars"  # Also use 'step'
    fill: bool = False
    kde: bool = True
    legend: bool = True


# ------------------------------------------------------------------------------------------------ #
#                                       HISTPLOT CONFIG                                            #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class KdeplotConfig(PlotConfig):
    cumulative: bool = False
    multiple: str = "layer"  # Valid values ['layer','dodge','stack','fill']
    fill: bool = None
    legend: bool = True


# ------------------------------------------------------------------------------------------------ #
#                                        BARPLOT CONFIG                                            #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class BarplotConfig(PlotConfig):
    estimator: str = "sum"  # ['mean','sum']
    saturation: float = 0.7
    dodge: bool = False


# ------------------------------------------------------------------------------------------------ #
#                                      COUNTPLOT CONFIG                                            #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class CountplotConfig(PlotConfig):
    saturation: float = 0.7
    dodge: bool = False


# ------------------------------------------------------------------------------------------------ #
#                                      POINTPLOT CONFIG                                            #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class PointplotConfig(PlotConfig):
    estimator: str = "mean"
    dodge: bool = False
    linestyles: str = "-"
    join: bool = True

    def __post_init__(self) -> None:
        return


# ------------------------------------------------------------------------------------------------ #
#                                       BOXPLOT CONFIG                                             #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class BoxplotConfig(PlotConfig):
    saturation: float = 0.7
    dodge: bool = True


# ------------------------------------------------------------------------------------------------ #
#                                      SCATTERPLOT CONFIG                                          #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class ScatterplotConfig(PlotConfig):
    size: str = None
    style: str = None
    markers: bool = True
    legend: str = "auto"  # Valid values: ['auto','brief','full',False]
    dodge: bool = True


# ------------------------------------------------------------------------------------------------ #
#                                      LINEPLOT CONFIG                                             #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class LineplotConfig(PlotConfig):
    size: str = None
    style: str = None
    dashes: bool = True
    estimator: str = "mean"
    markers: bool = None
    sort: bool = True
    legend: str = "auto"  # Valid values: ['auto','brief','full',False]
    dodge: bool = True


# ------------------------------------------------------------------------------------------------ #
#                                      HEATMAP CONFIG                                              #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class HeatmapConfig(PlotConfig):
    vmin: float = None
    vmax: float = None
    cmap: str = str
    center: float = None
    annot: bool = True
    fmt: str = None
    linewidths: float = 0
    linecolor: str = "white"
    cbar: bool = True
    cbar_kws: dict = None
    square: bool = False
    xticklabels: str = "auto"
    yticklabels: str = "auto"
