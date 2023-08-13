#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Exploratory Data Analysis Framework                                                 #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.11                                                                             #
# Filename   : /d8analysis/visual/config.py                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/d8analysis                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday May 24th 2023 04:11:27 pm                                                 #
# Modified   : Saturday August 12th 2023 06:22:15 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations
from abc import ABC
import math
from dataclasses import dataclass, field
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

from d8analysis import IMMUTABLE_TYPES, SEQUENCE_TYPES

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
    crimson: str = "#BA0020"

    def __post_init__(self) -> None:
        return


# ------------------------------------------------------------------------------------------------ #
#                                            PALETTES                                              #
# ------------------------------------------------------------------------------------------------ #
SEABORN_PALETTES = {
    "darkblue": sns.dark_palette("#69d", reverse=False, as_cmap=False),
    "darkblue_r": sns.dark_palette("#69d", reverse=True, as_cmap=False),
    "winter_blue": sns.color_palette(
        [Colors.cool_black, Colors.police_blue, Colors.teal_blue, Colors.pale_robin_egg_blue],
        as_cmap=True,
    ),
    "blue_orange": sns.color_palette(
        [Colors.russian_violet, Colors.dark_cornflower_blue, Colors.meat_brown, Colors.peach],
        as_cmap=True,
    ),
}


@dataclass
class Palettes(PlotConfig):
    blues: str = "Blues"
    blues_r: str = "Blues_r"
    mako: str = "mako"
    bluegreen: str = "crest"
    paired: str = "Paired"
    dark: str = "dark"
    colorblind: str = "colorblind"
    seaborn_palettes: dict = field(default_factory=dict)


# ------------------------------------------------------------------------------------------------ #
#                                            LEGEND                                                #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class LegendConfig(PlotConfig):
    loc: str = "best"
    ncols: int = 1
    fontsize: int = 8
    labels: list[str] = field(default_factory=list)
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


# ------------------------------------------------------------------------------------------------ #
#                                            CANVAS                                                #
# ------------------------------------------------------------------------------------------------ #


@dataclass
class Canvas(PlotConfig):
    """Canvas class encapsulating figure level configuration."""

    width: int = 12  # The maximum width of the canvas
    height: int = 4  # The height of a single row.
    maxcols: int = 2  # The maximum number of columns in a multi-plot visualization.
    palette: str = "Blues_r"  # Seaborn palette or matplotlib colormap
    style: str = "whitegrid"  # A Seaborn aesthetic
    saturation: float = 0.5
    fontsize: int = 10
    fontsize_title: int = 10
    colors: Colors = Colors()
    palettes: Palettes = Palettes()
    legend_config: LegendConfig = LegendConfig()
    histplot_config: HistplotConfig = HistplotConfig()
    kdeplot_config: KdeplotConfig = KdeplotConfig()
    heatmap_config: HeatmapConfig = HeatmapConfig()
    lineplot_config: LineplotConfig = LineplotConfig()
    scatterplot_config: ScatterplotConfig = ScatterplotConfig()
    boxplot_config: BoxplotConfig = BoxplotConfig()
    pointplot_config: PointplotConfig = PointplotConfig()
    countplot_config: CountplotConfig = CountplotConfig()
    barplot_config: BarplotConfig = BarplotConfig()

    def get_figaxes(self, nplots: int = 1, figsize: tuple = None) -> Canvas:
        """Configures the figure and axes objects.

        Args:
            nplots (int): The number of plots to be rendered on the canvas.
            figsize (tuple[int,int]): Plot width and row height.
        """
        figsize = figsize or (self.width, self.height)

        if nplots == 1:
            fig, axes = plt.subplots(figsize=figsize)
        else:
            nrows = math.ceil(nplots / self.maxcols)
            ncols = min(self.maxcols, nplots)

            fig = plt.figure(layout="constrained", figsize=figsize)
            gs = GridSpec(nrows=nrows, ncols=ncols, figure=fig)

            axes = []
            for idx in range(nplots):
                row = int(idx / ncols)
                col = idx % ncols

                if idx < nplots - 1:
                    ax = fig.add_subplot(gs[row, col])
                else:
                    ax = fig.add_subplot(gs[row, col:])
                axes.append(ax)

        return fig, axes
