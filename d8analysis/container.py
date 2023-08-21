#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Exploratory Data Analysis Framework                                                 #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.10                                                                             #
# Filename   : /d8analysis/container.py                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/d8analysis                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday March 27th 2023 07:02:56 pm                                                  #
# Modified   : Monday August 21st 2023 01:49:41 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Framework Dependency Container"""
from dependency_injector import containers, providers

from d8analysis.visual.seaborn.config import SeabornCanvas
from d8analysis.visual.seaborn.plot import SeabornVisualizer


# ------------------------------------------------------------------------------------------------ #
#                                    VISUALIZER CONTAINER                                          #
# ------------------------------------------------------------------------------------------------ #
class VisualizerContainer(containers.DeclarativeContainer):
    canvas = providers.Dependency()
    seaborn = providers.Factory(SeabornVisualizer, canvas=canvas)


# ------------------------------------------------------------------------------------------------ #
#                                     CANVAS CONTAINER                                             #
# ------------------------------------------------------------------------------------------------ #
class CanvasContainer(containers.DeclarativeContainer):
    seaborn = providers.Factory(SeabornCanvas)


# ------------------------------------------------------------------------------------------------ #
#                                          FRAMEWORK                                               #
# ------------------------------------------------------------------------------------------------ #
class D8AnalysisContainer(containers.DeclarativeContainer):
    canvas = providers.Container(CanvasContainer)

    visualizer = providers.Container(VisualizerContainer, canvas=canvas.seaborn)
