#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Exploratory Data Analysis Framework                                                 #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.11                                                                             #
# Filename   : /d8analysis/__init__.py                                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/d8analysis                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday May 26th 2023 11:59:46 pm                                                    #
# Modified   : Saturday August 19th 2023 06:07:56 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
# Data Package
# Generation
from d8analysis.data.generation import RVSDistribution

# Qualitative Package
# - Quantitative Package
# -- Descriptive Subpackage
from d8analysis.quantitative.descriptive.categorical import (
    CategoricalFreqDistribution,
    CategoricalStats,
)
from d8analysis.quantitative.descriptive.continuous import (
    ContinuousFreqDistribution,
    ContinuousStats,
)

# Inferential Subpackage
from d8analysis.quantitative.inferential.centrality.ttest import TTest, TTestResult
from d8analysis.quantitative.inferential.distribution.kstest import KSTest, KSTestResult
from d8analysis.quantitative.inferential.relational.chisquare import (
    ChiSquareIndependenceResult,
    ChiSquareIndependenceTest,
)
from d8analysis.quantitative.inferential.relational.pearson import (
    PearsonCorrelationResult,
    PearsonCorrelationTest,
)
from d8analysis.quantitative.inferential.relational.spearman import (
    SpearmanCorrelationResult,
    SpearmanCorrelationTest,
)

# Visualization Package
# - Seaborn
from d8analysis.visual.seaborn.association import PairPlot, ScatterPlot, JointPlot, LinePlot
from d8analysis.visual.seaborn.distribution import (
    Histogram,
    HistPDFPlot,
    ECDFPlot,
    KDEPlot,
    BoxPlot,
    ViolinPlot,
    PdfCdfPlot,
)
from d8analysis.visual.seaborn.centrality import Barplot
from d8analysis.visual.seaborn.grid import GridPlot
