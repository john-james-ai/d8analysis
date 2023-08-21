#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Exploratory Data Analysis Framework                                                 #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.12                                                                             #
# Filename   : /d8analysis/__init__.py                                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/d8analysis                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday May 26th 2023 11:59:46 pm                                                    #
# Modified   : Monday August 21st 2023 02:37:24 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import logging
from d8analysis.container import D8AnalysisContainer

# ------------------------------------------------------------------------------------------------ #
logging.getLogger(__name__).addHandler(logging.NullHandler())
# ------------------------------------------------------------------------------------------------ #
container = D8AnalysisContainer()
container.init_resources()
container.wire(packages=["d8analysis"])
