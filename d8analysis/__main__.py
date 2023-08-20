#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Exploratory Data Analysis Framework                                                 #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.12                                                                             #
# Filename   : /d8analysis/__main__.py                                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/d8analysis                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday August 20th 2023 07:22:18 pm                                                 #
# Modified   : Sunday August 20th 2023 07:34:57 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from d8analysis.service.config import LoggingConfig
from d8analysis.container import D8AnalysisContainer


# ------------------------------------------------------------------------------------------------ #
if __name__ == "__main__":  # pragma: no cover
    # Dependencies
    LoggingConfig.set_level("WARNING")
    container = D8AnalysisContainer()
    container.init_resources()
    container.wire(modules=[__name__], packages=["d8analysis"])
