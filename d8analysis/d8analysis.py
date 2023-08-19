#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Exploratory Data Analysis Framework                                                 #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.12                                                                             #
# Filename   : /d8analysis/d8analysis.py                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/d8analysis                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday August 19th 2023 06:24:35 pm                                               #
# Modified   : Saturday August 19th 2023 06:34:34 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
from d8analysis.container import D8AnalysisContainer
from d8analysis.service.config import LoggingConfig


def main():
    # Dependencies
    LoggingConfig.set_level("WARNING")
    container = D8AnalysisContainer()
    container.init_resources()
    container.wire(packages=["d8analysis"])


if __name__ == "__main__":
    main()
