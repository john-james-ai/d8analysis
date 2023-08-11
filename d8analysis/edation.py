#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Enter Project Name in Workspace Settings                                            #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.11                                                                             #
# Filename   : /d8analysis/edation.py                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : Enter URL in Workspace Settings                                                     #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday June 21st 2023 03:41:39 am                                                #
# Modified   : Thursday August 10th 2023 10:32:19 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #

from d8analysis.container import d8analysisContainer

# ------------------------------------------------------------------------------------------------ #

if __name__ == "__main__":
    container = d8analysisContainer()
    container.init_resources()
    container.wire(modules=["d8analysis.application.ports.driver"])
