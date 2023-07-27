#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Enter Project Name in Workspace Settings                                            #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.11                                                                             #
# Filename   : /edation/explorer.py                                                                #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : Enter URL in Workspace Settings                                                     #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday June 21st 2023 03:41:39 am                                                #
# Modified   : Thursday July 27th 2023 08:51:56 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #

from edation.container import EdationContainer

# ------------------------------------------------------------------------------------------------ #

if __name__ == "__main__":
    container = EdationContainer()
    container.init_resources()
    container.wire(modules=["edation.application.ports.driver"])
