#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Explorer                                                                            #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.11                                                                             #
# Filename   : /explorer/explorer.py                                                               #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/explorer                                           #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday June 21st 2023 03:41:39 am                                                #
# Modified   : Wednesday June 21st 2023 09:23:21 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #

from explorer.container import ExplorerContainer

# ------------------------------------------------------------------------------------------------ #

if __name__ == "__main__":
    container = ExplorerContainer()
    container.init_resources()
    container.wire(modules=[__name__, "explorer.domain.base"])
