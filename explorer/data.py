#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Enter Project Name in Workspace Settings                                            #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.10                                                                             #
# Filename   : /explorer/data.py                                                                   #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : Enter URL in Workspace Settings                                                     #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday May 27th 2023 12:29:58 am                                                  #
# Modified   : Saturday May 27th 2023 01:33:43 am                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Module to handle data related cross-cutting concerns."""
import functools

# ------------------------------------------------------------------------------------------------ #


def check_args(*args1):
    """Converts argument types to those designated in the decorator."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            args = list(args)  # convert args to a list here
            for i in range(len(args)):
                if type(args[i]) != args1[i] and args[i] is not None:
                    args[i] = args1[i](args[i])
            result = func(*args, **kwargs)  # *args works for both tuples and lists
            return result

        return wrapper

    return decorator
