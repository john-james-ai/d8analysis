#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Exploratory Data Analysis Framework                                                 #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.12                                                                             #
# Filename   : /d8analysis/service/config.py                                                       #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/d8analysis                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday August 15th 2023 05:36:24 pm                                                #
# Modified   : Monday August 21st 2023 03:34:11 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import os
from dotenv import load_dotenv

from d8analysis.service.io import IOService

# ------------------------------------------------------------------------------------------------ #
load_dotenv()


# ------------------------------------------------------------------------------------------------ #
class LoggingConfig:  # pragma: no cover
    """Manages logging configuration"""

    __filepath = os.getenv("LOGGING_CONFIG_FILEPATH")

    @classmethod
    def get(cls) -> dict:
        """Returns the configuration"""
        return IOService.read(LoggingConfig.__filepath)

    @classmethod
    def get_level(cls) -> str:
        """Retursn logging level for the console."""

        config = IOService.read(LoggingConfig.__filepath)
        return config["logging"]["handlers"]["console"]["level"]

    @classmethod
    def set_level(cls, level: str = "DEBUG") -> None:
        """Sets logging level for the console.

        Args:
            level (str): The logging level in all caps.
        """

        config = IOService.read(LoggingConfig.__filepath)
        config["logging"]["handlers"]["console"]["level"] = level
        IOService.write(filepath=LoggingConfig.__filepath, data=config)
