#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Enter Project Name in Workspace Settings                                            #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.10                                                                             #
# Filename   : /tests/test_statistics/test_profile.py                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : Enter URL in Workspace Settings                                                     #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday June 5th 2023 06:13:21 pm                                                    #
# Modified   : Monday June 5th 2023 07:13:36 pm                                                    #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import inspect
from datetime import datetime
import pytest
import logging

from explorer.stats.profile import StatTestProfileOne
from explorer.service.io import IOService

ID = "x2gof"
PROFILES = "config/stats.yml"
# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"


@pytest.mark.profile
class TestStatProfile:  # pragma: no cover
    # ============================================================================================ #
    def test_profile(self, caplog):
        start = datetime.now()
        logger.info(
            "\n\nStarted {} {} at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                start.strftime("%I:%M:%S %p"),
                start.strftime("%m/%d/%Y"),
            )
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        profiles = IOService.read(PROFILES)
        p = profiles[ID]
        profile = StatTestProfileOne.create(id=ID)
        assert profile.id == ID
        assert profile.name == p["name"]
        assert profile.description == p["description"]
        assert profile.statistic == p["statistic"]
        assert profile.analysis == p["analysis"]
        assert profile.hypothesis == p["hypothesis"]
        assert profile.h0 == p["h0"]
        assert profile.parametric == p["parametric"]
        assert profile.min_sample_size == p["min_sample_size"]
        assert profile.assumptions == p["assumptions"]

        logger.debug(repr(profile))
        logger.debug(profile)

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            "\nCompleted {} {} in {} seconds at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                duration,
                end.strftime("%I:%M:%S %p"),
                end.strftime("%m/%d/%Y"),
            )
        )
        logger.info(single_line)
