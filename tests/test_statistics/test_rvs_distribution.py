#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Enter Project Name in Workspace Settings                                            #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.11                                                                             #
# Filename   : /tests/test_statistics/test_rvs_distribution.py                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : Enter URL in Workspace Settings                                                     #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday May 28th 2023 12:41:00 am                                                    #
# Modified   : Friday August 11th 2023 03:03:02 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import inspect
from datetime import datetime
import pytest
import logging

import numpy as np

from d8analysis.data.generation import RVSDistribution, DISTRIBUTIONS, Distribution


# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"


@pytest.mark.rvs
class TestRVSDistribution:  # pragma: no cover
    # ============================================================================================ #
    def test_rvs(self, dataset, caplog):
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
        data = dataset["Income"].values
        dg = RVSDistribution()
        for dist in DISTRIBUTIONS.keys():
            dg(data=data, distribution=dist)
            assert isinstance(dg.data, np.ndarray)
            assert isinstance(dg.rvs, Distribution)
            assert isinstance(dg.pdf, Distribution)
            assert isinstance(dg.cdf, Distribution)
            assert len(dg.data) == len(data)
            logger.debug(f"\n{dg.rvs}")
            logger.debug(f"\n{dg.pdf}")
            logger.debug(f"\n{dg.cdf}")
            logger.debug(repr(dg.rvs))

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
