#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Exploratory Data Analysis Framework                                                 #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.12                                                                             #
# Filename   : /tests/test_quantitative/test_descriptive/test_stats.py                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/d8analysis                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday August 15th 2023 08:02:48 pm                                                #
# Modified   : Tuesday August 15th 2023 08:26:53 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import inspect
from datetime import datetime
import pytest
import logging

from d8analysis.quantitative.descriptive import categorical, continuous


# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"


@pytest.mark.stats
class TestStats:  # pragma: no cover
    # ============================================================================================ #
    def test_continuous(self, dataset, caplog):
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
        stats = continuous.DescriptiveStats.describe(x=dataset["Income"])
        assert stats.name == "Income"
        assert isinstance(stats.length, int)
        assert isinstance(stats.count, int)
        assert isinstance(stats.size, int)
        assert isinstance(stats.min, (int, float))
        assert isinstance(stats.q25, (int, float))
        assert isinstance(stats.mean, (int, float))
        assert isinstance(stats.q75, (int, float))
        assert isinstance(stats.max, (int, float))
        assert isinstance(stats.range, (int, float))
        assert isinstance(stats.std, float)
        assert isinstance(stats.var, float)
        assert isinstance(stats.skew, float)
        assert isinstance(stats.kurtosis, float)
        logging.debug(stats)

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

    # ============================================================================================ #
    def test_categorical(self, dataset, caplog):
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
        stats = categorical.DescriptiveStats.describe(x=dataset["Education"])
        assert stats.name == "Education"
        assert isinstance(stats.length, int)
        assert isinstance(stats.count, int)
        assert isinstance(stats.size, int)
        assert isinstance(stats.mode, (str, int, float))
        assert isinstance(stats.unique, int)
        logging.debug(stats)

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            "\n\tCompleted {} {} in {} seconds at {} on {}".format(
                self.__class__.__name__,
                inspect.stack()[0][3],
                duration,
                end.strftime("%I:%M:%S %p"),
                end.strftime("%m/%d/%Y"),
            )
        )
        logger.info(single_line)
