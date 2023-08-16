#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Exploratory Data Analysis Framework                                                 #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.10                                                                             #
# Filename   : /tests/test_quantitative/test_inferential/test_relational/test_spearman.py          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/d8analysis                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday June 7th 2023 09:15:17 pm                                                 #
# Modified   : Tuesday August 15th 2023 05:12:36 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import inspect
from datetime import datetime
import pytest
import logging
import pandas as pd

from d8analysis.quantitative.inferential.relational.spearman import SpearmanCorrelationTest
from d8analysis.quantitative.inferential.base import StatTestProfileTwo


# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"


@pytest.mark.stats
@pytest.mark.corr
@pytest.mark.spearman
class TestSpearman:  # pragma: no cover
    # ============================================================================================ #
    def test_spearman(self, dataset, caplog):
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
        test = SpearmanCorrelationTest(data=dataset, a="Income", b="Age")
        test.run()
        assert "Spearman" in test.result.test
        assert isinstance(test.result.H0, str)
        assert isinstance(test.result.value, float)
        assert isinstance(test.result.pvalue, float)
        assert test.result.alpha == 0.05
        assert isinstance(test.result.result, str)
        assert isinstance(test.result.data, pd.DataFrame)
        assert isinstance(test.profile, StatTestProfileTwo)
        logging.debug(test.result)

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
