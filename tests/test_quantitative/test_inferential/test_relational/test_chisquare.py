#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Exploratory Data Analysis Framework                                                 #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.10                                                                             #
# Filename   : /tests/test_quantitative/test_inferential/test_relational/test_chisquare.py         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/d8analysis                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday June 5th 2023 09:32:36 pm                                                    #
# Modified   : Tuesday August 15th 2023 05:19:56 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import inspect
from datetime import datetime
import pytest
import logging
import pandas as pd

from d8analysis.quantitative.inferential.relational.chisquare import ChiSquareIndependenceTest
from d8analysis.quantitative.inferential.base import StatTestProfile


# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"


@pytest.mark.stats
@pytest.mark.ind
@pytest.mark.x2ind
class TestX2Independence:  # pragma: no cover
    # ============================================================================================ #
    def test_x2(self, dataset, caplog):
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
        test = ChiSquareIndependenceTest(data=dataset, a="Education", b="Credit Rating")
        test.run()
        assert "Chi" in test.result.test
        assert isinstance(test.result.H0, str)
        assert isinstance(test.result.value, float)
        assert isinstance(test.result.pvalue, float)
        assert test.result.alpha == 0.05
        assert isinstance(test.result.data, pd.DataFrame)
        assert isinstance(test.profile, StatTestProfile)
        assert isinstance(test.result.result, str)
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
