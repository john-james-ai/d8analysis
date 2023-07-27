#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Enter Project Name in Workspace Settings                                            #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.10                                                                             #
# Filename   : /tests/test_statistics/test_independence/test_chisq_ind.py                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : Enter URL in Workspace Settings                                                     #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday June 5th 2023 09:32:36 pm                                                    #
# Modified   : Thursday July 27th 2023 08:48:22 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import inspect
from datetime import datetime
import pytest
import logging
import pandas as pd

from edation.stats.independence.chisquare import ChiSquareIndependenceTest
from edation.stats.base import StatTestProfile


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
        test = ChiSquareIndependenceTest()
        test(data=dataset[["Education", "Marital Status"]])
        assert "Chi" in test.result.test
        assert isinstance(test.result.H0, str)
        assert isinstance(test.result.pvalue, float)
        assert test.result.alpha == 0.05
        assert isinstance(test.result.data, pd.DataFrame)
        assert isinstance(test.profile, StatTestProfile)
        logging.debug(test.result)

        # Test Result
        dfc = test.result._combine_contingency_tables()
        assert isinstance(dfc, pd.DataFrame)
        logger.debug(dfc)

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
    def test_arguments(self, dataset, caplog):
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
        test = ChiSquareIndependenceTest()
        # Test two arrays no dataframe
        x = dataset["Education"]
        y = dataset["Marital Status"]
        test(x=x, y=y)
        assert "Chi" in test.result.test
        assert isinstance(test.result.x, str)
        assert isinstance(test.result.y, str)
        assert test.result.x == "Sample 1"
        assert test.result.y == "Sample 2"

        test(data=dataset, x="Education", y="Income")

        with pytest.raises(ValueError):
            test()
        with pytest.raises(ValueError):
            test(x=x, y="something")
        with pytest.raises(ValueError):
            test(dataset, x="no", y="way")
        with pytest.raises(ValueError):
            test(dataset, x=x, y=y)
        with pytest.raises(ValueError):
            test(dataset["Income"], x=x, y=y)
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
