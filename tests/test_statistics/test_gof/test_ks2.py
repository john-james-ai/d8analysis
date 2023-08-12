#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Exploratory Data Analysis Framework                                                 #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.10                                                                             #
# Filename   : /tests/test_statistics/test_gof/test_ks2.py                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/d8analysis                                         #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday June 5th 2023 09:32:36 pm                                                    #
# Modified   : Friday August 11th 2023 09:55:32 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import inspect
from datetime import datetime
import pytest
import logging
import pandas as pd

from d8analysis.quantitative.statistical.distribution.kstwo import KSTwoTest
from d8analysis.quantitative.statistical.base import StatTestProfile


# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"


@pytest.mark.stats
@pytest.mark.gof
@pytest.mark.ks2
class TestKSTwoTest:  # pragma: no cover
    # ============================================================================================ #
    def test_gof_reject(self, dataset, caplog):
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
        female = dataset.loc[dataset["Gender"] == "Female"]["Income"]
        male = dataset.loc[dataset["Gender"] == "Male"]["Income"]
        female.name = "female"
        male.name = "male"
        test = KSTwoTest()
        test(sample1=female, sample2=male)
        assert "Kolmo" in test.result.test
        assert test.result.pvalue < 0.05
        assert test.result.value < 1
        assert isinstance(test.result.H0, str)
        assert isinstance(test.result.pvalue, float)
        assert test.result.alpha == 0.05
        assert isinstance(test.sample1, pd.Series)
        assert isinstance(test.sample2, pd.Series)
        assert isinstance(test.profile, StatTestProfile)
        assert isinstance(test.profile, StatTestProfile)
        logger.debug(f"\n{test.result}")
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
    def test_gof_fail_to_reject(self, dataset, caplog):
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
        test = KSTwoTest()
        test(sample1=dataset["Income"], sample2=dataset["Income"])
        assert "Kolmo" in test.result.test
        assert test.result.pvalue > 0.05
        assert test.result.value < 1
        assert isinstance(test.result.H0, str)
        assert isinstance(test.result.pvalue, float)
        assert test.result.alpha == 0.05
        assert isinstance(test.sample1, pd.Series)
        assert isinstance(test.sample2, pd.Series)
        assert isinstance(test.profile, StatTestProfile)
        assert isinstance(test.profile, StatTestProfile)
        logger.debug(f"\n{test.result}")
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
