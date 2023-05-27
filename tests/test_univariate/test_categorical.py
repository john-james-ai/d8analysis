#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Explorer                                                                            #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.10                                                                             #
# Filename   : /tests/test_univariate/test_categorical.py                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/explorer                                           #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday May 26th 2023 11:10:42 pm                                                    #
# Modified   : Saturday May 27th 2023 04:55:39 am                                                  #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import inspect
from datetime import datetime
import pytest
import logging
import pandas as pd
import numpy as np

from explorer.base import StatTestOne
from explorer.univariate import CategoricalOne


# ------------------------------------------------------------------------------------------------ #
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"
# ------------------------------------------------------------------------------------------------ #
NAME = "educ"


@pytest.mark.cat1
class TestCategoricalOne:  # pragma: no cover
    # ============================================================================================ #
    def test_describe(self, dataset, caplog):
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
        a = CategoricalOne(data=dataset)
        result = a.describe(name=NAME)
        assert isinstance(result, pd.Series)
        assert result["count"] == len(dataset[NAME])
        assert result["unique"] == dataset[NAME].nunique()
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
    def test_frequency(self, dataset, caplog):
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
        a = CategoricalOne(data=dataset)
        result = a.frequency(name=NAME)
        logger.debug(f"\nresult={result}\n")
        assert result.shape[1] == 3
        assert result.shape[0] == dataset[NAME].nunique()

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

    # ============================================================================================ #
    def test_distribution(self, dataset, caplog):
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
        a = CategoricalOne(data=dataset)
        result = a.test_distribution(name=NAME)
        assert isinstance(result, StatTestOne)
        assert isinstance(result.test, str)
        assert isinstance(result.statistic, float)
        assert isinstance(result.pvalue, float)
        assert isinstance(result.x, str)
        logger.debug(result)

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

    # ============================================================================================ #
    def test_distribution_expected_mismatch(self, dataset, caplog):
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
        a = CategoricalOne(data=dataset)

        # Test Exception
        idx = ["high"]
        value = [222]
        e = pd.Series(value)
        e.index = idx
        with pytest.raises(ValueError):
            a.test_distribution(name=NAME, expected=e)

        e = "string of some sort"
        with pytest.raises(ValueError):
            _ = a.test_distribution(name=NAME, expected=e)

        # Test Different Distributions
        idx = dataset[NAME].unique()
        values = np.random.randint(low=1, high=5, size=len(idx))
        values[-1] = len(dataset) - sum(values[:-1])
        assert sum(values) == len(dataset)
        e = pd.Series(values)
        e.index = idx
        result = a.test_distribution(name=NAME, expected=e)
        assert result.pvalue < 0.05
        assert isinstance(result, StatTestOne)
        assert isinstance(result.test, str)
        assert isinstance(result.statistic, float)
        assert isinstance(result.pvalue, float)
        assert isinstance(result.x, str)
        logger.debug(result)

        # Test Equal Distributions
        e = dataset[NAME].value_counts()
        result = a.test_distribution(name=NAME, expected=e)
        assert result.pvalue > 0.05
        assert isinstance(result, StatTestOne)
        assert isinstance(result.test, str)
        assert isinstance(result.statistic, float)
        assert isinstance(result.pvalue, float)
        assert isinstance(result.x, str)
        logger.debug(result)

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
