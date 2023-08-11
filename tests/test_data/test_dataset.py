#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Data Exploration Framework                                                          #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.11                                                                             #
# Filename   : /tests/test_data/test_dataset.py                                                    #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/edation                                            #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday August 10th 2023 08:42:57 pm                                               #
# Modified   : Thursday August 10th 2023 09:24:23 pm                                               #
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

from edation.data.credit_score import CreditScoreDataset


# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"


@pytest.mark.dataset
class TestDataset:  # pragma: no cover
    # ============================================================================================ #
    def test_dataset(self, dataset, caplog):
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
        ds = CreditScoreDataset(df=dataset)
        assert (len(ds)) == 164
        assert isinstance(ds[10], pd.Series)
        logger.debug(ds.summary)
        assert len(ds.columns) == 8
        assert isinstance(ds.size, np.int64)
        logger.debug(ds.info())
        logger.debug(ds.overview())
        assert len(ds.sample()) == 5
        df = ds.select(include=["Gender", "Education"])
        assert df.shape[1] == 2
        df = ds.select(exclude=["Gender"])
        assert df.shape[1] == dataset.shape[1] - 1
        condition = lambda df: df["Gender"] == "Male"  # noqa
        df = ds.subset(condition=condition)
        assert df.shape[0] < dataset.shape[0]
        df = ds.head()
        assert len(df) == 5
        logger.debug(ds.summary())

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