#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Enter Project Name in Workspace Settings                                            #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.11                                                                             #
# Filename   : /tests/test_data/test_dataset.py                                                    #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : Enter URL in Workspace Settings                                                     #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday August 10th 2023 08:42:57 pm                                               #
# Modified   : Friday August 11th 2023 04:29:21 am                                                 #
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

from d8analysis.data.credit_score import CreditScoreDataset


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
        # Length
        assert (len(ds)) == 164
        # Get item
        assert isinstance(ds[10], pd.Series)
        # Summmary
        logger.debug(ds.summary)
        # Columns
        assert len(ds.columns) == 8
        # Size
        assert isinstance(ds.size, np.int64)
        # Info
        assert isinstance(ds.info, pd.DataFrame)
        logger.debug(ds.info)
        # Overview
        logger.debug(ds.overview)
        # Sample
        assert len(ds.sample()) == 5
        # Dtypes
        dt = ds.dtypes
        assert isinstance(dt, pd.DataFrame)
        logger.debug(dt)
        # Select
        df = ds.select(include=["Gender", "Education"])
        assert df.shape[1] == 2
        df = ds.select(exclude=["Gender"])
        assert df.shape[1] == dataset.shape[1] - 1
        # Subset
        condition = lambda df: df["Gender"] == "Male"  # noqa
        df = ds.subset(condition=condition)
        assert df.shape[0] < dataset.shape[0]
        # Head
        df = ds.head()
        assert len(df) == 5
        # Describe
        df = ds.describe(include=["category", "object"])
        logger.debug(df)
        df = ds.describe(exclude=["object"])
        logger.debug(df)
        # Unique
        df = ds.unique(columns=["Gender", "Education"])
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 10

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
