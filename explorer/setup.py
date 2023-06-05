#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Enter Project Name in Workspace Settings                                            #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.10                                                                             #
# Filename   : /explorer/setup.py                                                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : Enter URL in Workspace Settings                                                     #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday June 5th 2023 04:58:20 pm                                                    #
# Modified   : Monday June 5th 2023 05:44:05 pm                                                    #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
import logging

import pandas as pd

from explorer.service.io import IOService

# ------------------------------------------------------------------------------------------------ #
SOURCE = "notes/Statistical Tests.xlsx"
DEST = "config/stats.yml"
# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)


def get_stat_tests(source: str) -> pd.DataFrame:
    return pd.read_excel(source, sheet_name="stats", index_col="id")


def save_as_yaml(df: pd.DataFrame, destination: str) -> None:
    d = df.to_dict(orient="index")
    IOService.write(filepath=destination, data=d)


def report(df: pd.DataFrame) -> None:
    report = df[["name", "atype"]]
    print(f"Statistical Tests Loaded\n{report}")


def main():
    df = get_stat_tests(source=SOURCE)
    save_as_yaml(df=df, destination=DEST)
    report(df=df)


if __name__ == "__main__":
    main()
