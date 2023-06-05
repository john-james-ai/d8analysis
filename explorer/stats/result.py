#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Enter Project Name in Workspace Settings                                            #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.11                                                                             #
# Filename   : /explorer/stats/result.py                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : Enter URL in Workspace Settings                                                     #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday May 28th 2023 06:25:22 pm                                                    #
# Modified   : Monday June 5th 2023 06:20:45 pm                                                    #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Statistical Test Result Module"""
from dataclasses import dataclass, field
from typing import List

from explorer.stats.profile import StatTestResult


# ------------------------------------------------------------------------------------------------ #
@dataclass
class StatTestResultOne(StatTestResult):
    x: str  # Variable name

    def __repr__(self) -> str:
        return "{}({})".format(
            self.__class__.__name__,
            ", ".join("{}={!r}".format(k, v) for k, v in self.__dict__.items()),
        )

    def __str__(self) -> str:
        s = ""
        width = 32
        for k, v in self.__dict__.items():
            s += f"\t{k.rjust(width,' ')} | {v}\n"
        return s


# ------------------------------------------------------------------------------------------------ #
@dataclass
class StatTestResultOneGoF(StatTestResultOne):
    distribution: str
    method: str = "approx"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(test={self.test}, x={self.x}, statistic={self.statistic}, pvalue={self.pvalue})"

    def __str__(self) -> str:
        width = 32
        s = f"\t{'Analyzer:'.rjust(width,' ')} | {self.analyzer}\n"
        s += f"\t{'Test:'.rjust(width,' ')} | {self.test}\n"
        s += f"\t{'Distribution:'.rjust(width,' ')} | {self.distribution}\n"
        s += f"\t{'Method:'.rjust(width,' ')} | {self.method}\n"
        s += f"\t{'Variable:'.rjust(width,' ')} | {self.x}\n"
        s += f"\t{'Statistic:'.rjust(width,' ')} | {self.statistic}\n"
        s += f"\t{'p-Value:'.rjust(width,' ')} | {self.pvalue}\n"
        return s


# ------------------------------------------------------------------------------------------------ #
@dataclass
class StatTestResultTwo(StatTestResult):
    a: str
    b: str

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(test={self.test}, a={self.a}, b={self.b}, statistic={self.statistic}, pvalue={self.pvalue})"

    def __str__(self) -> str:
        width = 32
        s = f"\t{'Analyzer:'.rjust(width,' ')} | {self.analyzer}\n"
        s += f"\t{'Test:'.rjust(width,' ')} | {self.test}\n"
        s += f"\t{'a:'.rjust(width,' ')} | {self.a}\n"
        s += f"\t{'b:'.rjust(width,' ')} | {self.b}\n"
        s += f"\t{'Statistic:'.rjust(width,' ')} | {self.statistic}\n"
        s += f"\t{'p-Value:'.rjust(width,' ')} | {self.pvalue}\n"
        return s


# ------------------------------------------------------------------------------------------------ #
@dataclass
class StatTestResultMV(StatTestResult):
    vars: List = field(default_factory=lambda: [str])
