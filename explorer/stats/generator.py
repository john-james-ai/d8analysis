#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Enter Project Name in Workspace Settings                                            #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.11                                                                             #
# Filename   : /explorer/stats/generator.py                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : Enter URL in Workspace Settings                                                     #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday May 27th 2023 08:56:02 pm                                                  #
# Modified   : Sunday June 4th 2023 09:24:50 pm                                                    #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Statistics Module"""
from __future__ import annotations
import logging

from scipy import stats
import numpy as np

logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
#                                    SCIPY DISTRIBUTIONS                                           #
# ------------------------------------------------------------------------------------------------ #
DISTRIBUTIONS = {
    "normal": stats.norm,
    "chi2": stats.chi2,
    "exponential": stats.expon,
    "gamma": stats.gamma,
    "logistic": stats.logistic,
    "uniform": stats.uniform,
}


# ------------------------------------------------------------------------------------------------ #
#                              RANDOM DATA GENERATORS                                              #
# ------------------------------------------------------------------------------------------------ #
def normal(data: np.ndarray) -> np.ndarray:
    """Generates random variates for the normal distribution

    Args:
        name (str): Variable name in the dataset
    """
    loc, scale = get_params(data=data, distribution="normal")
    return stats.norm.rvs(loc=loc, scale=scale, size=len(data))


def chi2(data: np.ndarray) -> np.ndarray:
    """Generates random variates for the chi-squared distribution

    Args:
        name (str): Variable name in the dataset
    """
    a, loc, scale = get_params(data=data, distribution="chi2")
    df = 2 * a
    return stats.chi2.rvs(df=df, loc=loc, scale=scale, size=len(data))


def exponential(data: np.ndarray) -> np.ndarray:
    """Generates random variates for the exponential distribution

    Args:
        name (str): Variable name in the dataset
    """
    loc, scale = get_params(data=data, distribution="exponential")
    return stats.expon.rvs(loc=loc, scale=scale, size=len(data))


def gamma(data: np.ndarray) -> np.ndarray:
    """Generates random variates for the gamma distribution

    Args:
        name (str): Variable name in the dataset
    """
    a, loc, scale = get_params(data=data, distribution="gamma")
    return stats.gamma.rvs(a=a, loc=loc, scale=scale, size=len(data))


def logistic(data: np.ndarray) -> np.ndarray:
    """Generates random variates for the logistic distribution

    Args:
        name (str): Variable name in the dataset
    """
    loc, scale = get_params(data=data, distribution="logistic")
    return stats.logistic.rvs(loc=loc, scale=scale, size=len(data))


def lognorm(data: np.ndarray) -> np.ndarray:
    """Generates random variates for the log normal distribution

    Args:
        name (str): Variable name in the dataset
    """
    s, loc, scale = get_params(data=data, distribution="lognorm")
    return stats.lognorm.rvs(s=s, loc=loc, scale=scale, size=len(data))


def pareto(data: np.ndarray) -> np.ndarray:
    """Generates random variates for the pareto distribution

    Args:
        name (str): Variable name in the dataset
    """
    b, loc, scale = get_params(data=data, distribution="pareto")
    return stats.pareto.rvs(b=b, loc=loc, scale=scale, size=len(data))


def uniform(data: np.ndarray) -> np.ndarray:
    """Generates random variates for the uniform distribution

    Args:
        name (str): Variable name in the dataset
    """
    loc, scale = get_params(data=data, distribution="uniform")
    return stats.uniform.rvs(loc=loc, scale=scale, size=len(data))


def get_params(data: np.ndarray, distribution: str) -> tuple:
    """Obtains the distribution parameters estimated from the data."""
    try:
        return DISTRIBUTIONS[distribution].fit(data)
    except AttributeError as e:  # pragma: no cover
        msg = f"{distribution.capitalize()} has no fit attribute."
        logger.error(msg)
        raise e


# ------------------------------------------------------------------------------------------------ #
#                               DISTRIBUTION GENERATOR                                             #
# ------------------------------------------------------------------------------------------------ #
class Generator:
    """Random variable generator for various distributions. Parameters estimated from data.

    The parameters for the specified distribution will be estimated from the data provided.
    It returns an array of the designated distribution, the parameters estimated from
    the data provided, of length matching provided data.

    This is used by goodness of fit tests to evaluate the degree to which a distribution
    matches an hypothesized distribution.

    Args:
        data (pd.DataFrame): Data from which distribution parameters are estimated.
    """

    __GENERATORS = {
        "normal": normal,
        "chi2": chi2,
        "exponential": exponential,
        "gamma": gamma,
        "logistic": logistic,
        "uniform": uniform,
    }

    def __init__(self) -> None:
        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    def __call__(self, data: np.ndarray, distribution: str) -> np.ndarray:
        """Returns random values of the designated distribution

        Args:
            data (np.ndarray): The data from which the distribution parameters are estimated
            distribution (str): One of the supported distributions. See the README.
        """
        try:
            return self.__GENERATORS[distribution](data=data)
        except KeyError:  # pragma: no cover
            msg = f"{distribution} is not supported."
            self._logger.debug(msg)
            raise NotImplementedError(msg)
