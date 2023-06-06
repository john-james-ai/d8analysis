#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Enter Project Name in Workspace Settings                                            #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.11                                                                             #
# Filename   : /explorer/stats/distribution.py                                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : Enter URL in Workspace Settings                                                     #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday May 27th 2023 08:56:02 pm                                                  #
# Modified   : Tuesday June 6th 2023 04:56:32 am                                                   #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Statistics Module"""
from __future__ import annotations
import logging

from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from explorer.visual.config import Canvas

logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
#                                    SCIPY DISTRIBUTIONS                                           #
# ------------------------------------------------------------------------------------------------ #
DISTRIBUTIONS = {
    "beta": stats.beta,
    "normal": stats.norm,
    "chi2": stats.chi2,
    "exponential": stats.expon,
    "f": stats.f,
    "gamma": stats.gamma,
    "logistic": stats.logistic,
    "lognorm": stats.lognorm,
    "pareto": stats.pareto,
    "uniform": stats.uniform,
    "weibull_min": stats.weibull_min,
}


# ------------------------------------------------------------------------------------------------ #
#                              RANDOM DATA GENERATORS                                              #
# ------------------------------------------------------------------------------------------------ #
def beta(data: np.ndarray) -> np.ndarray:
    """Generates random variates for the beta distribution

    Args:
        data (np.ndarray): 1D Numpy array of data from which parameters will be estimated.

    Returns:
        rvs: Random variate of the distribution
        pdf: Data from the probability density function
    """
    a, b, loc, scale = get_params(data=data, distribution="beta")
    rvs = stats.beta.rvs(a, b, loc=loc, scale=scale, size=len(data))
    pdf = (
        np.linspace(
            stats.beta.ppf(0.01, a, b, loc, scale),
            stats.beta.ppf(0.99, a, b, loc, scale),
            num=len(data),
        ),
        stats.beta.pdf(x=data, a=a, b=b, loc=loc, scale=scale),
    )
    return rvs, pdf


def normal(data: np.ndarray) -> np.ndarray:
    """Generates random variates for the normal distribution

    Args:
        data (np.ndarray): 1D Numpy array of data from which parameters will be estimated.

    Returns:
        rvs: Random variate of the distribution
        pdf: Data from the probability density function
    """
    loc, scale = get_params(data=data, distribution="normal")
    rvs = stats.norm.rvs(loc=loc, scale=scale, size=len(data))
    pdf = (
        np.linspace(
            stats.norm.ppf(0.01, loc, scale), stats.norm.ppf(0.99, loc, scale), num=len(data)
        ),
        stats.norm.pdf(x=data, loc=loc, scale=scale),
    )
    return rvs, pdf


def chi2(data: np.ndarray) -> np.ndarray:
    """Generates random variates for the chi-squared distribution

    Args:
        data (np.ndarray): 1D Numpy array of data from which parameters will be estimated.

    Returns:
        rvs: Random variate of the distribution
        pdf: Data from the probability density function
    """
    a, loc, scale = get_params(data=data, distribution="chi2")
    df = 2 * a
    rvs = stats.chi2.rvs(df=df, loc=loc, scale=scale, size=len(data))
    pdf = (
        np.linspace(
            stats.chi2.ppf(0.01, df, loc, scale),
            stats.chi2.ppf(0.99, df, loc, scale),
            num=len(data),
        ),
        stats.chi2.pdf(x=data, df=df, loc=loc, scale=scale),
    )

    return rvs, pdf


def exponential(data: np.ndarray) -> np.ndarray:
    """Generates random variates for the exponential distribution

    Args:
        data (np.ndarray): 1D Numpy array of data from which parameters will be estimated.

    Returns:
        rvs: Random variate of the distribution
        pdf: Data from the probability density function
    """
    loc, scale = get_params(data=data, distribution="exponential")
    rvs = stats.expon.rvs(loc=loc, scale=scale, size=len(data))
    pdf = (
        np.linspace(
            stats.expon.ppf(0.01, loc, scale),
            stats.expon.ppf(0.99, loc, scale),
            num=len(data),
        ),
        stats.expon.pdf(x=data, loc=loc, scale=scale),
    )
    return rvs, pdf


def f(data: np.ndarray) -> np.ndarray:
    """Generates random variates for the f distribution

    Args:
        data (np.ndarray): 1D Numpy array of data from which parameters will be estimated.

    Returns:
        rvs: Random variate of the distribution
        pdf: Data from the probability density function
    """
    dfn, dfd, loc, scale = get_params(data=data, distribution="f")
    rvs = stats.f.rvs(dfn=dfn, dfd=dfd, size=len(data))
    pdf = (
        np.linspace(
            stats.f.ppf(0.01, dfn, dfd, loc, scale),
            stats.f.ppf(0.99, dfn, dfd, loc, scale),
            num=len(data),
        ),
        stats.f.pdf(x=data, dfn=dfn, dfd=dfd, loc=loc, scale=scale),
    )
    return rvs, pdf


def gamma(data: np.ndarray) -> np.ndarray:
    """Generates random variates for the gamma distribution

    Args:
        data (np.ndarray): 1D Numpy array of data from which parameters will be estimated.

    Returns:
        rvs: Random variate of the distribution
        pdf: Data from the probability density function
    """
    a, loc, scale = get_params(data=data, distribution="gamma")
    rvs = stats.gamma.rvs(a=a, loc=loc, scale=scale, size=len(data))
    pdf = (
        np.linspace(
            stats.gamma.ppf(0.01, a, loc, scale),
            stats.gamma.ppf(0.99, a, loc, scale),
            num=len(data),
        ),
        stats.gamma.pdf(x=data, a=a, loc=loc, scale=scale),
    )
    return rvs, pdf


def logistic(data: np.ndarray) -> np.ndarray:
    """Generates random variates for the logistic distribution

    Args:
        data (np.ndarray): 1D Numpy array of data from which parameters will be estimated.

    Returns:
        rvs: Random variate of the distribution
        pdf: Data from the probability density function
    """
    loc, scale = get_params(data=data, distribution="logistic")
    rvs = stats.logistic.rvs(loc=loc, scale=scale, size=len(data))
    pdf = (
        np.linspace(
            stats.logistic.ppf(0.01, loc, scale),
            stats.logistic.ppf(0.99, loc, scale),
            num=len(data),
        ),
        stats.logistic.pdf(x=data, loc=loc, scale=scale),
    )
    return rvs, pdf


def lognorm(data: np.ndarray) -> np.ndarray:
    """Generates random variates for the log normal distribution

    Args:
        data (np.ndarray): 1D Numpy array of data from which parameters will be estimated.

    Returns:
        rvs: Random variate of the distribution
        pdf: Data from the probability density function
    """
    s, loc, scale = get_params(data=data, distribution="lognorm")
    rvs = stats.lognorm.rvs(s=s, loc=loc, scale=scale, size=len(data))
    pdf = (
        np.linspace(
            stats.lognorm.ppf(0.01, s, loc, scale),
            stats.lognorm.ppf(0.99, s, loc, scale),
            num=len(data),
        ),
        stats.lognorm.pdf(x=data, s=s, loc=loc, scale=scale),
    )
    return rvs, pdf


def pareto(data: np.ndarray) -> np.ndarray:
    """Generates random variates for the pareto distribution

    Args:
        data (np.ndarray): 1D Numpy array of data from which parameters will be estimated.

    Returns:
        rvs: Random variate of the distribution
        pdf: Data from the probability density function
    """
    b, loc, scale = get_params(data=data, distribution="pareto")
    rvs = stats.pareto.rvs(b=b, loc=loc, scale=scale, size=len(data))
    pdf = (
        np.linspace(
            stats.pareto.ppf(0.01, b, loc, scale),
            stats.pareto.ppf(0.99, b, loc, scale),
            num=len(data),
        ),
        stats.pareto.pdf(x=data, b=b, loc=loc, scale=scale),
    )
    return rvs, pdf


def uniform(data: np.ndarray) -> np.ndarray:
    """Generates random variates for the uniform distribution

    Args:
        data (np.ndarray): 1D Numpy array of data from which parameters will be estimated.

    Returns:
        rvs: Random variate of the distribution
        pdf: Data from the probability density function
    """
    loc, scale = get_params(data=data, distribution="uniform")
    rvs = stats.uniform.rvs(loc=loc, scale=scale, size=len(data))
    pdf = (
        np.linspace(
            stats.uniform.ppf(0.01, loc, scale),
            stats.uniform.ppf(0.99, loc, scale),
            num=len(data),
        ),
        stats.uniform.pdf(x=data, loc=loc, scale=scale),
    )
    return rvs, pdf


def weibull(data: np.ndarray) -> np.ndarray:
    """Generates random variates for the uniform distribution

    Args:
        data (np.ndarray): 1D Numpy array of data from which parameters will be estimated.

    Returns:
        rvs: Random variate of the distribution
        pdf: Data from the probability density function
    """
    c, loc, scale = get_params(data=data, distribution="weibull_min")

    rvs = stats.weibull_min.rvs(c=c, loc=loc, scale=scale, size=len(data))
    pdf = (
        np.linspace(
            stats.weibull_min.ppf(0.01, c, loc, scale),
            stats.weibull_min.ppf(0.99, c, loc, scale),
            num=len(data),
        ),
        stats.weibull_min.pdf(x=data, c=c, loc=loc, scale=scale),
    )
    return rvs, pdf


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
class RVSDistribution:
    """Random variates for various distributions. Parameters estimated from data.

    The parameters for the specified distribution will be estimated from the data provided.
    It returns an array of the designated distribution, the parameters estimated from
    the data provided, of length matching provided data.

    This is used by goodness of fit tests to evaluate the degree to which a distribution
    matches an hypothesized distribution.

    Args:
        data (pd.DataFrame): Data from which distribution parameters are estimated.
    """

    __GENERATORS = {
        "beta": beta,
        "normal": normal,
        "chi2": chi2,
        "exponential": exponential,
        "f": f,
        "gamma": gamma,
        "logistic": logistic,
        "lognorm": lognorm,
        "uniform": uniform,
        "weibull_min": weibull,
        "pareto": pareto,
    }

    def __init__(self) -> None:
        self._logger = logging.getLogger(f"{self.__class__.__name__}")
        self._rvs = None
        self._pdf = None
        self._distribution = None

    @property
    def data(self) -> np.ndarray:
        return self._data

    def __call__(self, data: np.ndarray, distribution: str) -> np.ndarray:
        """Returns random values of the designated distribution

        Args:
            data (np.ndarray): The data from which the distribution parameters are estimated
            distribution (str): One of the supported distributions. See the README.
        """
        self._data = data
        self._distribution = distribution

        try:
            self._rvs, self._pdf = self.__GENERATORS[distribution](data=data)
        except KeyError:  # pragma: no cover
            msg = f"{distribution} is not supported."
            self._logger.debug(msg)
            raise NotImplementedError(msg)

    def plot(self, ax: plt.Axes = None) -> plt.Axes:
        """Plots the data, a random sample, and the probability density function."""

        ax = ax or Canvas().ax
        # Combine the data and random sample into a single dataframe
        data = {"Source": "Data", "Values": self._data}
        sample = {"Source": "Random Variate", "Values": self._rvs}
        # pdf = {"Source": "Probability Density Function", "Values": self._pdf[1]}
        df1 = pd.DataFrame(data=data)
        df2 = pd.DataFrame(data=sample)
        # df3 = pd.DataFrame(data=pdf)
        df4 = pd.concat([df1, df2], axis=0)
        # Plot data and random sample
        ax = sns.kdeplot(data=df4, x="Values", hue="Source", palette=Canvas.palette, ax=ax)
        # Plot pdf
        # ax = sns.lineplot(x=self._pdf[0], y=self._pdf[1], palette=Canvas.palette, ax=ax)

        title = f"{self._distribution.capitalize()} Distribution\nData, Random Variate & Probability Density Function"

        ax.set_title(title)
        return ax
