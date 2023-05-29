#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Explorer                                                                            #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.11                                                                             #
# Filename   : /explorer/service/identity.py                                                       #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/explorer                                           #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday May 28th 2023 03:00:41 pm                                                    #
# Modified   : Sunday May 28th 2023 10:08:57 pm                                                    #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Id Generation Module"""
from __future__ import annotations
from abc import ABC
import logging
import shelve
import random
import uuid


# ------------------------------------------------------------------------------------------------ #
class BaseIDGen(ABC):
    """Base class for id generators."""

    def __iter__(self) -> RandomIDGen:  # pragma: no cover
        """Initializes the generator"""

    def __next__(self):
        """Supplies next id"""


# ------------------------------------------------------------------------------------------------ #
#                                            RANDOM IDGEN                                          #
# ------------------------------------------------------------------------------------------------ #
class RandomIDGen(BaseIDGen):
    """Generates ids of n digits

    Args:
        n (int): The length of the id to generate.
    """

    def __init__(self, filepath: str, size: int = 4) -> None:
        self._n = 0
        self._filepath = filepath
        self._size = size
        self._key = "idlist"
        self._idlist = []
        self._max = 10**size
        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    def __iter__(self) -> RandomIDGen:  # pragma: no cover
        self.save()
        return self

    def __next__(self):
        if self._n > self._max:  # pragma: no cover
            raise StopIteration

        self.load()
        while True:
            id = str(random.randint(0, self._max))
            id = id.zfill(self._size)
            if id not in self._idlist:
                self._idlist.append(id)
                self.save()
                self._n += 1
                return id

    @property
    def idlist(self) -> list:
        return self._idlist

    def reset(self) -> None:
        """Resets by deleting the idlist."""
        msg = "This may result in duplicate ids. Are you sure [y/n]?"
        go = input(msg)
        if "y" in go:
            self.delete()
            self._idlist = []
            self.save()

    def load(self, ignore_errors: bool = False) -> None:
        """Loads the idlist from file"""
        try:
            with shelve.open(self._filepath) as db:
                self._idlist = db[self._key]
        except KeyError as e:  # pragma: no cover
            msg = "Id List not found."
            self._logger.error(msg)
            if not ignore_errors:
                raise e

    def save(self, ignore_errors: bool = False) -> None:
        try:
            with shelve.open(self._filepath) as db:
                db[self._key] = self._idlist
        except Exception as e:  # pragma: no cover
            msg = "Exception occurred while saving Id List."
            self._logger.error(msg)
            if not ignore_errors:
                raise e

    def exists(self) -> bool:
        """Checks existence of Id List"""
        with shelve.open(self._filepath) as db:
            return self._key in db

    def delete(self, ignore_errors: bool = True) -> None:
        """Deletes the key from the repository."""
        try:
            with shelve.open(self._filepath) as db:
                del db[self._key]
        except KeyError as e:  # pragma: no cover
            msg = "ID List was not found in the repository."
            self._logger.error(msg)
            if not ignore_errors:
                raise e


# ------------------------------------------------------------------------------------------------ #
#                                            UNIQUE IDGEN                                          #
# ------------------------------------------------------------------------------------------------ #
class UniqueIDGen(BaseIDGen):
    """Globally unique id generator based upon the UUID4 Stand"""

    def __iter__(self) -> UniqueIDGen:
        return self

    def __next__(self):
        return str(uuid.uuid4())
