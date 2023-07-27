#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Enter Project Name in Workspace Settings                                            #
# Version    : 0.1.19                                                                              #
# Python     : 3.10.11                                                                             #
# Filename   : /edation/adapter/identity.py                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : Enter URL in Workspace Settings                                                     #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday June 23rd 2023 12:52:51 am                                                   #
# Modified   : Thursday July 27th 2023 08:48:18 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2023 John James                                                                 #
# ================================================================================================ #
"""Id Generation Module"""
from __future__ import annotations

import logging
import shelve
import random

from edation.application.ports.driven.identity import IDGenPort


# ------------------------------------------------------------------------------------------------ #
#                                            RANDOM IDGEN                                          #
# ------------------------------------------------------------------------------------------------ #
class IDGenerator(IDGenPort):
    """Generates ids of n digits

    Args:
        filepath (str): persistence location for used identifiers.
        n (int): The length of the id to generate.
    """

    __key = "idlist"

    def __init__(self, filepath: str, size: int = 5) -> None:
        self._n = 0
        self._filepath = filepath
        self._size = size
        self._key = self.__key
        self._idlist = []
        self._max = 10**size
        self._logger = logging.getLogger(f"{self.__class__.__name__}")

    def generate_id(self) -> str:
        """Generates the identifier."""
        self.load()
        while True:
            id = str(random.randint(0, self._max))
            id = id.zfill(self._size)
            if id not in self._idlist:
                self._idlist.append(id)
                self.save()
                self._n += 1
                return id

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
#                                            FAKE IDGEN                                            #
# ------------------------------------------------------------------------------------------------ #
class FakeIDGenerator(IDGenPort):
    """Returns the same identifier for unit testing."""

    def generate_id(self) -> str:
        return "1234"
