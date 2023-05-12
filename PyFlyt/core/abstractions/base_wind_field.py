"""Defines a basic wind field class to inherit from when implementing custom wind field models."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class WindFieldClass(ABC):
    """Basic WindField class to implement custom wind field models.

    Example Usage:
        >>> from PyFlyt.core import Aviary
        >>> from PyFlyt.core.abstractions import WindFieldClass
        >>>
        >>> # define the wind field
        >>> class MyWindField(WindFieldClass):
        >>>     def __init__(self, my_parameter=1.0, np_random: None | np.random.RandomState = None):
        >>>         super().__init__(np_random)
        >>>         self.strength = my_parameter
        >>>
        >>>     def __call__(self, time: float, position: np.ndarray):
        >>>         # simulate a thermal windfield, where the xy velocities are 0,
        >>>         wind = np.zeros_like(position)
        >>>
        >>>         # but the z velocity varies to the log of height,
        >>>         wind[:, -1] = np.log(position[:, -1]) * self.strength
        >>>
        >>>         # plus some noise,
        >>>         wind += self.np_random.randn(*wind.shape)
        >>>
        >>>         return wind
        >>>
        >>> # environment setup, attach the windfield, override the parameter `my_parameter` from the default
        >>> env = Aviary(..., wind_type=MyWindField, wind_options=dict(my_parameter=1.2))
        >>>
        >>> # step as usual
        >>> ...
    """

    def __init__(self, np_random: None | np.random.RandomState = None):
        """Initializes the wind_field."""
        self.np_random = np.random.RandomState() if np_random is None else np_random

    @abstractmethod
    def __call__(self, time: float, position: np.ndarray) -> np.ndarray:
        """When given the time float and a position as an (n, 3) array, must return a (n, 3) array representing the local wind velocity.

        Args:
            time (float): float representing the timestep of the simulation in seconds.
            position (np.ndarray): (n, 3) array representing a series of n positions to sample wind velocites.
        """
        pass

    @staticmethod
    def _check_wind_field_validity(wind_field):
        test_velocity = wind_field(0.0, np.array([[0.0, 0.0, 1.0]] * 5))
        assert isinstance(
            test_velocity, np.ndarray
        ), f"Returned wind velocity must be a np.ndarray, got {type(test_velocity)}."
        assert np.issubdtype(
            test_velocity.dtype, np.floating
        ), f"Returned wind velocity must be type float, got {test_velocity.dtype}."
        assert test_velocity.shape == (
            5,
            3,
        ), f"Returned wind velocity must be array of shape (n, 3), got (n+({test_velocity.shape[0] - 5}), {test_velocity.shape[1:]})."
