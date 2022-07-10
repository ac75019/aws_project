from copy import deepcopy
import time
from timeit import default_timer
from collections import OrderedDict
import datetime
from datetime import timedelta, datetime, date
from typing import Any, Union, List, Dict, Tuple
from random import random
from uuid import uuid4
from ulid import ulid
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# @dataclass()

#do the change outside the DataPooint

#@dataclass(frozen=True)
class DataPoints:
    __slots__ = ["__obs_dates", "__obs_levels", "__obs_size", "__hash", "___ulid",
                 "__original_time_series", "__final_time_series",
                 "__mean", "__std", "__variance", "__elasticity"
                 ]

    def __init__(self, obs_dates: List[datetime] = [], obs_levels: List[Union[int, float]] = [], obs_size: int = 0):

        if isinstance(obs_dates, list) and isinstance(obs_levels, list) and isinstance(obs_size, (int, float)):
            assert len(obs_dates) == len(obs_levels) == int(obs_size)
            self.__obs_dates = obs_dates
            self.__obs_levels = obs_levels
            self.__obs_size = int(obs_size)
            self.__hash = hash((tuple(self.__obs_dates), tuple(self.__obs_levels), self.__obs_size))
            self.___ulid = self.__hash
            self.__compute_statistics()
            #self.__original_time_series = self.__keeper()
        else:
            raise TypeError("please check")

    # def __keeper(self):
    #     self.__final_time_series = {"ts": None,
    #                                 "stats": None}
    #     #if self.___ulid == self.__hash:
    #     #cp_obs_dates = deepcopy(self.__obs_dates)
    #     #cp_obs_levels = deepcopy(self.__obs_levels)
    #     # else:
    #     cp_obs_dates = self.__obs_dates
    #     cp_obs_levels = self.__obs_levels
    #
    #     return {
    #         "ts": sorted(tuple([(t, l) for t, l in zip(cp_obs_dates, cp_obs_levels)]),
    #                      key=lambda x: x[0]),
    #         "stats": self.__compute_statistics()
    #     }

    # def add_points(self, obs_dates:List[datetime], obs_levels:List[Union[int, float]], obs_size:int =0 ):
    #     if isinstance(obs_dates, list) and isinstance(obs_levels, list):
    #         assert len(obs_dates) == len(obs_levels) == int(obs_size)
    #
    #     if obs_size <= 0:
    #         return
    #
    #     if not set(obs_dates).isdisjoint(set(self.__obs_dates)):
    #         raise ValueError("you can't change existing time series")
    #
    #     self.__obs_dates += obs_dates
    #     self.__obs_levels += obs_levels
    #     self.__obs_size += obs_size
    #     self.___ulid = ulid()
    #     self.__final_time_series = {"ts": self.__keeper(),
    #                                 "stats": self.__compute_statistics()
    #                                 }

    def __compute_statistics(self):
        if self.__obs_size > 0:
            self.__mean = np.nanmean(self.__obs_levels)
            self.__std = np.nanstd(self.__obs_levels)
            self.__variance = np.nanvar(self.__obs_levels)
            self.__elasticity = self.__mean / self.__std if not np.isclose(self.__std, 0) else 0
        else:
            self.__mean = np.NaN
            self.__std = np.NaN
            self.__variance = np.NaN
            self.__elasticity = np.NaN

        return {"mean": self.__mean,
                "std": self.__std,
                "variance": self.__variance,
                "elasticity": self.__elasticity
                }

    @property
    def mean(self):
        return self.__mean

    @property
    def std(self):
        return self.__std

    @property
    def variance(self):
        return self.__variance

    @property
    def elasticity(self):
        return self.__elasticity

    def __hash__(self):
        return hash((self.__obs_dates, self.__obs_levels, self.__obs_size))

    def __eq__(self, other):
        return self.__obs_dates == other.__obs_dates and self.__obs_levels == other.__obs_levels and self.__obs_size == other.__obs_size

    def __lt__(self, other):
        return self.__elasticity > other.__elasticity  # variance self lower than variance other

    def __le__(self, other):
        return self < other or self == other

    def __gt__(self, other):
        return not self <= other

    def __ge__(self, other):
        return not self < other

    def plot(self):
        plt.plot(self.__obs_dates, self.__obs_levels)

    @property
    def obs_dates(self):
        return deepcopy(self.__obs_dates)

    @property
    def obs_levels(self):
        return deepcopy(self.__obs_levels)

    @property
    def obs_size(self):
        return self.__obs_size

    @property
    def hash(self):
        return self.__hash

    @property
    def ulid(self):
        return self.___ulid

    # @property
    # def original_time_series(self):
    #     return deepcopy(self.__original_time_series)


if __name__ == "__main__":
    N2 = 500
    t2 = [datetime(2008, 12, 30) + timedelta(seconds=-i) for i in range(N2)]
    s21 = [random() for i in range(N2)]


    N = 5000000
    t = [datetime(2021, 12, 30) + timedelta(seconds=-i) for i in range(N)]
    s1 = [random() for i in range(N)]
    s2 = [random() for i in range(N)]

    st = default_timer()
    data_point1 = DataPoints(obs_dates=t, obs_levels=s1, obs_size=len(s1))
    et = default_timer() - st
    print(et)

    st = default_timer()
    data_point2 = DataPoints(obs_dates=t, obs_levels=s2, obs_size=len(s2))
    et = default_timer() - st
    print(et)

    # print(DataPoints(obs_dates=t, obs_levels=s, obs_size=len(s)) in set([data_point1, data_point2]))

    print(data_point1 <= data_point2)
    print(data_point1.variance, data_point2.variance)

    #data_point1.add_points(obs_dates=t2, obs_levels=s21,obs_size=len(s21))
    #print(data_point1.variance, data_point2.variance)

