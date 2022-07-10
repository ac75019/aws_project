from copy import deepcopy
import time
from timeit import default_timer
from collections import OrderedDict
import datetime
from datetime import timedelta, datetime, date
from typing import Any, Union, List, Dict, Tuple, Optional
from random import random
from uuid import uuid4
from ulid import ulid
from dataclasses import dataclass, field, InitVar, fields
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from functools import reduce
from operator import iconcat


# do the change of the TS outside the DataPoint
@dataclass(frozen=True, eq=False)
class DataPoints:
    # __slots__ = ['obs_dates','obs_levels','obs_size']
    obs_dates: List[datetime]
    obs_levels: List[Union[float, int]] = field(compare=False)
    obs_size: int

    ulid = ulid()
    current_reference_date: datetime = field(init=False, default_factory=lambda: datetime.utcnow().isoformat())
    hash: int = field(init=False)
    time_series: Dict[str, Any] = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "hash",
                           hash((tuple(self.obs_dates), tuple(self.obs_levels), self.obs_size, self.ulid)))
        object.__setattr__(self, "time_series", self.__sorted_time_series())
        # object.__setattr__(self, "current_reference_date","")

    def __sorted_time_series(self):
        return {"time_series": sorted(tuple([(t, l) for t, l in zip(self.obs_dates, self.obs_levels)]),
                                      key=lambda x: x[0]), "basic_statistic": self.__compute_statistics()
                }

    def __compute_statistics(self):
        if self.obs_size > 0:
            mean = np.nanmean(self.obs_levels)
            std = np.nanstd(self.obs_levels)
            variance = np.nanvar(self.obs_levels)
            elasticity = mean / std if not np.isclose(std, 0) else 0
        else:
            mean = np.NaN
            std = np.NaN
            variance = np.NaN
            elasticity = np.NaN

        return {"mean": mean,
                "std": std,
                "variance": variance,
                "elasticity": elasticity
                }

    @property
    def mean(self):
        return self.time_series["basic_statistic"]["mean"]

    @property
    def std(self):
        return self.time_series["basic_statistic"]["std"]

    @property
    def variance(self):
        return self.time_series["basic_statistic"]["variance"]

    @property
    def elasticity(self):
        return self.time_series["basic_statistic"]["elasticity"]

    def __hash__(self):
        return hash((tuple(self.obs_dates), tuple(self.obs_levels), self.obs_size, self.ulid))

    def __eq__(self, other):
        return self.obs_dates == other.obs_dates and self.obs_levels == other.obs_levels and self.obs_size == other.obs_size

    def __lt__(self, other):
        return self.time_series["basic_statistic"]["elasticity"] > other.time_series["basic_statistic"][
            "elasticity"]  # variance self lower than variance other

    def __le__(self, other):
        return self < other or self == other

    def __gt__(self, other):
        return not self <= other

    def __ge__(self, other):
        return not self < other

    def plot(self):
        plt.plot(self.obs_dates, self.obs_levels)
        plt.show()

    # @property - can't do that


@dataclass(init=False)
class Config:
    VAR_NAME_1: str
    VAR_NAME_2: str

    def __init__(self, **kwargs):
        names = set([f.name for f in fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)


@dataclass(frozen=True, eq=True, order=True)
class CDSIndexForward:
    reference_data: str = field(init=False, compare=False)
    expiry_tenor_date: int = field(compare=True)

    data_point: DataPoints = field(compare=False)
    config: InitVar[Config] = field(compare=False)

    current_reference_date: datetime = field(init=False, compare=True,
                                             default_factory=lambda: datetime.utcnow().isoformat())

    def __post_init__(self, config: Config):
        ref_data = config
        object.__setattr__(self, "reference_data", ref_data)


@dataclass(frozen=True, eq=False)
class CDSIndexForwardCurve:
    expiry_tenor_dates: List[str] = field(init=False, compare=False, default_factory=list)
    curve: Optional[Dict[str, CDSIndexForward]] = field(compare=False, default_factory=dict)
    current_reference_date: datetime = datetime.utcnow().isoformat()

    def add_forward_point(self, cds_forward: CDSIndexForward):
        # add checks
        self.expiry_tenor_dates.append(cds_forward.expiry_tenor_date)
        self.curve[cds_forward.expiry_tenor_date] = cds_forward

        object.__setattr__(self, "curve", OrderedDict(sorted(self.curve.items(), key=lambda x: int(x[0][:-1]))))
        object.__setattr__(self, "expiry_tenor_dates", sorted(self.expiry_tenor_dates, key=lambda x: int(x[:-1])))

    def remove_forward_point(self, id: Union[int, str]):
        del self.curve[id]


@dataclass(frozen=True, eq=True, order=True)
class CDSIndexImpliedVol:
    reference_data: str = field(init=False, compare=False)

    expiry_tenor_date: str = field(compare=True)
    strike: int = field(compare=True)

    data_point: DataPoints = field(compare=False)
    config: InitVar[Config] = field(compare=False)

    current_reference_date: datetime = field(init=False, compare=True,
                                             default_factory=lambda: datetime.utcnow().isoformat())

    def __post_init__(self, config: Config):
        ref_data = config
        object.__setattr__(self, "reference_data", ref_data)


@dataclass(frozen=True, eq=False)
class CDSIndexImpliedVolSkew:
    expiry_tenor_dates: List[str] = field(init=False, compare=False, default_factory=list)
    strikes: List[str] = field(compare=False, default_factory=list)

    curve: Optional[Dict[Tuple[str, int], CDSIndexImpliedVol]] = field(compare=False, default_factory=dict)

    current_reference_date: datetime = datetime.utcnow().isoformat()

    def add_skew_point(self, implied_vol: CDSIndexImpliedVol):
        # add check ref data and ref dates

        if len(self.expiry_tenor_dates) == 0:
            self.expiry_tenor_dates.append(implied_vol.expiry_tenor_date)

        elif len(self.expiry_tenor_dates) == 1 and self.expiry_tenor_dates[0] == implied_vol.expiry_tenor_date:
            if implied_vol.strike not in self.strikes:
                self.strikes.append(implied_vol.strike)
                self.curve[tuple([implied_vol.expiry_tenor_date, implied_vol.strike])] = implied_vol

                object.__setattr__(self, "curve", OrderedDict(sorted(self.curve.items(), key=lambda x: int(x[0][1]))))
                object.__setattr__(self, "strikes", sorted(self.strikes, key=lambda x: int(x)))
            else:
                print("already in ")
        else:
            raise ValueError(" ")

    def remove_skew_point(self, id: Tuple[str, int]):
        del self.curve[id]


@dataclass(frozen=True, eq=False)
class CDSIndexImpliedVolTermStructure:
    expiry_tenor_dates: List[str] = field(init=False, compare=False, default_factory=list)
    strikes: List[str] = field(compare=False, default_factory=list)

    curve: Optional[Dict[Tuple[str, int], CDSIndexImpliedVol]] = field(compare=False, default_factory=dict)
    current_reference_date: datetime = datetime.utcnow().isoformat()

    def add_term_structure_point(self, implied_vol: CDSIndexImpliedVol):
        # add check ref data and ref dates

        if len(self.strikes) == 0:
            self.strikes.append(implied_vol.strike)

        elif len(self.strikes) == 1 and self.strikes[0] == implied_vol.strike:
            if implied_vol.expiry_tenor_date not in self.expiry_tenor_dates:
                self.expiry_tenor_dates.append(implied_vol.expiry_tenor_date)

                self.curve[tuple([implied_vol.expiry_tenor_date, implied_vol.strike])] = implied_vol

                object.__setattr__(self, "curve",
                                   OrderedDict(sorted(self.curve.items(), key=lambda x: int(x[0][0][:-1]))))
                object.__setattr__(self, "expiry_tenor_dates",
                                   sorted(self.expiry_tenor_dates, key=lambda x: int(x[:-1])))
            else:
                print("already in ")
        else:
            raise ValueError(" ")

    def remove_skew_point(self, id: Tuple[str, int]):
        del self.curve[id]


@dataclass(frozen=True, eq=False)
class CDSIndexImpliedVolSurface:
    is_from_term_structure: List[str] = field(init=False, compare=False, default_factory=list)
    is_from_skew: List[str] = field(init=False, compare=False, default_factory=list)
    is_from_implied_vol: List[str] = field(init=False, compare=False, default_factory=list)

    expiry_tenor_dates_and_strikes: List[Tuple[str, int]] = field(init=False, compare=False, default_factory=list)

    surface: Optional[Dict[
        Tuple[str, int], Union[CDSIndexImpliedVol, CDSIndexImpliedVolTermStructure, CDSIndexImpliedVolSkew]]] = field(
        compare=False, default_factory=dict)
    current_reference_date: datetime = datetime.utcnow().isoformat()

    # def __post_init__(self):
    #     object.__setattr__(self, "is_from_term_structure", -1)

    def add_vol(self, curve: Union[CDSIndexImpliedVol, CDSIndexImpliedVolTermStructure, CDSIndexImpliedVolSkew]):
        # add check ref data and ref dates
        name = curve.__class__.__name__

        if len(self.is_from_term_structure) == 0:
            self.is_from_term_structure.append(name)  # == "CDSIndexImpliedVolTermStructure"
        elif len(self.is_from_term_structure) == 1 and name != self.is_from_term_structure[0]:
            raise ValueError(" ")

        if len(self.is_from_skew) == 0:
            self.is_from_skew.append(name)  # == "CDSIndexImpliedVolSkew"
        elif len(self.is_from_skew) == 1 and name != self.is_from_skew[0]:
            raise ValueError(" ")

        if len(self.is_from_implied_vol) == 0:
            self.is_from_implied_vol.append(name)  # == "CDSIndexImpliedVol"
        elif len(self.is_from_implied_vol) == 1 and name != self.is_from_implied_vol[0]:
            raise ValueError(" ")

        if name in ["CDSIndexImpliedVolTermStructure", "CDSIndexImpliedVolSkew"]:
            curve_expiry_tenor_dates = curve.expiry_tenor_dates
            curve_strikes = curve.strikes
        elif name == "CDSIndexImpliedVol":
            curve_expiry_tenor_dates = [curve.expiry_tenor_date]
            curve_strikes = [curve.strike]
        else:
            raise TypeError("")

        is_empty_vol_surface = len(self.expiry_tenor_dates_and_strikes) == 0

        is_disjoint_set = True
        for tenor, strike in zip(curve_expiry_tenor_dates, curve_strikes):
            test = (tenor, strike) not in self.expiry_tenor_dates_and_strikes
            is_disjoint_set &= test

        if is_empty_vol_surface or is_disjoint_set:
            for tenor, strike in zip(curve_expiry_tenor_dates, curve_strikes):
                self.expiry_tenor_dates_and_strikes.append((tenor, strike))
                self.surface[tuple([tenor, strike])] = curve

            object.__setattr__(self, "surface",
                               OrderedDict(sorted(self.surface.items(), key=lambda x: (int(x[0][0][:-1]), int(x[0][1]))
                                                  )
                                           )
                               )

            object.__setattr__(self, "expiry_tenor_dates_and_strikes",
                               sorted(self.expiry_tenor_dates_and_strikes, key=lambda x: (int(x[0][:-1]), int(x[1]))
                                      )
                               )


        else:
            raise


def test_forward():
    forward_curve = CDSIndexForwardCurve(curve=OrderedDict([]))

    for i, expiry_tenor_date in enumerate(tenors):
        reference_data = Config(**{"VAR_NAME_1": 1, "VAR_NAME_2": 2})  # use for TENOR
        st = default_timer()
        b_dates = [datetime(2021, 12, 30) + timedelta(seconds=-i) for i in range(N)]
        levels = [random() for i in range(N)]

        data_point = DataPoints(obs_dates=b_dates, obs_levels=levels, obs_size=N)

        et = default_timer() - st
        print("data creation", et)

        st = default_timer()

        forward_curve.add_forward_point(
            CDSIndexForward(config=reference_data, expiry_tenor_date=expiry_tenor_date, data_point=data_point))

        et = default_timer() - st
        print("forward creation", et)

    # curve.update_forward_curve(fwrd)
    print(forward_curve.expiry_tenor_dates)

    # print(curve.curves.1M)


if __name__ == "__main__":
    N = 50000000
    tenors = [f"{i}M" for i in [1, 100, 10, 5, 9, 0]]
    strikes = [f"{i}" for i in [90, 100, 110, 56, 115]]

    index_skew = CDSIndexImpliedVolSkew(curve=OrderedDict([]))
    index_term_structure = CDSIndexImpliedVolTermStructure(curve=OrderedDict([]))
    index_vol_surface = CDSIndexImpliedVolSurface(surface=OrderedDict())

    for i, expiry_tenor_date in enumerate(tenors):
        for j, strike in enumerate(strikes):
            reference_data = Config(**{"VAR_NAME_1": 1, "VAR_NAME_2": 2})  # use for TENOR
            st = default_timer()
            b_dates = [datetime(2021, 12, 30) + timedelta(seconds=-i) for i in range(N)]
            levels = [random() for i in range(N)]

            data_point = DataPoints(obs_dates=b_dates, obs_levels=levels, obs_size=N)
            



            et = default_timer() - st
            print("data creation", et)

            st = default_timer()

            index_skew.add_skew_point(
                CDSIndexImpliedVol(config=reference_data,
                                   expiry_tenor_date="5M",
                                   strike=strike,
                                   data_point=data_point))

            et = default_timer() - st
            print("index_skew creation", et)
            st = default_timer()
            index_term_structure.add_term_structure_point(
                CDSIndexImpliedVol(config=reference_data,
                                   expiry_tenor_date=expiry_tenor_date,
                                   strike=100,
                                   data_point=data_point))
            et = default_timer() - st
            print("index_term_structure creation", et)
            st = default_timer()

            index_vol_surface.add_vol(CDSIndexImpliedVol(config=reference_data,
                                                         expiry_tenor_date=expiry_tenor_date,
                                                         strike=strike,
                                                         data_point=data_point))

            et = default_timer() - st
            print("index_vol_surface creation", et)
            st = default_timer()
            #print("expiry_tenor_dates_and_strikes: ", index_vol_surface.surface[('1M', '90')])
            #i = 0

