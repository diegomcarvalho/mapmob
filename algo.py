from collections import namedtuple
from abc import ABC, abstractmethod
from typing import List


class Algorithm(ABC):
    """A Algorithm metaclass"""

    def __init__(self, name: str, columns: List) -> None:
        self._name = name
        self._columns = columns

    @property
    def name(self):
        return self._name

    @property
    def columns(self):
        return self._columns

    def ret_type(self):
        return namedtuple("DataRet", self.columns)

    @abstractmethod
    def visit_data_frame(cls, tag, df):
        pass


class AlgorithmFactoryClass:
    known_algo = dict()

    def register(self, name: str, cls: Algorithm):
        self.known_algo[name] = cls
        return

    def get_algorithm(self, name: str) -> Algorithm:
        return self.known_algo[name](name)


class Algorithm_01(Algorithm):
    columns = [
        "TAG",
        "NUM_OBS",
        "VELOCITY",
        "SPEED",
        "DISTANCE",
        "INTERVAL",
        "NUM_BUS",
        "CO",
        "CO_2",
        "NO_x",
        "HC",
    ]

    def __init__(self, name) -> None:
        super().__init__(name, self.columns)

    def visit_data_frame(self, tag, df):
        import numpy as np

        tag = tag.replace("G1-", "")
        num_obs = len(df)
        velocity = df.VELOCITY[df["VELOCITY"] != 0].mean()
        speed = df["SPEED"].mean()
        distance = df["DISTANCE"].mean()
        interval = df["INTERVAL"].mean() / np.timedelta64(1, "s")
        num_bus = len(df["BUSID"].unique())
        co = sum(((df["INTERVAL"] / np.timedelta64(1, "s")) * df["CO"]).dropna())
        co2 = sum(((df["INTERVAL"] / np.timedelta64(1, "s")) * df["CO_2"]).dropna())
        nox = sum(((df["INTERVAL"] / np.timedelta64(1, "s")) * df["NO_x"]).dropna())
        hc = sum(((df["INTERVAL"] / np.timedelta64(1, "s")) * df["HC"]).dropna())

        ret_value = self.ret_type()._make(
            (
                tag,
                num_obs,
                velocity,
                speed,
                distance,
                interval,
                num_bus,
                co,
                co2,
                nox,
                hc,
            )
        )

        return ret_value


AlgorithmFactory = AlgorithmFactoryClass()
AlgorithmFactory.register("algo01", Algorithm_01)
