from collections import namedtuple
from abc import ABC, abstractmethod
from typing import List
import numpy as np
from datetime import datetime


class Algorithm(ABC):
    """An abstract base class for algorithms."""

    def __init__(self, name: str, columns: List[str]) -> None:
        self._name = name
        self._columns = columns

    @property
    def name(self):
        """Returns the name of the algorithm."""
        return self._name

    @property
    def columns(self):
        """Returns the list of columns."""
        return self._columns

    def ret_type(self):
        """Returns a named tuple type for the algorithm's columns."""
        return namedtuple("DataRet", self.columns)

    @abstractmethod
    def visit_data_frame(self, tag, df):
        """Abstract method to visit a data frame and perform algorithm-specific operations."""
        pass


class AlgorithmFactoryClass:
    known_algorithms = {}

    def register(self, name: str, cls: Algorithm):
        """Registers an algorithm class with a given name."""
        self.known_algorithms[name] = cls

    def get_algorithm(self, name: str) -> Algorithm:
        """Returns an instance of the specified algorithm."""
        return self.known_algorithms[name](name)

# Create the Algorithm Factory in order to be used...
AlgorithmFactory = AlgorithmFactoryClass()

#
# This is an example how to implement an statistics
# On the constructor, you need to append on the columns list the name of each column
# that will be present on the data frame.
# Then, for every select day in the data base, the visit_data_frame will be called to
# inspect the day's data frame and build each value for each column.
# In order to build the each line of the statistics data frame, you calculate each column's variable
# and use the function _make to create a line for the inspect data frame in the statistics data
#
class Algorithm_EX(Algorithm):
    """An example implementation of an algorithm."""

    columns = list()

    def __init__(self, name) -> None:
        
        # Check if this class has been already constructed
        if len(self.columns) != 0:
            return
        
        # The user can create columns name programmatically. 
        for i in ["TAG", "NUM_OBS", "VELOCITY", "SPEED", "DISTANCE", "INTERVAL", "NUM_BUS"]:
            self.columns.append(i)

        # Now calls the super class constructor
        super().__init__(name, self.columns)

        return

    def visit_data_frame(self, tag, df):
        """Visits a data frame and performs calculations for each column."""
        tag = tag.replace("G1-", "")
        num_obs = len(df)

        velocity = df["VELOCITY"].mean()
        speed = df["SPEED"].mean()
        distance = df["DISTANCE"].mean()
        interval = df["INTERVAL"].mean() / np.timedelta64(1, "s")
        num_bus = len(df["BUSID"].unique())

        ret_value = self.ret_type()._make(
            (
                tag,
                num_obs,
                velocity,
                speed,
                distance,
                interval,
                num_bus,
            )
        )

        return ret_value

AlgorithmFactory.register("algo_example", Algorithm_EX)

class Algorithm_01(Algorithm):
    columns = []
    def __init__(self, name) -> None:
        if len(self.columns) != 0:
            return
        for i in ["TAG", "NUM_OBS", "VELOCITY", "SPEED", "DISTANCE", "INTERVAL", "NUM_BUS", "CO", "CO_2", "NO_x", "HC"]:
            self.columns.append(i)

        for i in range(1,49):
            self.columns.append(f"VCORR_{i:02}")

        for i in range(1,49):
            self.columns.append(f"SCORR_{i:02}")

        super().__init__(name, self.columns)

    def visit_data_frame(self, tag, df):
        import numpy as np

        tag = tag.replace("G1-", "")
        num_obs = len(df)

        velocity = df["VELOCITY"].mean()
        speed = df["SPEED"].mean()
        distance = df["DISTANCE"].mean()
        interval = df["INTERVAL"].mean() / np.timedelta64(1, "s")
        num_bus = len(df["BUSID"].unique())
        co = sum(((df["INTERVAL"] / np.timedelta64(1, "s")) * df["CO"]).dropna())
        co2 = sum(((df["INTERVAL"] / np.timedelta64(1, "s")) * df["CO_2"]).dropna())
        nox = sum(((df["INTERVAL"] / np.timedelta64(1, "s")) * df["NO_x"]).dropna())
        hc = sum(((df["INTERVAL"] / np.timedelta64(1, "s")) * df["HC"]).dropna())

        result_list = []

        for i in range(1,49):
            t = df[df["CORREDOR"] == i].VELOCITY.mean()
            result_list.append(t)

        for i in range(1,49):
            t = df[df["CORREDOR"] == i].SPEED.mean()
            result_list.append(t)

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
                *result_list,
            )
        )

        return ret_value


class Algorithm_02(Algorithm):
    columns = [
        "TAG",  # Info do dia ex.2022-07-12
        "NUM_OBS",
        "VELOCITY01",
        "VELOCITY02",
        "VELOCITY03",
    ]

    def __init__(self, name) -> None:
        super().__init__(name, self.columns)

    def visit_data_frame(self, tag, df):
        import numpy as np

        num_obs = len(df)

        velocity01 = df.VELOCITY.mean()
        velocity02 = df.VELOCITY.mean()
        velocity03 = df.VELOCITY.mean()

        ret_value = self.ret_type()._make(
            (
                tag,
                num_obs,
                velocity01,
                velocity02,
                velocity03,
            )
        )

        return ret_value


class Algorithm_03(Algorithm):
    columns = [
        "TAG",  # Info do dia ex.2022-07-12
        "NUM_OBS",
    ]

    def __init__(self, name) -> None:

        for i in range(165):
            self.columns.append(f"BAIRRO{i:03}")

        super().__init__(name, self.columns)

    def visit_data_frame(self, tag, df):
        import numpy as np

        num_obs = len(df)

        result_list = []

        for i in range(165):
            result_list.append(df[df.CODBAIRRO == f"{i:03}"].VELOCITY.mean())

        ret_value = self.ret_type()._make(
            (
                tag,
                num_obs,
                *result_list
            )
        )

        return ret_value

class Algorithm_04(Algorithm):
    columns = []
    def __init__(self, name) -> None:
        if len(self.columns) == 0:
            for i in ["TAG", "NUM_OBS", "VELOCITY", "SPEED", "DISTANCE", "INTERVAL", "NUM_BUS", "CO", "CO_2", "NO_x", "HC"]:
                self.columns.append(i)

            for i in range(1,49):
                self.columns.append(f"VCORR_{i:02}")

            for i in range(1,49):
                self.columns.append(f"SCORR_{i:02}")

            for i in range(1,49):
                self.columns.append(f"SDVCORR_{i:02}")

        super().__init__(name, self.columns)

    def visit_data_frame(self, tag, df):
        import numpy as np

        tag = tag.replace("G1-", "")

        d  = datetime.strptime(tag, '%Y-%m-%d')
        dw = d.weekday()
        if dw in [0,4,5,6]:
            return None
        
        num_obs = len(df)
        velocity = df["VELOCITY"].mean()
        speed = df["SPEED"].mean()
        distance = df["DISTANCE"].mean()
        interval = df["INTERVAL"].mean() / np.timedelta64(1, "s")
        num_bus = len(df["BUSID"].unique())
        co = sum(((df["INTERVAL"] / np.timedelta64(1, "s")) * df["CO"]).dropna())
        co2 = sum(((df["INTERVAL"] / np.timedelta64(1, "s")) * df["CO_2"]).dropna())
        nox = sum(((df["INTERVAL"] / np.timedelta64(1, "s")) * df["NO_x"]).dropna())
        hc = sum(((df["INTERVAL"] / np.timedelta64(1, "s")) * df["HC"]).dropna())

        result_list = []

        for i in range(1,49):
            t = df[df["CORREDOR"] == i].VELOCITY.mean()
            result_list.append(t)

        for i in range(1,49):
            t = df[df["CORREDOR"] == i].SPEED.mean()
            result_list.append(t)

        for i in range(1,49):
            t = df[df["CORREDOR"] == i].SPEED.std()
            result_list.append(t)

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
                *result_list,
            )
        )

        return ret_value




AlgorithmFactory.register("algo01", Algorithm_01)
AlgorithmFactory.register("algo_velo", Algorithm_02)
AlgorithmFactory.register("algo_bairro", Algorithm_03)
AlgorithmFactory.register("algo_corredor", Algorithm_04)
