""" statvar.py. Bus Reconstruction Applications (@) 2022
This module encapsulates all Parsl applications used in the reconstruction 
processes.
This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

from typing import List


class StatisticsVariable(object):
    def __init__(self) -> None:
        self.sum_value: List = list()
        self.first_momentum: List = list()
        self.second_momentum: List = list()
        self.length: List = list()
        self.description: List = list()
        self.size: int = 0

    def create_variable(self, description: str = None) -> int:
        self.sum_value.append(0.0)
        self.first_momentum.append(0.0)
        self.second_momentum.append(0.0)
        self.length.append(0)
        self.description.append(description)
        id = self.size
        self.size += 1
        return id

    def add_value(self, id: int, value: float) -> None:
        self.sum_value[id] += value
        delta = value - self.first_momentum[id]
        self.length[id] += 1
        self.first_momentum[id] += delta / self.length[id]
        delta2 = value - self.first_momentum[id]
        self.second_momentum[id] += delta * delta2
        return

    def bulk_add_value(self, id: int, value_list: list) -> None:
        for i in value_list:
            self.add_value(id, i)
        return

    def sum(self, id: int) -> float:
        return self.sum_value[id]

    def mean(self, id: int) -> float:
        return self.first_momentum[id]

    def variance(self, id: int) -> float:
        return self.second_momentum[id]

    def length(self, id: int) -> int:
        return self.length[id]

    def size(self, id: int) -> int:
        return self.size

    def dump(self, file_name: str) -> None:
        with open(file_name, "w") as fd:
            fd.write("ID,MEAN,VARIANCE,NUMOBS,DESCRIPTION\n")
            for i, desc in enumerate(self.description):
                m = self.first_momentum[i]
                v = self.second_momentum[i]
                l = self.length[i]
                fd.write(f"{i},{m},{v},{l},{desc}\n")
        return
