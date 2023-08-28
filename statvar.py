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


from typing import List

class StatisticsVariable:
    def __init__(self) -> None:
        self.variables = []
        self.size = 0

    def create_variable(self, description: str = None) -> int:
        """
        Create a new variable and return its ID.
        """
        self.variables.append({
            'sum_value': 0.0,
            'first_momentum': 0.0,
            'second_momentum': 0.0,
            'length': 0,
            'description': description
        })
        variable_id = self.size
        self.size += 1
        return variable_id

    def add_value(self, id: int, value: float) -> None:
        """
        Add a value to the variable with the given ID.
        """
        variable = self.variables[id]
        variable['sum_value'] += value
        delta = value - variable['first_momentum']
        variable['length'] += 1
        variable['first_momentum'] += delta / variable['length']
        delta2 = value - variable['first_momentum']
        variable['second_momentum'] += delta * delta2

    def bulk_add_value(self, id: int, value_list: List[float]) -> None:
        """
        Add a list of values to the variable with the given ID.
        """
        for value in value_list:
            self.add_value(id, value)

    def sum(self, id: int) -> float:
        """
        Return the sum of values for the variable with the given ID.
        """
        return self.variables[id]['sum_value']

    def mean(self, id: int) -> float:
        """
        Return the mean of values for the variable with the given ID.
        """
        return self.variables[id]['first_momentum']

    def variance(self, id: int) -> float:
        """
        Return the variance of values for the variable with the given ID.
        """
        return self.variables[id]['second_momentum']

    def length(self, id: int) -> int:
        """
        Return the number of observations for the variable with the given ID.
        """
        return self.variables[id]['length']

    def size(self) -> int:
        """
        Return the total number of variables.
        """
        return self.size

    def dump(self, file_name: str) -> None:
        """
        Dump the variables' statistics to a file.
        """
        with open(file_name, "w") as fd:
            fd.write("ID,MEAN,VARIANCE,NUMOBS,DESCRIPTION\n")
            for i, variable in enumerate(self.variables):
                m = variable['first_momentum']
                v = variable['second_momentum']
                l = variable['length']
                desc = variable['description']
                fd.write(f"{i},{m},{v},{l},{desc}\n")
