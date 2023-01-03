""" tools.py. Bus Reconstruction Applications (@) 2022
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

# COPYRIGHT SECTION
__author__ = "Diego Carvalho"
__copyright__ = "Copyright 2023"
__credits__ = ["Diego Carvalho"]
__license__ = "GPL"
__version__ = "2.0.0"
__maintainer__ = "Diego Carvalho"
__email__ = "d.carvalho@ieee.org"
__status__ = "Research"

from typing import List
import numpy as np
import pandas as pd

from pyarrow.parquet import ParquetFile
from pyarrow.lib import ArrowInvalid
import pyarrow as pa


def haversine(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
    to_radians: bool = True,
    earth_radius: float = 6371000.0,
) -> np.ndarray:
    """
    slightly modified version: of http://stackoverflow.com/a/29546836/2901002

    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees or in radians)

    All (lat, lon) coordinates must have numeric dtypes and be of equal length.

    """
    if to_radians:
        lat1 = np.radians(lat1)
        lat2 = np.radians(lat2)
        lon1 = np.radians(lon1)
        lon2 = np.radians(lon2)
        # lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

    θ = np.sin((lat2 - lat1) / 2.0)
    λ = np.sin((lon2 - lon1) / 2.0)
    γ = np.cos(lat1) * np.cos(lat2)
    a = np.square(θ) + γ * np.square(λ)

    return earth_radius * 2 * np.arcsin(np.sqrt(a))


def files_and_version_are_ok(
    parquet_file_list: List, version: str, df_list: List = [], df_read_all=False
) -> bool:
    ret_list = []

    # run over every file and check if it exists and the version
    for parquet_file in parquet_file_list:
        try:
            if df_read_all:
                df = pd.read_parquet(parquet_file)
            else:
                pf = ParquetFile(parquet_file)
                first_row = next(pf.iter_batches(batch_size=1))
                df = pa.Table.from_batches([first_row]).to_pandas()
            if df.SWVERSION[0] == version:
                df_list.append(df)
                ret_list.append(True)
            else:
                df_list.append(None)
                ret_list.append(False)
        except (
            FileNotFoundError,
            IndexError,
            KeyError,
            StopIteration,
            ArrowInvalid,
        ) as error:
            df_list.append(None)
            ret_list.append(False)

    # return true if every test was ok (true).
    return sum(ret_list) == len(ret_list)
