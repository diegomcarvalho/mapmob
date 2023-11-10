""" zipstorage.py. Bus Reconstruction Applications (@) 2022
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
__copyright__ = "Copyright 2022"
__credits__ = ["Diego Carvalho"]
__license__ = "GPL"
__version__ = "2.0.0"
__maintainer__ = "Diego Carvalho"
__email__ = "d.carvalho@ieee.org"
__status__ = "Research"

import json
import os
import typing
import zipfile
import pandas as pd


def decode_entries_in_file(f: typing.TextIO, null_DATA: typing.Any) -> typing.List:
    """
    decode the JSON entry into a dict, returning the "DATA" member.

    If the conversion fails, return the null_DATA sentinel.
    """
    data = dict()
    data["DATA"] = null_DATA
    try:
        data = json.load(f)
    except ValueError:
        # includes simplejson.decoder.JSONDecodeError
        # invalid JSON numbers are encountered
        pass
    return data["DATA"]


def decode_meta_name(file_name):
    info = os.path.basename(file_name).split(".")
    return info[0]

def decode_parquet_storage(parquet_file_name: str) -> typing.Any:
    tag = decode_meta_name(parquet_file_name)

    # format: G1-2017-07-12
    _, meta_day = tag[:2], tag[3:]
    h_tag = f"{meta_day}:XX"

    try:
       df = pd.read_parquet(parquet_file_name)
    except:
        return False, h_tag, "FILE NOT FOUND"

    # VINICIUS - completar o r_rval...
    # convert the GPS time into a string
    #gps_date = str(entry[0])
    # convert the bus index into a string
    #busid = str(entry[1])
    # convert the service or service to str and remove the float marker
    #line = str(entry[2]).replace(".0", "")
    # convert the latitude into a float
    #latitude = float(entry[3])
    # convert the longitude into a float
    #longitude = float(entry[4])
    # convert the velocity into a float and m/s
    #velocity = float(entry[5]) / 3.6

    for index, row in df.iterrows():
        r_val = [row['DATE'], row['BUSID'], row['LINE'], row['LATITUDE'], row['LONGITUDE'], row['VELOCITY']]
        yield True, h_tag, r_val

def decode_zip_storage(zip_file_name: str) -> typing.Any:

    tag = decode_meta_name(zip_file_name)

    # format: G1-2017-07-12
    _, meta_day = tag[:2], tag[3:]
    h_tag = f"{meta_day}:XX"

    # null data is a standard sentinel for an error during the conversion
    null_DATA = [[]]

    # Let's do the hard work
    try:
        # open the zipfile. We should find a directory file called tmp and a
        # file for every minute in a day, such as hh-mm.txt (as decoded by the
        # function decode_entries_in_file)
        with zipfile.ZipFile(zip_file_name) as file_handler:
            # now loop over every minute in a day
            for file_name in file_handler.namelist():
                meta_hour = decode_meta_name(file_name)
                h_tag = f"{meta_day}:{meta_hour}"
                try:
                    # open the minute file in the zip and process every entry
                    with file_handler.open(file_name, "r") as f:
                        entries_in_minute_file = decode_entries_in_file(f, null_DATA)
                        if (
                            len(entries_in_minute_file) == 0
                            or entries_in_minute_file == null_DATA
                        ):

                            yield False, h_tag, "DECODE FAIL"
                            continue

                        # loop over every vehicle entry
                        for each_entry in entries_in_minute_file:

                            yield True, h_tag, each_entry
                except zipfile.BadZipFile as e:  # zipfile.BadZipFile
                    yield False, h_tag, "FIELDERROR"
                    continue
    except zipfile.BadZipFile as e:  # BadZipFile
        yield False, h_tag, "BADZIPFILE"
    except FileNotFoundError as e:
        yield False, h_tag, "FILE NOT FOUND"
