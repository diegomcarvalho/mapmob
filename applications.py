""" applications.py. Bus Reconstruction Applications (@) 2022
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
__version__ = "2.0.8"
__maintainer__ = "Diego Carvalho, Pablo Moreira, Vinicius Vancelloti"
__email__ = "d.carvalho@ieee.org"
__status__ = "Research"

import os
from typing import Any, Tuple
import ray

import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio


import pandas as pd
from tools import haversine, files_and_version_are_ok

from zipstorage import decode_meta_name, decode_zip_storage
from algo import AlgorithmFactory


# DST-0
def read_unique_entries_from_file(
    zip_file_name: str,
    output_file: str,
    directory: str,
    tag: str,
    version: str,
):
    # every record is structured into a tuple and inserted into a set. This
    # avoids repetitions and redundances.
    # NB.: Two entries are different if any component is not equal, so, a
    # vehicle can appear on the same position (geotag) but with different
    # GPS velocities
    unique_entries = set()

    # create the error metadata dict that will have each motif, filename and
    # other info for every datum eviction.
    error_metadata = dict()
    error_metadata["MOTIF"] = list()
    error_metadata["FILENAME"] = list()
    error_metadata["EXTRAINFO"] = list()

    for data in decode_zip_storage(zip_file_name):
        valid_entry, h_tag, entry = data
        if not valid_entry:
            error_metadata["MOTIF"].append(entry)
            error_metadata["FILENAME"].append(str(zip_file_name))
            error_metadata["EXTRAINFO"].append(str(h_tag))
            continue

        try:
            # convert the GPS time into a string
            gps_date = str(entry[0])
            # convert the bus index into a string
            busid = str(entry[1])
            # convert the service or service to str and remove the float marker
            line = str(entry[2]).replace(".0", "")
            # convert the latitude into a float
            latitude = float(entry[3])
            # convert the longitude into a float
            longitude = float(entry[4])
            # convert the velocity into a float and m/s
            velocity = float(entry[5]) / 3.6
        except (ValueError, IndexError) as e:
            error_metadata["MOTIF"].append(f"FIELD_ERROR:CORRUPTED{str(e)}")
            error_metadata["FILENAME"].append(str(zip_file_name))
            error_metadata["EXTRAINFO"].append(str(h_tag))
            continue

        # eio clip -o Rio-DEM.tif --bounds -24 -44 -22 -40
        if (
            latitude < -24.0
            or latitude > -22.0
            or longitude < -44.0
            or longitude > -40.0
        ):
            error_metadata["MOTIF"].append("FIELD_ERROR:LAT_LONG")
            error_metadata["FILENAME"].append(str(zip_file_name))
            error_metadata["EXTRAINFO"].append(str(h_tag))
            continue

        if velocity < 0.0 or velocity > 200.0:
            error_metadata["MOTIF"].append("FIELD_ERROR:VELOCITY")
            error_metadata["FILENAME"].append(str(zip_file_name))
            error_metadata["EXTRAINFO"].append(str(h_tag))
            continue

        # add to the unique_entries set the tuple with data
        unique_entries.add((gps_date, busid, line, latitude, longitude, velocity))

    # build the metadata with all processing errors and write it
    df_error = pd.DataFrame(error_metadata)
    df_error.to_parquet(f"{directory}/{tag}-ERROR-PH1.parquet")

    # build the data frame with all valid entries
    df = pd.DataFrame(
        unique_entries,
        columns=["DATE", "BUSID", "LINE", "LATITUDE", "LONGITUDE", "VELOCITY"],
    )

    # convert the date column to datetime and sort it, besides create a
    # privet key (index) for every record
    df["DATE"] = pd.to_datetime(df["DATE"], format="%m-%d-%Y %H:%M:%S", errors="coerce")
    df.index = df["DATE"]
    df.sort_index(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["ID"] = df.index

    df["SWVERSION"] = version

    df.to_parquet(output_file)

    return df


# DST-A
def elevation_pipeline(
    df: Any,
    output_file: str = "DST-A",
    version: str = "NA",
):

    if len(df) == 0:
        return

    dem_file = "regions/Rio-DEM.tif"

    with rasterio.open(dem_file) as dem_data:
        dem_array = dem_data.read(1)
        try:
            df["ELEVATION"] = dem_array[dem_data.index(df.LONGITUDE, df.LATITUDE)]
            df["SWVERSION"] = version
            df.to_parquet(output_file)
        except:
            df["ELEVATION"] = np.nan
            df["SWVERSION"] = version
            df.to_parquet(output_file)

    return


#  DST-C
def regions_pipeline(
    df: Any,
    region_file: str,
    ext_cols: list,
    map_cols: list,
    output_file: str,
    version: str,
):
    if len(df) == 0:
        return

    area_gpd = gpd.read_file(region_file)

    df["latitude"] = df["LATITUDE"].astype(str)
    df["longitude"] = df["LONGITUDE"].astype(str)

    df = df.assign(
        geometry=(
            "POINT Z (" + df["longitude"] + " " + df["latitude"] + " " + "0.00000)"
        )
    )

    cp_union = gpd.GeoDataFrame(
        df.loc[:, [c for c in df.columns if c != "geometry"]],
        geometry=gpd.GeoSeries.from_wkt(df["geometry"]),
        crs="epsg:4326",
    )

    dfjoin = gpd.sjoin(cp_union, area_gpd, how="left")

    df_result = pd.DataFrame()

    df_result["ID"] = df["ID"]

    for m, e in zip(map_cols, ext_cols):
        df_result[m] = pd.merge(df_result, dfjoin, on="ID")[e]
        # df_result[m] = dfjoin[e]

    df_result["SWVERSION"] = version

    df_result.to_parquet(output_file)

    return


# Join regions_pipeline DST-B
def join_barrios_garages_terminals_pipeline(
    df: Any,
    output_file: str,
    version: str,
):
    if len(df) == 0:
        return

    garagens = gpd.read_file("/home/carvalho/mapmob/regions/Garagens.geojson")
    terminais = gpd.read_file("/home/carvalho/mapmob/regions/Terminais.geojson")
    bairros = gpd.read_file("/home/carvalho/mapmob/regions/Limite_de_Bairros.geojson")
    velocidade = gpd.read_file("/home/carvalho/mapmob/regions/corredores.geojson")

    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.LONGITUDE, df.LATITUDE), crs="EPSG:4326"
    )

    dfjoin = gpd.sjoin(gdf, bairros, how="left")
    dfjoin = dfjoin.drop(
        columns=[
            "OBJECTID",
            "Área",
            "NOME",
            "AREA_PLANE",
            "REGIAO_ADM",
            "CODBNUM",
            "LINK",
            "RP",
            "Cod_RP",
            "CODBAIRRO_LONG",
            "SHAPESTArea",
            "SHAPESTLength",
            "index_right",
        ],
        axis=1,
    )

    # Fazendo o Join com as Garagens
    dfjoin = gpd.sjoin(dfjoin, garagens, how="left")
    dfjoin = dfjoin.drop(
        columns=[
            "latitude_max",
            "latitude_min",
            "longitude_max",
            "longitude_min",
            "nome_empresa",
            "tipo",
            "ativa",
            "index_right",
        ],
        axis=1,
    )

    # Fazendo o Join com os Terminais
    dfjoin = gpd.sjoin(dfjoin, terminais, how="left")
    dfjoin = dfjoin.drop(
        columns=[
            "nome_terminal",
            "index_right",
        ],
        axis=1,
    )

    # Fazendo o Join com os Corredores
    dfjoin = gpd.sjoin(dfjoin, velocidade, how="left")
    dfjoin = dfjoin.drop(
        columns=[
            "Name",
            "latitude",
            "longitude",
            "OBJECTID",
            "index_right",
            "geometry",
            "DATE",
            "BUSID",
            "LINE",
            "LATITUDE",
            "LONGITUDE",
            "VELOCITY",
            "SWVERSION",
            "ELEVATION"
        ],
        axis=1,
    )

    dfjoin.rename(
        columns={"id_garagem": "PARKING", "num_identificacao": "TERMINAL", "Cod": "CORREDOR"}, inplace=True
    )

    dfjoin["SWVERSION"] = version

    dfjoin.to_parquet(output_file)

    return


# DST-D
def calculate_daily_mobility(
    df: Any,
    d_file: str = "DST-D",
    e_file: str = "DST-E",
    version: str = "NA",
) -> Tuple[str, str]:

    if len(df) == 0:
        return

    df["SWVERSION"] = version

    g = df.groupby("BUSID")

    df["HDISTANCE"] = haversine(
        df["LATITUDE"], df["LONGITUDE"], g["LATITUDE"].shift(), g["LONGITUDE"].shift()
    )
    df["HEIGHT"] = df["ELEVATION"] - g["ELEVATION"].shift()
    df["DISTANCE"] = np.sqrt(np.square(df["HDISTANCE"]) + np.square(df["HEIGHT"]))
    df["INTERVAL"] = g["DATE"].diff()
    df["SPEED"] = df["DISTANCE"] / (df["INTERVAL"] / np.timedelta64(1, "s"))

    df.drop(
        columns=[
            "DATE",
            "BUSID",
            "LINE",
            "LATITUDE",
            "LONGITUDE",
            "VELOCITY",
            "ELEVATION",
        ],
    ).to_parquet(d_file)

    df["ACCELERATION"] = (df["SPEED"] - g["SPEED"].shift()) / (
        df["INTERVAL"] / np.timedelta64(1, "s")
    )

    df["VSP"] = (
        df["SPEED"] * df["ACCELERATION"] * df["HEIGHT"] + 0.092
    ) + 0.00021 * df["SPEED"] ** 3

    df["VSPMode"] = pd.cut(
        df["VSP"],
        bins=[-100, 0, 2, 4, 6, 8, 10, 13, 100],
        labels=["1", "2", "3", "4", "5", "6", "7", "8"],
    )

    df["CO_2"] = (
        df["VSPMode"]
        .map(
            {
                "1": 2.4,
                "2": 7.8,
                "3": 12.5,
                "4": 17.1,
                "5": 21.2,
                "6": 24.8,
                "7": 27.6,
                "8": 29.5,
            }
        )
        .astype("float64")
    )

    df["CO"] = (
        df["VSPMode"]
        .map(
            {
                "1": 0.009,
                "2": 0.036,
                "3": 0.045,
                "4": 0.072,
                "5": 0.085,
                "6": 0.091,
                "7": 0.084,
                "8": 0.062,
            }
        )
        .astype("float64")
    )

    df["NO_x"] = (
        df["VSPMode"]
        .map(
            {
                "1": 0.04,
                "2": 0.13,
                "3": 0.18,
                "4": 0.22,
                "5": 0.24,
                "6": 0.26,
                "7": 0.28,
                "8": 0.31,
            }
        )
        .astype("float64")
    )

    df["HC"] = (
        df["VSPMode"]
        .map(
            {
                "1": 1.23,
                "2": 1.70,
                "3": 1.75,
                "4": 1.84,
                "5": 1.94,
                "6": 2.05,
                "7": 2.08,
                "8": 2.15,
            }
        )
        .astype("float64")
    )

    df.drop(
        columns=[
            "DATE",
            "BUSID",
            "LINE",
            "LATITUDE",
            "LONGITUDE",
            "VELOCITY",
            "ELEVATION",
            "HDISTANCE",
            "HEIGHT",
            "DISTANCE",
            "INTERVAL",
            "SPEED",
            "ACCELERATION",
        ],
    ).to_parquet(e_file)

    return


@ray.remote(num_cpus=1)
def pipeline(
    zip_file_name: str, output_dir: str, metadata_dir: str, version: str = __version__
):

    tag = decode_meta_name(zip_file_name)

    dst_0_dir = f"{output_dir}/database/DST-0"
    dst_A_dir = f"{output_dir}/database/DST-A"
    dst_B_dir = f"{output_dir}/database/DST-B"
    dst_C_dir = f"{output_dir}/database/DST-C"
    dst_D_dir = f"{output_dir}/database/DST-D"
    dst_E_dir = f"{output_dir}/database/DST-E"

    dst_0_file = f"{dst_0_dir}/{tag}.parquet"
    dst_A_file = f"{dst_A_dir}/{tag}.parquet"
    dst_B_file = f"{dst_B_dir}/{tag}.parquet"
    dst_C_file = f"{dst_C_dir}/{tag}.parquet"
    dst_D_file = f"{dst_D_dir}/{tag}.parquet"
    dst_E_file = f"{dst_E_dir}/{tag}.parquet"

    output_list = [
        dst_0_file,
        dst_A_file,
        dst_B_file,
        dst_C_file,
        dst_D_file,
        dst_E_file,
    ]

    metadata_dir = f"{output_dir}/metadata"

    for i in [
        dst_0_dir,
        dst_A_dir,
        dst_B_dir,
        dst_C_dir,
        dst_D_dir,
        dst_E_dir,
        metadata_dir,
    ]:
        os.makedirs(i, exist_ok=True)

    df_list = []

    if files_and_version_are_ok(output_list, version, df_list, False):
        return 0

    # We need to read a Data Frame into memory... ;-)
    if df_list[0] is None:
        df = read_unique_entries_from_file(
            zip_file_name, dst_0_file, metadata_dir, tag, version
        )
    else:
        df = pd.read_parquet(dst_0_file)

    if df_list[1] is None:
        elevation_pipeline(df, dst_A_file, version)
    else:
        df = pd.read_parquet(dst_A_file)

    if df_list[2] is None:
        join_barrios_garages_terminals_pipeline(
            df.copy(),
            dst_B_file,
            version,
        )

    if df_list[3] is None:
        regions_pipeline(
            df.copy(),
            "regions/Zonas_Pluviometricas.geojson",
            ["Cod"],
            ["PLUVIOMETRICREG"],
            dst_C_file,
            version,
        )

    if (df_list[4] is None) or (df_list[5] is None):
        calculate_daily_mobility(df, dst_D_file, dst_E_file, version)

    return len(df)


@ray.remote(num_cpus=1)
def stat_pipeline(file_name: str, tag: str, base: str, file_list: str, algo):

    file_base = decode_meta_name(file_name)

    dst_0_dir = f"{base}/DST-0"
    dst_A_dir = f"{base}/DST-A"
    dst_B_dir = f"{base}/DST-B"
    dst_C_dir = f"{base}/DST-C"
    dst_D_dir = f"{base}/DST-D"
    dst_E_dir = f"{base}/DST-E"

    dst_0_file = f"{dst_0_dir}/{file_base}.parquet"
    dst_A_file = f"{dst_A_dir}/{file_base}.parquet"
    dst_B_file = f"{dst_B_dir}/{file_base}.parquet"
    dst_C_file = f"{dst_C_dir}/{file_base}.parquet"
    dst_D_file = f"{dst_D_dir}/{file_base}.parquet"
    dst_E_file = f"{dst_E_dir}/{file_base}.parquet"

    df = pd.read_parquet(dst_A_file).drop("SWVERSION", axis=1)

    if "B" in file_list:
        df = pd.merge(
            df,
            pd.read_parquet(dst_B_file).drop("SWVERSION", axis=1),
            on="ID",
        )

    if "C" in file_list:
        df = pd.merge(
            df,
            pd.read_parquet(dst_C_file).drop("SWVERSION", axis=1),
            on="ID",
        )

    if "D" in file_list:
        df = pd.merge(
            df,
            pd.read_parquet(dst_D_file).drop("SWVERSION", axis=1),
            on="ID",
        )

    if "E" in file_list:
        df = pd.merge(
            df,
            pd.read_parquet(dst_E_file).drop("SWVERSION", axis=1),
            on="ID",
        )

    ag = AlgorithmFactory.get_algorithm(algo)

    return ag.visit_data_frame(tag, df)


"""     except:
        ret_t = ag.ret_type()
        ret_val = [tag]
        for x in range(len(ret_t._fields) - 1):
            ret_val.append(0)
        return ret_t._make(ret_val)
 """
