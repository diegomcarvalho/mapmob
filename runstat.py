""" run.py. Bus Reconstruction Applications (@) 2022
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

from alive_progress import alive_bar
import ray
import glob

import pandas as pd
from algo import AlgorithmFactory

from applications import stat_pipeline
from statvar import StatisticsVariable
from codetiming import Timer

from zipstorage import decode_meta_name


def main():

    ray.init()

    stat_pool = StatisticsVariable()

    run_lines_id = stat_pool.create_variable("NUM_LINES")

    ag = AlgorithmFactory.get_algorithm("algo01")

    futures = []
    max_pending_tasks = 24 * 4

    database_dir = "/home/carvalho/processed_data/database"
    workload = f"{database_dir}/DST-A/G1-20[12][82]*.parquet"

    stat_data = list()
    meta_timer = dict()
    meta_data = list()

    n = len(glob.glob(workload))

    with alive_bar(n, dual_line=True, title=">") as bar:
        for w in glob.glob(workload):
            if len(futures) > max_pending_tasks:
                ready_tasks, not_ready = ray.wait(futures, num_returns=1)
                bar.text = f"-> Ready: {len(ready_tasks)}, Not ready tasks: {len(futures)} Next to be submitted: {w}"
                ret = ray.get(ready_tasks)
                for r in ret:
                    tag = r.TAG
                    meta_data.append((tag, meta_timer[tag].stop()))
                    stat_pool.add_value(run_lines_id, r.NUM_OBS)
                    bar()
                    print(
                        f" {tag} - {stat_pool.sum(run_lines_id):16} processed lines - {r.NUM_BUS} vehicles"
                    )
                    stat_data.append(r)
                futures = not_ready
            tag = decode_meta_name(w).replace("G1-", "")
            meta_timer[tag] = Timer(name=tag, logger=None)
            meta_timer[tag].start()
            fut = stat_pipeline.remote(
                w,
                tag,
                "/home/carvalho/processed_data/database",
                ["A", "D", "E"],
                ag.name,
            )
            futures.append(fut)

        ret = ray.get(futures)
        for r in ret:
            tag = r.TAG
            meta_data.append((tag, meta_timer[tag].stop()))
            stat_data.append(r)
            stat_pool.add_value(run_lines_id, r.NUM_OBS)
            bar()

    df = pd.DataFrame(stat_data, columns=ag.columns)

    df.to_parquet("/home/carvalho/processed_data/general_stat.parquet")

    df_meta = pd.DataFrame(meta_data, columns=["TAG", "TIME"])

    df_meta.to_parquet("/home/carvalho/processed_data/processing_time.parquet")

    return


if __name__ == "__main__":
    main()
