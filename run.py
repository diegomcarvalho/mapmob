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
__version__ = "2.0.12"
__maintainer__ = "Diego Carvalho"
__email__ = "d.carvalho@ieee.org"
__status__ = "Research"

from alive_progress import alive_bar
import ray
import glob

from applications import pipeline
from statvar import StatisticsVariable


def main():

    ray.init()

    stat_pool = StatisticsVariable()

    run_lines_id = stat_pool.create_variable("NUM_LINES")

    futures = []
    max_pending_tasks = 24 * 2

    database_dir = "/home/vinicius.vancellote/newmapbus/new/processed_data"
    metadata_dir = "metadata"
    workload = "/home/vinicius.vancellote/newmapbus/newbusdata/G1-*.parquet"

    n = len(glob.glob(workload))

    with alive_bar(n, dual_line=True, title=">") as bar:
        for w in glob.glob(workload):
            if len(futures) > max_pending_tasks:
                ready_tasks, not_ready = ray.wait(futures, num_returns=1)
                bar.text = f"-> Ready: {len(ready_tasks)}, Not ready tasks: {len(futures)} Next to be submitted: {w}"
                ret = ray.get(ready_tasks)
                for r in ret:
                    stat_pool.add_value(run_lines_id, r)
                    bar()
                    print(f" {stat_pool.sum(run_lines_id):16} processed lines")
                futures = not_ready
            fut = pipeline.remote(w, database_dir, metadata_dir)
            futures.append(fut)

        ret = ray.get(futures)
        for r in ret:
            stat_pool.add_value(run_lines_id, r)
            bar()


if __name__ == "__main__":
    main()
