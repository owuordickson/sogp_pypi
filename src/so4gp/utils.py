# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GNU GPL v3
# This file is licensed under the terms of the GNU GPL v3.0.
# See the LICENSE file at the root of this
# repository for complete details.

"""
@author: Dickson Owuor
@credits: Thomas Runkler, Edmond Menya, and Anne Laurent
@license: GNU GPL v3
@email: owuordickson@gmail.com
@created: 21 July 2021
@modified: 27 October 2022

A collection of miscellaneous classes and methods.
"""

import os
import statistics
import numpy as np
import pandas as pd
import multiprocessing as mp
from tabulate import tabulate

try:
    from . import DataGP
    from .algorithms import GRAANK, TGradAMI
except ImportError:
    from src.so4gp import DataGP
    from src.so4gp.algorithms import GRAANK, TGradAMI


def analyze_gps(data_src, min_sup, est_gps, approach='bfs') -> str:
    """
    For each estimated GP, computes its true support using the GRAANK approach and returns the statistics (% error,
    and standard deviation).

    >>> import src.so4gp as sgp
    >>> import pandas
    >>> dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
    >>> columns = ['Age', 'Salary', 'Cars', 'Expenses']
    >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
    >>>
    >>> estimated_gps = list()
    >>> temp_gp = sgp.GP()
    >>> for gi_str in ['0+', '1-']:
    >>>    temp_gp.add_gradual_item(sgp.GI.from_string(gi_str))
    >>> temp_gp.support = 0.5
    >>> estimated_gps.append(temp_gp)
    >>> temp_gp = sgp.GP()
    >>> for gi_str in ['1+', '3-', '0+']:
    >>>    temp_gp.add_gradual_item(sgp.GI.from_string(gi_str))
    >>> temp_gp.support = 0.48
    >>> estimated_gps.append(temp_gp)
    >>> res = sgp.analyze_gps(dummy_df, min_sup=0.4, est_gps=estimated_gps, approach='bfs')
    >>> print(res)
    Gradual Pattern       Estimated Support    True Support  Percentage Error      Standard Deviation
    ------------------  -------------------  --------------  ------------------  --------------------
    ['0+', '1-']                       0.5              0.4  25.0%                              0.071
    ['1+', '3-', '0+']                 0.48             0.6  -20.0%                             0.085

    :param data_src: Data set file
    :type data_src: pd.DataFrame | str

    :param min_sup: Minimum support (set by user)
    :type min_sup: float

    :param est_gps: Estimated GPs
    :type est_gps: list

    :param approach: 'Bfs' (default) or 'dfs'
    :type approach: str

    :return: Tabulated results
    """
    if approach == 'dfs':
        d_set = DataGP(data_src, min_sup)
        d_set.fit_tids()
    else:
        d_set = DataGP(data_src, min_sup)
        d_set.fit_bitmap()
    headers = ["Gradual Pattern", "Estimated Support", "True Support", "Percentage Error", "Standard Deviation"]
    data = []
    for est_gp in est_gps:
        est_sup = est_gp.support
        est_gp.support = 0
        if approach == 'dfs':
            true_gp = est_gp.validate_tree(d_set)
        else:
            true_gp = est_gp.validate_graank(d_set)
        true_sup = true_gp.support

        if true_sup == 0:
            percentage_error = np.inf
            st_dev = np.inf
        else:
            percentage_error = ((est_sup - true_sup) / true_sup) * 100
            st_dev = statistics.stdev([est_sup, true_sup])

        if len(true_gp.gradual_items) == len(est_gp.gradual_items):
            data.append([est_gp.to_string(), round(est_sup, 3), round(true_sup, 3), str(round(percentage_error, 3))+'%',
                         round(st_dev, 3)])
        else:
            data.append([est_gp.to_string(), round(est_sup, 3), -1, np.inf, np.inf])
    return tabulate(data, headers=headers)


def gradual_decompose(data: pd.DataFrame, target: int) -> dict | Exception:
    """
    A method that decomposes multivariate timeseries data into its gradual components.  Attributes that have
    strong correlation will produce a decomposition graph with dense zigzag patterns. Those with weak correlation will
    produce a decomposition graph with sparse zigzag patterns.

    :param data: [required] the multivariate timeseries data as Pandas DataFrame.
    :param target: [required] the target column or feature or attribute.

    >>> import pandas
    >>> import src.so4gp as sgp
    >>> import matplotlib.pyplot as plt
    >>>
    >>> dummy_data = [["2021-03", 30, 3, 1, 10], ["2021-04", 35, 2, 2, 8], ["2021-05", 40, 4, 2, 7], ["2021-06", 50, 1, 1, 6], ["2021-07", 52, 7, 1, 2]]
    >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Date', 'Age', 'Salary', 'Cars', 'Expenses'])
    >>>
    >>> gp_trends = sgp.gradual_decompose(dummy_df, target=1)
    >>> print(gp_trends.keys())
    >>>
    >>> for key, val in gp_trends.items():
    >>>     plt.figure()
    >>>     plt.plot([p[0] for p in val], [p[1] for p in val], '-', label=f"{key}")
    >>>     plt.legend()
    >>>     plt.xlabel("GP 1 Index")
    >>>     plt.ylabel("GP 2 Index")
    >>>     plt.title(f"GP Warping Path")
    """

    try:
        t_grad = TGradAMI(data, target_col=target)
        eval_dict = t_grad.discover_tgp(use_clustering=False, eval_mode=True)
        gp_components = eval_dict['GP Components']
        """:type gp_components: dict"""
        return gp_components
    except Exception as e:
        # return {'Fatal Error': 'Try again with a bigger dataset or different dataset.'}
        raise Exception(e)


def get_num_cores() -> int:
    """
    Finds the count of CPU cores in a computer or a SLURM supercomputer.
    :return: Number of cpu cores (int)
    """
    num_cores = get_slurm_cores()
    if not num_cores:
        num_cores = mp.cpu_count()
    return num_cores


def get_slurm_cores() -> int | bool:
    """
    Test the computer to see if it is a SLURM environment, then gets the number of CPU cores.
    :return: Count of CPUs (int) or False
    """
    try:
        cores = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
        return cores
    except ValueError:
        try:
            str_cores = str(os.environ['SLURM_JOB_CPUS_PER_NODE'])
            temp = str_cores.split('(', 1)
            cpus = int(temp[0])
            str_nodes = temp[1]
            temp = str_nodes.split('x', 1)
            str_temp = str(temp[1]).split(')', 1)
            nodes = int(str_temp[0])
            cores = cpus * nodes
            return cores
        except ValueError:
            return False
    except KeyError:
        return False


def write_file(data, path, wr=True) -> None:
    """Description

    Writes data into a file
    :param data: information to be written
    :param path: name of file and storage path
    :param wr: writes data into the file if True
    :return:
    """
    if wr:
        with open(path, 'w') as f:
            f.write(data)
            f.close()
    else:
        pass
