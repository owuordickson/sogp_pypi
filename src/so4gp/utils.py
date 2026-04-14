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
import numpy as np
import multiprocessing as mp


def gen_gradual_warping_path(pairwise_mat: np.ndarray, as_array: bool = False) -> list[tuple[int, str]] | np.ndarray:
    """
    A method that decomposes the pairwise matrix of a gradual item/pattern into a warping path. Attributes that have
strong correlation will produce a warping path with dense zigzag patterns. Those with weak correlation will
produce a warping path with sparse zigzag patterns.

    :param pairwise_mat: The pairwise matrix of a gradual item/pattern.
    :param as_array: If True, returns the warping path as a numpy array else as a list of tuples.

    :return: A list array of the warping path (as an edge list).
    """

    edge_lst = [(i, j) for i, row in enumerate(pairwise_mat) for j, val in enumerate(row) if val]
    """:type edge_lst: list"""
    if as_array:
        return np.array(edge_lst)
    return edge_lst


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
    """
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
