# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GNU GPL v3
# This file is licensed under the terms of the GNU GPL v3.0.
# See the LICENSE file at the root of this
# repository for complete details.

import gc
import json
import numpy as np

try:
    from ..data_gp import DataGP
    from ..gradual_patterns import GI, ExtGP
except ImportError:
    from src.so4gp import DataGP, GI, ExtGP

class GRAANK(DataGP):
    """Description

        Extracts gradual patterns (GPs) from a numeric data source using the GRAANK approach (proposed in a published
        research paper by Anne Laurent).

             A GP is a set of gradual items (GI) and its quality is measured by its computed support value. For example
             given a data set with 3 columns (age, salary, cars) and 10 objects. A GP may take the form: {age+, salary-}
             with a support of 0.8. This implies that 8 out of 10 objects have the values of column age 'increasing' and
             column 'salary' decreasing.

        This class extends class DataGP which is responsible for generating the GP bitmaps.

        """

    def __init__(self, *args, **kwargs):
        """
        Extracts gradual patterns (GPs) from a numeric dataset using the GRAANK algorithm. The algorithm relies on the
        APRIORI approach to generate GP candidates. This work was proposed by Anne Laurent
        and published in: https://link.springer.com/chapter/10.1007/978-3-642-04957-6_33.

             A GP is a set of gradual items (GI) and its quality is measured by its computed support value. For example
             given a data set with 3 columns (age, salary, cars) and 10 objects. A GP may take the form: {age+, salary-}
             with a support of 0.8. This implies that 8 out of 10 objects have the values of column age 'increasing' and
             column 'salary' decreasing.

        This class extends class DataGP which is responsible for generating the GP bitmaps.

        :param args: [required] data source path of Pandas DataFrame, [optional] minimum-support, [optional] eq

        >>> import so4gp as sgp
        >>> import pandas
        >>> import json
        >>> dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
        >>> dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
        >>>
        >>> mine_obj = sgp.GRAANK(data_source=dummy_df, min_sup=0.5, eq=False)
        >>> result_json = mine_obj.discover()
        >>> result = json.loads(result_json)
        >>> # print(result['Patterns'])
        >>> print(result_json) # doctest: +SKIP
        """
        super(GRAANK, self).__init__(*args, **kwargs)

    def _gen_apriori_candidates(self, gi_bins: np.ndarray, ignore_sup: bool = False,
                                target_col: int | None = None, exclude_target: bool = False):
        """Description

        Generates Apriori GP candidates (w.r.t target-feature/reference-column if provided). If user wishes to generate
        candidates that do not contain the target-feature then they do so by specifying exclude_target parameter.

        :param gi_bins: GI together with bitmaps
        :param ignore_sup: do not filter GPs based on minimum support threshold.
        :param target_col: target feature's column index
        :param exclude_target: only accept GP candidates that do not contain the target feature.
        :return: list of extracted GPs and the invalid count.
        """

        def inv_arr(g_item) -> tuple[int, str]:
            """Description

            Computes the inverse of a GI formatted as an array or tuple

            :param g_item: gradual item (array/tuple)
            :type g_item: (tuple, list) | np.ndarray

            :return: inverted gradual item
            """
            if g_item[1] == "+":
                return tuple((g_item[0], "-"))
            else:
                return tuple((g_item[0], "+"))

        min_sup = self.thd_supp
        n = self.attr_size

        invalid_count = 0
        res = []
        all_candidates = []
        if len(gi_bins) < 2:
            return []

        for i in range(len(gi_bins) - 1):
            for j in range(i + 1, len(gi_bins)):
                # 1. Fetch pairwise matrix
                try:
                    gi_i = {gi_bins[i][0]}
                    gi_j = {gi_bins[j][0]}
                    gi_o = {gi_bins[0][0]}
                except TypeError:
                    gi_i = set(gi_bins[i][0])
                    gi_j = set(gi_bins[j][0])
                    gi_o = set(gi_bins[0][0])

                # 2. Identify GP candidate (create its inverse)
                gp_cand = gi_i | gi_j
                inv_gp_cand = {inv_arr(x) for x in gp_cand}

                # 3. Apply target-feature search
                # (ONLY proceed if target-feature is part of the GP candidate - exclude_target is False)
                # (ONLY proceed if target-feature is NOT part of the GP candidate - exclude_target is True)
                if target_col is not None:
                    has_tgt_col = np.any(np.array([(y[0] == target_col) for y in gp_cand], dtype=bool))
                    if exclude_target and has_tgt_col:
                        continue
                    elif (not exclude_target) and (not has_tgt_col):
                        continue

                # 4. Verify validity of the GP candidate through the following conditions
                is_length_valid = (len(gp_cand) == len(gi_o) + 1)
                is_unique_candidate = ((not (all_candidates != [] and gp_cand in all_candidates)) and
                                    (not (all_candidates != [] and inv_gp_cand in all_candidates)))

                # 4. Validate GP and save it
                if is_length_valid and is_unique_candidate:
                    test = 1
                    repeated_attr = -1
                    for k in gp_cand:
                        if k[0] == repeated_attr:
                            test = 0
                            break
                        else:
                            repeated_attr = k[0]
                    if test == 1:
                        m = gi_bins[i][1] * gi_bins[j][1]
                        sup = float(np.sum(m)) / float(n * (n - 1.0) / 2.0)
                        if sup > min_sup or ignore_sup:
                            res.append([gp_cand, m, sup])
                        else:
                            invalid_count += 1
                    all_candidates.append(gp_cand)
                    gc.collect()
        return res, invalid_count

    def discover(self, ignore_support: bool = False, apriori_level: int | None = None,
                 target_col: int | None = None, exclude_target: bool = False):
        """Description

        Uses apriori algorithm to find gradual pattern (GP) candidates. The candidates are validated if their computed
        support is greater than or equal to the minimum support threshold specified by the user.

        :param ignore_support: do not filter extracted GPs using user-defined minimum support threshold.
        :param apriori_level: maximum APRIORI level for generating candidates.
        :param target_col: target feature's column index.
        :param exclude_target: only accept GP candidates that do not contain the target feature.

        :return: JSON object
        """

        self.fit_bitmap()

        self.gradual_patterns = []
        """:type gradual_patterns: list(so4gp.ExtGP)"""
        str_winner_gps = []
        valid_bins = self.valid_bins

        invalid_count = 0
        candidate_level = 1
        while len(valid_bins) > 0:
            valid_bins, inv_count = self._gen_apriori_candidates(valid_bins,
                                                                 ignore_sup=ignore_support,
                                                                 target_col=target_col,
                                                                 exclude_target=exclude_target)
            invalid_count += inv_count
            for v_bin in valid_bins:
                gi_arr = v_bin[0]
                # bin_data = v_bin[1]
                sup = v_bin[2]
                # if not ignore_support:
                self.gradual_patterns = ExtGP.remove_subsets(self.gradual_patterns, set(gi_arr))

                gp = ExtGP()
                """:type gp: ExtGP"""
                for obj in gi_arr:
                    gi = GI(obj[0], obj[1].decode())
                    """:type gi: GI"""
                    gp.add_gradual_item(gi)
                gp.set_support(sup)
                self.gradual_patterns.append(gp)
                str_winner_gps.append(gp.print(self.titles))
            candidate_level += 1
            if (apriori_level is not None) and candidate_level >= apriori_level:
                break
        # Output
        out = json.dumps({"Algorithm": "GRAANK", "Patterns": str_winner_gps, "Invalid Count": invalid_count})
        """:type out: object"""
        return out

    @staticmethod
    def decompose_to_gp_component(pairwise_mat: np.ndarray):
        """
        A method that decomposes the pairwise matrix of a gradual item/pattern into a warping path. This path is the
        decomposed component of that gradual item/pattern.

        :param pairwise_mat:
        :return: ndarray of warping path.
        """

        edge_lst = [(i, j) for i, row in enumerate(pairwise_mat) for j, val in enumerate(row) if val]
        """:type edge_lst: list"""
        return edge_lst
