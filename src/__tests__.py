import pandas
import so4gp as sgp
from so4gp.algorithms import GRAANK


if __name__ == "__main__":

    # dummy_data = [["2021-03", 30, 3, 1, 10], ["2021-03", 35, 2, 2, 8], ["2021-03", 40, 4, 3, 7], ["2021-03", 50, 1, 4, 6], ["2021-03", 52, 7, 5, 2]]
    # dummy_df = pandas.DataFrame(dummy_data, columns=['Date', 'Age', 'Salary', 'Cars', 'Expenses'])
    dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
    dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
    mine_obj = GRAANK(dummy_df, min_sup=0.5, eq=False)
    # mine_obj = ClusterGP(dummy_df, 0.5, max_iter=3, e_prob=0.5)
    # mine_obj = AntGRAANK(dummy_df, 0.5, max_iter=3, e_factor=0.5)
    # result_json = mine_obj.discover(target_col=4)
    result_json = mine_obj.discover()
    print(result_json)

    # from so4gp.miscellaneous import gradual_decompose
    # gp_trends = gradual_decompose(dummy_df, target=1)
    # print(gp_trends)
    # print(DataGP.test_time("09-01-2005"))
