import pandas
import so4gp.so4gp as sgp
# import so4gp as sgp

if __name__ == "__main__":

    dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
    dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
    # mine_obj = sgp.GRAANK(data_source=dummy_df, min_sup=0.5, eq=False)
    # mine_obj = sgp.ClusterGP(dummy_df, 0.5, max_iter=3, e_prob=0.5)
    mine_obj = sgp.AntGRAANK(dummy_df, 0.5, max_iter=3, e_factor=0.5)
    result_json = mine_obj.discover()
    print(result_json)
