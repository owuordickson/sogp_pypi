import pandas
from src.so4gp.algorithms import GRAANK, AntGRAANK, GeneticGRAANK, HillClimbingGRAANK, RandomGRAANK, ParticleGRAANK, TGrad, TGradAMI

if __name__ == "__main__":

    # dummy_data = [["2021-03", 30, 3, 1, 10], ["2021-03", 35, 2, 2, 8], ["2021-03", 40, 4, 3, 7], ["2021-03", 50, 1, 4, 6], ["2021-03", 52, 7, 5, 2]]
    dummy_data = [["2021-03", 30, 3, 1, 10], ["2021-04", 35, 2, 2, 8], ["2021-05", 40, 4, 2, 7], ["2021-06", 50, 1, 1, 6], ["2021-07", 52, 7, 1, 2]]
    dummy_df = pandas.DataFrame(dummy_data, columns=['Date', 'Age', 'Salary', 'Cars', 'Expenses'])
    # dummy_data = [[30, 3, 1, 10], [35, 2, 2, 8], [40, 4, 2, 7], [50, 1, 1, 6], [52, 7, 1, 2]]
    # dummy_df = pandas.DataFrame(dummy_data, columns=['Age', 'Salary', 'Cars', 'Expenses'])
    # mine_obj = GRAANK(dummy_df, min_sup=0.5, eq=False)
    # mine_obj = ClusterGP(dummy_df, 0.5, max_iter=3, e_prob=0.5)
    # mine_obj = AntGRAANK(dummy_df, 0.5, max_iter=20, e_factor=0.5)
    # mine_obj = GeneticGRAANK(dummy_df, 0.5, max_iter=20)
    # mine_obj = HillClimbingGRAANK(dummy_df, 0.5, max_iter=20)
    # mine_obj = RandomGRAANK(dummy_df, 0.5, max_iter=20)
    mine_obj = ParticleGRAANK(dummy_df, 0.5, max_iter=20)
    # mine_obj = TGrad(dummy_df, target_col=1, min_sup=0.2, min_rep=0.1)
    # mine_obj = TGradAMI(dummy_df, min_sup=0.5, target_col=1, min_rep=0.5, min_error=0.1)
    # result_json = mine_obj.discover(target_col=2)  # GRAANK
    result_json = mine_obj.discover()
    # result_json = mine_obj.discover_tgp(parallel=False)  # TGrad
    # result_json = mine_obj.discover_tgp(use_clustering=True, eval_mode=False)  # TGradAMI
    print(result_json)

    #from so4gp.utils import gradual_decompose
    #gp_trends = gradual_decompose(dummy_df, target=1)
    #print(gp_trends)
    # print(DataGP.test_time("09-01-2005"))
