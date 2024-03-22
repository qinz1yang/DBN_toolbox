import multiprocessing
from DBN_toolbox import qzy

cores = multiprocessing.cpu_count()
print(f"Core number: {cores}")

# variables = ['Occlusivity', 'Visible_Helpful_Sign', 'Time_Helpful_SignSeen', 'Num_intersection', 'group', 'sbsod', 'sa', 'mt', 'age', 'uncertain']
# predictors = ['Occlusivity', 'Visible_Helpful_Sign', 'Time_Helpful_SignSeen', 'Num_intersection', 'group', 'sbsod', 'sa', 'mt', 'age']
# local_bins = [3, 5, 5, 3, 4, 3, 4, 5, 3, 3]

variables = ['Num_intersection', 'Time_Helpful_SignSeen', 'sbsod', 'framecount', 'uncertain']
predictors = ['Num_intersection', 'Time_Helpful_SignSeen', 'sbsod', 'framecount']
local_bins = [5, 4, 4, 3, 3] 

network = qzy.DBN_ini(predictors)

# # T0
# test_data, network = qzy.DBN_train(network, columns=variables, data_name="Agent_UpdatedTra_Simulation.csv", bins = local_bins, shuffled=True, seed=3524, fast=True)
# qzy.DBN_evaluate(network=network, test_data=test_data, variables_to_add=predictors)

# # fast
# test_data, network = qzy.DBN_train(network, columns=variables, data_name="Agent_UpdatedTra_Simulation.csv", bins = local_bins, shuffled=True, seed=1097, fast=True)
# print(qzy.DBN_fast_acc_and_sensitivity(network, test_data, variables_to_add=predictors))

# # T1
# qzy.DBN_T1(network, predictors=predictors, number_of_iterations=8, variables=variables, bins=local_bins)

# T1_multithreaded
qzy.DBN_T1_multithreaded(network, number_of_iterations=500, num_threads=cores, predictors=predictors, variables=variables, bins=local_bins)

# T2
# qzy.DBN_T2(network, "Agent_UpdatedTra_Simulation.csv", 123)

# # Hyperdata
# qzy.DBN_hyper(network, columns=variables, predictors=predictors, ran = 8, seed=1019)

# # T3
# qzy.DBN_T3(network=dbn, data_name="Agent_UpdatedTra_Simulation.csv",bins = [4, 3, 3], shuffled=True, seed=1021)
# qzy.DBN_evaluate(dbn)

# # T4
# qzy.DBN_T4(network=dbn, data_name="Agent_UpdatedTra_Simulation.csv",bins = [4, 5, 3])
# qzy.DBN_evaluate(dbn)

# # Hyperdata T4
# qzy.DBN_T4_hyper(dbn, ran = 11)

# # Logistic Regression
# print(qzy.Logit(network=dbn, data_name="Agent_UpdatedTra_Simulation.csv",bins = [4, 5, 3], shuffled=True, seed=99))

# # Random_Forest
# print(qzy.RandomForestModel(network=dbn, data_name="Agent_UpdatedTra_Simulation.csv",bins = [4, 5, 3], shuffled=True, seed=124))

# # Test qzy.DBN_acc_and_sensitivity
# qzy.DBN_train(network=dbn, data_name="Agent_UpdatedTra_Simulation.csv",bins = [4, 3, 3], shuffled=True, seed=1021)
# print(qzy.DBN_acc_and_sensitivity(dbn))

# # Compare models
# qzy.compare_models()

# # By task
# qzy.DBN_train_and_evaluate_by_task(network, columns=variables, data_name="Agent_UpdatedTra_Simulation.csv",predictors=predictors, bins = local_bins, seed=3524)

# #range
# qzy.DBN_evaluate_over_seed_range(network=network, seed_start=1001, seed_end=1999, columns=variables, data_name="Agent_UpdatedTra_Simulation.csv", predictors=predictors, bins = local_bins)

# # try combinations
# print(qzy.DBN_predictor_impact(data_name="Agent_UpdatedTra_Simulation.csv", columns=variables, predictors=predictors, bins=local_bins, seed=3524))
