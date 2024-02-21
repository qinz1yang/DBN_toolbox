from DBN_toolbox import qzy

variables = ['Num_intersection', 'Time_Helpful_SignSeen', 'Circularity', 'Occlusivity', 'Elongation', 'Drift Angle','Visible_All_Sign', 'Visible_Helpful_Sign', 'Cloest_Helpful_Dist','Jagged_360', 'sbsod', 'age', 'uncertain']
predictors = ['Num_intersection', 'Time_Helpful_SignSeen', 'Circularity', 'Occlusivity', 'Elongation', 'Drift Angle','Visible_All_Sign', 'Visible_Helpful_Sign', 'Cloest_Helpful_Dist','Jagged_360', 'sbsod', 'age']

network = qzy.DBN_ini(predictors)

# # T0
# qzy.DBN_train(network, columns=variables, data_name="Agent_UpdatedTra_Simulation.csv", bins = [4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], shuffled=True, seed=1056)
# qzy.DBN_evaluate(network, variables_to_add = predictors)

# # T1
qzy.DBN_T1(network, predictors=predictors, number_of_iterations=5, variables=variables)

# T2
# qzy.DBN_T2(dbn, "Agent_UpdatedTra_Simulation.csv", 123)

# # Hyperdata
# qzy.DBN_hyper(dbn, ran = 11)

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
