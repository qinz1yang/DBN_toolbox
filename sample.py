from pgmpy.models import DynamicBayesianNetwork as DBN
from DBN_toolbox import qzy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore")

dbn = DBN()
dbn.add_edges_from([
    (('Num_intersection', 0), ('uncertain', 0)),
    (('Time_Helpful_SignSeen', 0), ('uncertain', 0)),
    # participant removed
    (('uncertain', 0), ('uncertain', 1)),
    (('Num_intersection', 1), ('uncertain', 1)),
    (('Time_Helpful_SignSeen', 1), ('uncertain', 1))
    # participant 1 removed
])

# data, thresholds = qzy.read_data("Agent_UpdatedTra_Simulation.csv", ['Time_Helpful_SignSeen', 'Num_intersection', 'uncertain', 'participant'], [4, 3, 3])
# data.to_csv("readdata.csv", index=False)
# print(thresholds)
# data = pd.read_csv("Agent_UpdatedTra_Simulation.csv", usecols=['Time_Helpful_SignSeen', 'Num_intersection', 'uncertain', 'participant'])
# plt.hist(data['uncertain'].dropna(), bins=50)  # Adjust bins if needed
# plt.title("Distribution of 'uncertain' Column")
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.show()

# # T0
# qzy.DBN_train(network=dbn, data_name="Agent_UpdatedTra_Simulation.csv",bins = [4, 3, 3], shuffled=True, seed=1021)
# qzy.DBN_evaluate(dbn)

# # T1
# i = 1001
# accs = []
# number_of_iterations = 20
# maxn = 1000+number_of_iterations
# highest_accuracy = -999
# highest_seed = -999
# while i <= maxn:
#     dbn.clear()
#     dbn = DBN()
#     dbn.add_edges_from([
#         (('Num_intersection', 0), ('uncertain', 0)),
#         (('Time_Helpful_SignSeen', 0), ('uncertain', 0)),
#         # participant removed
#         (('uncertain', 0), ('uncertain', 1)),
#         (('Num_intersection', 1), ('uncertain', 1)),
#         (('Time_Helpful_SignSeen', 1), ('uncertain', 1))
#         # participant 1 removed
#     ])

#     qzy.DBN_train(network=dbn, data_name="Agent_UpdatedTra_Simulation.csv",bins = [4, 3, 3], shuffled=True, seed=i)
#     print(str(i-1000) + "/" + str(number_of_iterations))
#     accuracy = qzy.DBN_acc(dbn)
#     if (accuracy > highest_accuracy):
#         highest_accuracy = accuracy
#         highest_seed = i
#     accs.append(accuracy)
#     i += 1

# print(f"highest accuracy: {highest_accuracy}")
# print(f"seed with highest accuracy: {highest_seed}")
# # qzy.plot_normal_distribution(accs)


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

# Compare models
qzy.compare_models()
