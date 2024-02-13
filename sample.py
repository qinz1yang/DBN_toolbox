from pgmpy.models import DynamicBayesianNetwork as DBN
from DBN_toolbox import qzy
import pandas as pd
import matplotlib.pyplot as plt

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

# T0
qzy.DBN_train(dbn, bins = [4, 3, 3])
qzy.DBN_evaluate(dbn)

# # T1
# i = 1001
# accs = []
# n = 1499
# while i <= n:
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

#     qzy.DBN_train(dbn, "Agent_UpdatedTra_Simulation.csv", True, i)
#     print(str(i-1000) + "/" + str(n-1000))
#     accs.append(qzy.DBN_acc(dbn))
#     i += 1

# qzy.plot_normal_distribution(accs)


# T2
# qzy.DBN_T2(dbn, "Agent_UpdatedTra_Simulation.csv", 123)

# # Hyperdata
# qzy.DBN_hyper(dbn, ran = 11)
