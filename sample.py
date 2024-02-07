from pgmpy.models import DynamicBayesianNetwork as DBN
from DBN_toolbox import qzy

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

# T1
i = 1001
accs = []
n = 1499
while i <= n:
    dbn.clear()
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

    qzy.DBN_train(dbn, "Agent_UpdatedTra_Simulation.csv", True, i)
    print(str(i-1000) + "/" + str(n-1000))
    accs.append(qzy.DBN_acc(dbn))
    i += 1

qzy.plot_normal_distribution(accs)


# # T2
# qzy.DBN_T2(dbn, "Agent_UpdatedTra_Simulation.csv", 123)
