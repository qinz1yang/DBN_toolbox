from pgmpy.models import DynamicBayesianNetwork as DBN
from DBN_toolbox import qzy

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

qzy.DBN_train(dbn, "Agent_UpdatedTra_Simulation.csv", True, 123)
qzy.DBN_evaluate(dbn)
