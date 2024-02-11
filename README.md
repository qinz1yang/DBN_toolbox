# DBN_toolbox
An encapsulated Python toolbox for training and evaluating the (Dynamic) Bayesian Network. It is designed for the research purposes in Cornell Design and Augmented Intelligence Lab(DAIL).

This class can be called using "from DBN_toolboxes import qzy", and then calling arguments like "qzy.DBN_train()"

## Contains functions:

{

plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix', cmap=plt.cm.Blues) 

plot_normal_distribution(data)

DBN_train(network, data_name = "Agent_UpdatedTra_Simulation.csv", shuffled = False, seed = 123)

DBN_evaluate(network, test_data_name = "test_data.csv", model_name = "trained_model.pkl")

DBN_acc(network, test_data_name = "test_data.csv", model_name = "trained_model.pkl")

}

* DBN_evaluate() will generate a html evaluation report containing confusion matrix and other statistics about the model(model_evaluation_results.html).
* DBN_acc will return the accuracy of the model only.

NOTE THAT this toolbox has designated way of dealing with data that only fits the research purposes of Cornell DAIL. You are free to edit the code to fit your data structures.

Sample call is attached in sample.py. It trains defines an DBN, reads the data, shuffles the data randomly according to participant index, separates the data into test data and train data, and use test data to evaluate the model.

I have to acknowledge that storing test data to csv in training stage and reading the csv in testing stage is silly and takes a huge amount of IO time. I will try to find a solution when we move onto cross validation section.
