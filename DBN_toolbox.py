# DBN_toolbox

# train
import pandas as pd
from pgmpy.models import DynamicBayesianNetwork as DBN
import pickle
import numpy as np
# evaluate
import pandas as pd
import pickle
from pgmpy.inference import DBNInference
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np
import webbrowser
import os
from statistics import mean 
# plot
import matplotlib.pyplot as plt
import scipy.stats as stats

# logistic regression
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

class qzy():
    """
    qzy's tools for training and evaluating DBN networks
    """

    def DBN_ini(variables_to_add = ['Num_intersection', 'Time_Helpful_SignSeen', 'Circularity', 'Occlusivity', 'Elongation', 'DriftAngle','Visible_All_Sign', 'Visible_Helpful_Sign', 'Closest_Helpful_Dist','Jagged_360', 'sbsod', 'age']):
        dbn = DBN()
        dbn_edges = [
            (('uncertain', 0), ('uncertain', 1)),
        ]
        for variable in variables_to_add:
            dbn_edges.append(((variable, 0), ('uncertain', 0)))
            dbn_edges.append(((variable, 1), ('uncertain', 1)))
        
        dbn.add_edges_from(dbn_edges)

        if dbn.check_model():
            return dbn
        else:
            print("There is an error initializing the model.")



    def plot_confusion_matrix(y_true, y_pred, classes, model_name, title='Confusion Matrix', cmap=plt.cm.Blues):
        """
        Method to plot confusion matrix
        Parameters:
        - true values
        - predictions
        - classes
        - title, default = 'Confusion Matrix'
        - cmap, default = plt.cm.Blues
        """
        confusion_matrix_filename = f"{model_name}_comfusion_matrix.png"
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 7))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(fname = confusion_matrix_filename)

    def plot_normal_curve(data, label, color):
        mu, std = np.mean(data), np.std(data)
        xs = np.linspace(mu - 3*std, mu + 3*std, 100)
        ys = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - mu) / std) ** 2)
        plt.plot(xs, ys, label=f'{label} Normal Dist.', color=color)
        plt.axvline(x=mu, color=color, linestyle='--', label=f'{label} Mean: {mu:.4f} SD: {std:.4f}')

    def plot_normal_distribution(data):
        """
        Method to plot normal distribution based on given data
        Parameters
        - data
        """
        data = np.array(data)
        median = np.median(data)
        sd = np.std(data)
        print(f"Median: {median}")
        print(f"Standard Deviation: {sd}")
        plt.figure(figsize=(10, 6))
        density = stats.gaussian_kde(data)
        xs = np.linspace(min(data), max(data), 200)
        plt.plot(xs, density(xs), label='Density')

        plt.axvline(x=median, color='r', linestyle='--', label='Median')
        plt.axvline(x=median-sd, color='g', linestyle=':', label='Median - 1 SD')
        plt.axvline(x=median+sd, color='g', linestyle=':', label='Median + 1 SD')
    
        plt.title('Normal Distribution of Data')
        plt.xlabel('Data')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig('normal_distribution.png')

    def plot_ground_truth(true_labels, predictions, model_name):
        plt.figure(figsize=(100, 6))
        plt.plot(true_labels, label='Ground Truth', linestyle='-', color='blue', linewidth=1)
        for i, (true, pred) in enumerate(zip(true_labels, predictions)):
            if true == pred:
                plt.scatter(i, true, color='red', label='Correct Prediction' if i == 0 else "")
            # else:
            #     plt.scatter(i, true, color='blue', label='Ground Truth' if i == 0 else "")
            # print(i)
            print(str(i+1) + "/" + str(len(true_labels)), end = "\r")
        plt.title('Ground Truth red')
        plt.xlabel('Sample Index')
        plt.ylabel('Class')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(f'{model_name}_ground_truth.png')

    def read_data(data_name, columns, bins):
        """
        Reads specified columns from a CSV file, divides each column into specified bins,
        and assigns labels to these bins starting from 0, 1, 2, and so on.
        
        Parameters:
        - data_name: Name of the CSV file to read from.
        - columns: List of column names to be read and discretized.
        - bins: List of integers representing the number of bins for each column.
        """
        local_columns = columns
        columns.append('participant')
        data = pd.read_csv(data_name, usecols=local_columns)
        thresholds = {}
        i = 0
        while i < len(local_columns):
            if local_columns[i] == 'participant' or local_columns == 'task':
                i += 1
            else:
                data[local_columns[i]], bins_edges = pd.qcut(data[local_columns[i]], q=bins[i], labels=False, duplicates='drop', retbins=True)
                thresholds[local_columns[i]] = bins_edges
                print(f"Thresholds for {local_columns[i]}: {bins_edges}")
                i += 1
        return data


    def DBN_train(network, columns = ['Time_Helpful_SignSeen', 'Num_intersection', 'uncertain', 'participant'], bins = [4, 3, 3],  data_name = "Agent_UpdatedTra_Simulation.csv", shuffled = False, seed = 123, model_name = "trained_model.pkl"):
        """
        Method to train a dbn network
        Parameters
        - network: accepts a Dynamic Bayesian network defined by pgmpy
        - data name, default = "Agent_UpdatedTra_Simulation.csv": the name of data file
        - shuffled, default = False: parameter controling whether to shuffle the data by participant index. Default is not shuffle
        - seed, default = 123: sets numpy seeds for shuffling, default seed is 123
        """
        data = qzy.read_data(data_name, columns, bins)
        if (shuffled == True):
            np.random.seed(seed)
            groups = [df for _, df in data.groupby('participant')]
            np.random.shuffle(groups)
            shuffled_data = pd.concat(groups).reset_index(drop=True)
            split_index = int(len(shuffled_data) * 0.8)
            train_data = shuffled_data[:split_index]
            test_data = shuffled_data[split_index:]
            train_data = train_data.drop(columns=['participant'])
            test_data = test_data.drop(columns=['participant'])
            data_t0 = train_data.rename(columns={col: (col, 0) for col in train_data.columns})
            data_t1 = train_data.shift(-1).rename(columns={col: (col, 1) for col in train_data.columns})
            complete_data = pd.concat([data_t0, data_t1], axis=1).dropna()
        else:
            np.random.seed(seed)
            shuffled_data = data
            split_index = int(len(shuffled_data) * 0.8)
            train_data = shuffled_data[:split_index]
            test_data = shuffled_data[split_index:]
            data_t0 = train_data.rename(columns={col: (col, 0) for col in train_data.columns})
            data_t1 = train_data.shift(-1).rename(columns={col: (col, 1) for col in train_data.columns})
            complete_data = pd.concat([data_t0, data_t1], axis=1).dropna()

        network.fit(complete_data, estimator='MLE')
        with open(model_name, "wb") as file:
            pickle.dump(network, file, protocol=pickle.HIGHEST_PROTOCOL)
        test_data.to_csv("test_data.csv", index=False)
        train_data.to_csv("train_data.csv", index=False)
        print("Training completed and model saved.")

    def DBN_evaluate(network, test_data_name="test_data.csv", model_name="trained_model.pkl", variables_to_add = ['Time_Helpful_SignSeen', 'Num_intersection']):
        """
        Method to evaluate the DBN model based on test data given.
        Parameters:
        - network: accepts a Dynamic Bayesian network defined by pgmpy
        - test data name, default = "test_data.csv": name of test data file
        - model name, default = "trained_model.pkl": name of trained model
        """
        confusion_matrix_filename = f"{model_name}_comfusion_matrix.png"

        with open(model_name, "rb") as file:
            dbn = pickle.load(file)
        test_data = pd.read_csv(test_data_name)
        dbn_inference = DBNInference(dbn)
        predictions = []
        true_labels = []
        classes = [0, 1]

        available_vars = [var for var in variables_to_add if var in test_data.columns]
        print("avaliable vars")
        print(available_vars)
        for i in range(len(test_data) - 1):
            current_row = test_data.iloc[i]
            evidence = {}
            for var in variables_to_add:
                if var in current_row:
                    evidence[(var, 0)] = current_row[var]
            # print(evidence)
            prediction = dbn_inference.forward_inference([('uncertain', 1)], evidence=evidence)
            most_confident_prediction = np.argmax(prediction[('uncertain', 1)].values)
            actual_value = test_data.iloc[i + 1]['uncertain']
            predictions.append(most_confident_prediction)
            true_labels.append(actual_value)

        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, labels=classes, average='weighted')
        recall = recall_score(true_labels, predictions, labels=classes, average='weighted')
        f1 = f1_score(true_labels, predictions, labels=classes, average='weighted')
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        qzy.plot_confusion_matrix(true_labels, predictions, classes, model_name)
        qzy.plot_ground_truth(true_labels, predictions, model_name)

        html_filename = f'{model_name}_results.html'
        with open(html_filename, 'w') as f:
            f.write(f"<h1>Model Evaluation Results</h1>")
            f.write(f"<p>Model Name: <b>{model_name}</b></p>")
            f.write(f"<p>Test data name: <b>{test_data_name}</b></p>")
            f.write(f"<p>Accuracy: {accuracy}</p>")
            f.write(f"<p>Confusion Matrix:</p>")
            f.write(f'<img src="{confusion_matrix_filename}"><br>')
            f.write(f"<p>Precision: {precision}</p>")
            f.write(f"<p>Recall: {recall}</p>")
            f.write(f"<p>F1 Score: {f1}</p>")
            f.write(f"<h2>Ground Truth vs Predicted Diagram</h2>")
            f.write(f'<img src="{model_name}_ground_truth.png"><br>')

        # webbrowser.open('file://' + os.path.realpath(html_filename))

    def DBN_acc(network, test_data_name = "test_data.csv", model_name = "trained_model.pkl"):
        with open(model_name, "rb") as file:
            dbn = pickle.load(file)
        test_data = pd.read_csv(test_data_name)
        dbn_inference = DBNInference(dbn)
        predictions = []
        true_labels = []
        classes = [0, 1]
        for i in range(len(test_data) - 1):
            current_row = test_data.iloc[i]
            evidence = {
                ('Time_Helpful_SignSeen', 0): current_row['Time_Helpful_SignSeen'],
                ('Num_intersection', 0): current_row['Num_intersection']
            }
            prediction = dbn_inference.forward_inference([('uncertain', 1)], evidence=evidence)
            most_confident_prediction = np.argmax(prediction[('uncertain', 1)].values)
            actual_value = test_data.iloc[i + 1]['uncertain']
            predictions.append(most_confident_prediction)
            true_labels.append(actual_value)
        return(accuracy_score(true_labels, predictions))
        
    def DBN_acc_and_sensitivity(network, test_data_name="test_data.csv", model_name="trained_model.pkl"):
        with open(model_name, "rb") as file:
            dbn = pickle.load(file)
        test_data = pd.read_csv(test_data_name)
        dbn_inference = DBNInference(dbn)
        predictions = []
        true_labels = []
        classes = [0, 1]

        for i in range(len(test_data) - 1):
            current_row = test_data.iloc[i]
            evidence = {
                ('Time_Helpful_SignSeen', 0): current_row['Time_Helpful_SignSeen'],
                ('Num_intersection', 0): current_row['Num_intersection']
            }
            prediction = dbn_inference.forward_inference([('uncertain', 1)], evidence=evidence)
            most_confident_prediction = np.argmax(prediction[('uncertain', 1)].values)
            actual_value = test_data.iloc[i + 1]['uncertain']
            predictions.append(most_confident_prediction)
            true_labels.append(actual_value)

        accuracy = accuracy_score(true_labels, predictions)

        cm = confusion_matrix(true_labels, predictions)
        TP = cm[1, 1]
        FN = cm[1, 0]

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0

        return accuracy, sensitivity

    def DBN_T2(network, data_name="Agent_UpdatedTra_Simulation.csv", seed=123):
        """
        Method to train and evaluate a dbn network for each participant separately with constant tasks across participants
        tasks are randomly selected, with the rest to be test data.
        Parameters:
        - network: DBN network
        - data_name: The name of the data file
        - seed: Seed for numpy's random number generator for reproducibility
        """

        accuracies = []
        data = pd.read_csv(data_name, usecols=['Time_Helpful_SignSeen', 'Num_intersection', 'uncertain', 'participant', 'task'])
        data['Time_Helpful_SignSeen'] = pd.cut(data['Time_Helpful_SignSeen'], bins=[0, 8, 85, 336, 9999], labels=[0, 1, 2, 3], right=False)
        data['uncertain'] = pd.cut(data['uncertain'], bins=[0, 0.52646873, 1.05293746], labels=[0, 1], right=False)
        data['Num_intersection'] = pd.cut(data['Num_intersection'], bins=[0, 2, 5, 6], labels=[0, 1, 2], right=True)
        
        np.random.seed(seed)
        tasks = data['task'].unique()
        np.random.shuffle(tasks)
        train_tasks, test_tasks = tasks[:5], tasks[5:7]
        participants = data['participant'].unique()
        for participant in participants:
            test_data_filename = f"test_data_participant_{participant}.csv"
            model_filename = f"trained_model_participant_{participant}.pkl"
            participant_data = data[data['participant'] == participant]
            train_data = participant_data[participant_data['task'].isin(train_tasks)]
            test_data = participant_data[participant_data['task'].isin(test_tasks)]
            train_data = train_data.drop(columns=['participant', 'task'])
            data_t0 = train_data.rename(columns={col: (col, 0) for col in train_data.columns})
            data_t1 = train_data.shift(-1).rename(columns={col: (col, 1) for col in train_data.columns})
            complete_data = pd.concat([data_t0, data_t1], axis=1).dropna()
            network.fit(complete_data, estimator='MLE')
            with open(model_filename, "wb") as file:
                pickle.dump(network, file, protocol=pickle.HIGHEST_PROTOCOL)

            test_data = test_data.drop(columns=['participant', 'task'])
            test_data.to_csv(test_data_filename, index=False)

            acc = qzy.DBN_acc(network, test_data_filename, model_filename)
            accuracies.append(acc)
            # qzy.DBN_evaluate(network, test_data_filename, model_filename)
            # accuracies.append(qzy.DBN_acc(network, test_data_filename, model_filename))
            print(acc)

        print(f"The mean accuracy of T2 is: {mean(accuracies)}")
        print(f"The standard deviation of T2 is: {np.std(accuracies)}")


    def DBN_hyper(network, data_name="Agent_UpdatedTra_Simulation.csv", columns=['Time_Helpful_SignSeen', 'Num_intersection', 'uncertain', 'participant'], ran = 11):
        best_accuracy = 0
        best_combination = None
        num_bins_uncertain = 3
        dbn = network
        # Iterate over possible bin numbers for each feature with a leap of 5
        for num_bins_Time_Helpful_SignSeen in range(2, ran):
            for num_bins_Num_intersection in range(2, ran):
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
                bins = [num_bins_Time_Helpful_SignSeen, num_bins_Num_intersection, num_bins_uncertain]
                print(bins)
                # Train the model with the current bin configuration
                qzy.DBN_train(dbn, columns, bins, data_name)
                # Evaluate the model
                accuracy = qzy.DBN_acc(dbn)
                # Update best combination if current configuration is better
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_combination = bins
                print("done")
        print(f"Best bin combination: {best_combination} with accuracy: {best_accuracy}")

    def DBN_T3(network, columns = ['Time_Helpful_SignSeen', 'Num_intersection', 'uncertain', 'participant'], bins = [4, 3, 3],  data_name = "Agent_UpdatedTra_Simulation.csv", shuffled = False, seed = 123, model_name = "trained_model.pkl"):
        """
        Method to train a dbn network
        Participant with index > 130 are excluded
        Parameters
        - network: accepts a Dynamic Bayesian network defined by pgmpy
        - data name, default = "Agent_UpdatedTra_Simulation.csv": the name of data file
        - shuffled, default = False: parameter controling whether to shuffle the data by participant index. Default is not shuffle
        - seed, default = 123: sets numpy seeds for shuffling, default seed is 123
        """
        data = qzy.read_data(data_name, columns, bins)

        data = data[data['participant'] <= 130]

        if (shuffled == True):
            np.random.seed(seed)
            groups = [df for _, df in data.groupby('participant')]
            np.random.shuffle(groups)
            shuffled_data = pd.concat(groups).reset_index(drop=True)
            split_index = int(len(shuffled_data) * 0.8)
            train_data = shuffled_data[:split_index]
            test_data = shuffled_data[split_index:]
            train_data = train_data.drop(columns=['participant'])
            test_data = test_data.drop(columns=['participant'])
            data_t0 = train_data.rename(columns={col: (col, 0) for col in train_data.columns})
            data_t1 = train_data.shift(-1).rename(columns={col: (col, 1) for col in train_data.columns})
            complete_data = pd.concat([data_t0, data_t1], axis=1).dropna()
        else:
            np.random.seed(seed)
            shuffled_data = data
            split_index = int(len(shuffled_data) * 0.8)
            train_data = shuffled_data[:split_index]
            test_data = shuffled_data[split_index:]
            data_t0 = train_data.rename(columns={col: (col, 0) for col in train_data.columns})
            data_t1 = train_data.shift(-1).rename(columns={col: (col, 1) for col in train_data.columns})
            complete_data = pd.concat([data_t0, data_t1], axis=1).dropna()

        network.fit(complete_data, estimator='MLE')
        with open(model_name, "wb") as file:
            pickle.dump(network, file, protocol=pickle.HIGHEST_PROTOCOL)
        test_data.to_csv("test_data.csv", index=False)
        train_data.to_csv("train_data.csv", index=False)
        print("Training completed and model saved.")

    def DBN_T4(network, columns = ['Time_Helpful_SignSeen', 'Num_intersection', 'uncertain', 'participant'], bins = [4, 3, 3],  data_name = "Agent_UpdatedTra_Simulation.csv", shuffled = False, seed = 123, model_name = "trained_model.pkl"):
        """
        Method to train a dbn network
        Train with participant >= 34 && participant <=130
        Parameters
        - network: accepts a Dynamic Bayesian network defined by pgmpy
        - data name, default = "Agent_UpdatedTra_Simulation.csv": the name of data file
        - shuffled, default = False: parameter controling whether to shuffle the data by participant index. Default is not shuffle
        - seed, default = 123: sets numpy seeds for shuffling, default seed is 123
        """
        data = qzy.read_data(data_name, columns, bins)

        data = data[data['participant'] <= 130]

        train_data = data[data['participant'] > 34]
        test_data = data[data['participant'] <= 34]
        data_t0 = train_data.rename(columns={col: (col, 0) for col in train_data.columns})
        data_t1 = train_data.shift(-1).rename(columns={col: (col, 1) for col in train_data.columns})
        complete_data = pd.concat([data_t0, data_t1], axis=1).dropna()

        network.fit(complete_data, estimator='MLE')
        with open(model_name, "wb") as file:
            pickle.dump(network, file, protocol=pickle.HIGHEST_PROTOCOL)
        test_data.to_csv("test_data.csv", index=False)
        train_data.to_csv("train_data.csv", index=False)
        print("Training completed and model saved.")

    def DBN_T4_hyper(network, data_name="Agent_UpdatedTra_Simulation.csv", columns=['Time_Helpful_SignSeen', 'Num_intersection', 'uncertain', 'participant'], ran = 11):
        best_accuracy = 0
        best_combination = None
        num_bins_uncertain = 3
        dbn = network
        # Iterate over possible bin numbers for each feature with a leap of 5
        for num_bins_Time_Helpful_SignSeen in range(2, ran):
            for num_bins_Num_intersection in range(2, ran):
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
                bins = [num_bins_Time_Helpful_SignSeen, num_bins_Num_intersection, num_bins_uncertain]
                print(bins)
                # Train the model with the current bin configuration
                qzy.DBN_T4(dbn, columns, bins, data_name)
                # Evaluate the model
                accuracy = qzy.DBN_acc(dbn)
                # Update best combination if current configuration is better
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_combination = bins
                print("done")
        print(f"Best bin combination: {best_combination} with accuracy: {best_accuracy}")

    def Logit(network, columns=['Time_Helpful_SignSeen', 'Num_intersection', 'uncertain', 'participant'], bins=[4, 3, 3], data_name="Agent_UpdatedTra_Simulation.csv", shuffled=False, seed=123, model_name="trained_model.pkl"):
    
        data = qzy.read_data(data_name, columns, bins)
        data = data[data['participant'] <= 130]
        if (shuffled == True):
            np.random.seed(seed)
            groups = [df for _, df in data.groupby('participant')]
            np.random.shuffle(groups)
            shuffled_data = pd.concat(groups).reset_index(drop=True)
            split_index = int(len(shuffled_data) * 0.8)
            train_data = shuffled_data[:split_index]
            test_data = shuffled_data[split_index:]
            train_data = train_data.drop(columns=['participant'])
            test_data = test_data.drop(columns=['participant'])
        else:
            np.random.seed(seed)
            shuffled_data = data
            split_index = int(len(shuffled_data) * 0.8)
            train_data = shuffled_data[:split_index]
            test_data = shuffled_data[split_index:]

        X_train = train_data[['Time_Helpful_SignSeen', 'Num_intersection']]
        y_train = train_data['uncertain']
        X_test = test_data[['Time_Helpful_SignSeen', 'Num_intersection']]
        y_test = test_data['uncertain']

        model = LogisticRegression(random_state=seed)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        cm = confusion_matrix(y_test, predictions)
        TP = cm[1, 1]
        FN = cm[1, 0]
        
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0

        # qzy.plot_confusion_matrix(y_test, predictions, [0,1], "Logistic_Regression")
        # qzy.plot_ground_truth(y_test, predictions, "Logistic_Regression")
        
        return accuracy, sensitivity

    def RandomForestModel(network, columns=['Time_Helpful_SignSeen', 'Num_intersection', 'uncertain', 'participant'], bins=[4, 3, 3], data_name="Agent_UpdatedTra_Simulation.csv", shuffled=False, seed=123, model_name="trained_model.pkl"):
    
        data = qzy.read_data(data_name, columns, bins)
        data = data[data['participant'] <= 130]
        if shuffled:
            np.random.seed(seed)
            groups = [df for _, df in data.groupby('participant')]
            np.random.shuffle(groups)
            shuffled_data = pd.concat(groups).reset_index(drop=True)
            split_index = int(len(shuffled_data) * 0.8)
            train_data = shuffled_data[:split_index]
            test_data = shuffled_data[split_index:]
            train_data = train_data.drop(columns=['participant'])
            test_data = test_data.drop(columns=['participant'])
        else:
            np.random.seed(seed)
            shuffled_data = data
            split_index = int(len(shuffled_data) * 0.8)
            train_data = shuffled_data[:split_index]
            test_data = shuffled_data[split_index:]

        X_train = train_data[['Time_Helpful_SignSeen', 'Num_intersection']]
        y_train = train_data['uncertain']
        X_test = test_data[['Time_Helpful_SignSeen', 'Num_intersection']]
        y_test = test_data['uncertain']

        model = RandomForestClassifier(random_state=seed)  # Use RandomForestClassifier here
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        cm = confusion_matrix(y_test, predictions)
        TP = cm[1, 1]
        FN = cm[1, 0]

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        
        # qzy.plot_confusion_matrix(y_test, predictions, [0,1], "Random_Forest")
        # qzy.plot_ground_truth(y_test, predictions, "Random_Forest")
        
        return accuracy, sensitivity

    def compare_models():
        i = 1001
        DBN_accs = []
        LOG_accs = []
        FOR_accs = []
        DBN_sen = []
        LOG_sen = []
        FOR_sen = []
        number_of_iterations = 100
        maxn = 1000+number_of_iterations
        dbn = DBN()

        while i <= maxn:
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

            qzy.DBN_train(network=dbn, data_name="Agent_UpdatedTra_Simulation.csv",bins = [4, 3, 3], shuffled=True, seed=i)
            DBN_accs, DBN_sen.append(qzy.DBN_acc(dbn))
            LOG_accs, LOG_sen.append(qzy.Logit(network=dbn, data_name="Agent_UpdatedTra_Simulation.csv",bins = [4, 3, 3], shuffled=True, seed=i))
            FOR_accs, FOR_sen.append(qzy.RandomForestModel(network=dbn, data_name="Agent_UpdatedTra_Simulation.csv",bins = [4, 3, 3], shuffled=True, seed=i))
            print(str(i-1000) + "/" + str(number_of_iterations))
            i += 1

        plt.figure(figsize=(10, 6))

        qzy.plot_normal_curve(DBN_accs, 'DBN', 'blue')
        qzy.plot_normal_curve(LOG_accs, 'Logistic Regression', 'red')
        qzy.plot_normal_curve(FOR_accs, 'Random Forest', 'green')

        plt.title('model comparison')
        plt.xlabel('Accuracy')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.savefig("DBNVSLOG.png")


        plt.figure(figsize=(10, 6))

        qzy.plot_normal_curve(DBN_sen, 'DBN', 'blue')
        qzy.plot_normal_curve(LOG_sen, 'Logistic Regression', 'red')
        qzy.plot_normal_curve(FOR_sen, 'Random Forest', 'green')

        plt.title('model sensitivity comparison')
        plt.xlabel('Sensitivity')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.savefig("Compare_sensitivity.png")
