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
import time
# plot
import matplotlib.pyplot as plt
import scipy.stats as stats

# logistic regression
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

import itertools

import threading
from queue import Queue

from concurrent.futures import ThreadPoolExecutor, as_completed
import copy


class qzy():
    """
    qzy's tools for training and evaluating DBN networks
    """

    def DBN_ini(variables_to_add = ['Num_intersection', 'Time_Helpful_SignSeen', 'Circularity', 'Occlusivity', 'Elongation', 'DriftAngle','Visible_All_Sign', 'Visible_Helpful_Sign', 'Closest_Helpful_Dist','Jagged_360', 'sbsod', 'age']):
        dbn = DBN()
        dbn.clear()
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

    def plot_normal_distribution(data, name="given model"):
        """
        Method to plot normal distribution based on given data
        Parameters
        - data
        """
        data = np.array(data)
        median = np.median(data)
        sd = np.std(data)
        print(f"Statistics for {name}: ")
        print(f"Median: {median}")
        print(f"Standard Deviation: {sd}")
        plt.figure(figsize=(10, 6))
        density = stats.gaussian_kde(data)
        xs = np.linspace(min(data), max(data), 200)
        plt.plot(xs, density(xs), label='Density')

        plt.axvline(x=median, color='r', linestyle='--', label=f"Median={median:.4f}")
        plt.axvline(x=median-sd, color='g', linestyle=':', label=f"Median - 1 SD({sd:.4f})")
        plt.axvline(x=median+sd, color='g', linestyle=':', label=f"Median + 1 SD({sd:.4f})")
    
        plt.title(f'Normal Distribution of {name} Data')
        plt.xlabel('Data')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(f'{name}_normal_distribution.png')

    def plot_ground_truth(true_labels, predictions, model_name):
        plt.figure(figsize=(100, 6))
        plt.plot(true_labels, label='Ground Truth', linestyle='-', color='blue', linewidth=1)
        print("Plot_ground_truth in progress:")
        for i, (true, pred) in enumerate(zip(true_labels, predictions)):
            if true == pred:
                plt.scatter(i, true, color='red', label='Correct Prediction' if i == 0 else "")
            # else:
            #     plt.scatter(i, true, color='blue', label='Ground Truth' if i == 0 else "")
            # print(i)
            print(str(i+1) + "/" + str(len(true_labels)), end = "\r")

        print("Plot_ground_truth done")
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
        if 'participant' not in local_columns:
            local_columns.append('participant')
        print(f"Read following columns :{local_columns}")
        data = pd.read_csv(data_name, usecols=local_columns)
        thresholds = {}
        i = 0
        while i < len(local_columns):
            if local_columns[i] == 'participant' or local_columns[i] == 'task':
                i += 1
            else:
                data[local_columns[i]], bins_edges = pd.qcut(data[local_columns[i]], q=bins[i], labels=False, duplicates='drop', retbins=True)
                thresholds[local_columns[i]] = bins_edges
                print(f"Thresholds for {local_columns[i]}: {bins_edges}")
                i += 1
        return data


    def DBN_train(network, columns = ['Time_Helpful_SignSeen', 'Num_intersection', 'uncertain', 'participant'], bins = [4, 3, 3],  data_name = "Agent_UpdatedTra_Simulation.csv", shuffled = False, seed = 123, model_name = "trained_model.pkl", fast=False):
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
        
        if (fast == False):
            with open(model_name, "wb") as file:
                pickle.dump(network, file, protocol=pickle.HIGHEST_PROTOCOL)
            test_data.to_csv("test_data.csv", index=False)
            train_data.to_csv("train_data.csv", index=False)
        return test_data, network
        print("Training completed and model saved.")
        

    def DBN_evaluate(network=None, test_data=None, test_data_name="test_data.csv", model_name="trained_model.pkl", variables_to_add = ['Time_Helpful_SignSeen', 'Num_intersection']):
        """
        Method to evaluate the DBN model based on test data given.
        Parameters:
        - network: accepts a Dynamic Bayesian network defined by pgmpy
        - test data name, default = "test_data.csv": name of test data file
        - model name, default = "trained_model.pkl": name of trained model
        """
        confusion_matrix_filename = f"{model_name}_comfusion_matrix.png"
        classes = [0, 1]

        if(network==None):
            with open(model_name, "rb") as file:
                dbn = pickle.load(file)
        else:
            dbn = network

        if(test_data is None):
            local_data = pd.read_csv(test_data_name)
        else:
            local_data = test_data

        
        local_variables = variables_to_add
        dbn_inference = DBNInference(dbn)
        available_vars = [var for var in local_variables if var in local_data.columns]
        print(f"available vars:{available_vars}")
        print("DBN_fast_acc_and_sensitivity prediction in progress:")
        
        actual_values = local_data['uncertain'].iloc[1:].values
        
        predictions = []
        for i, (_, current_row) in enumerate(local_data.iloc[:-1].iterrows()):
            evidence = {}
            for var in local_variables:
                if var in current_row:
                    evidence[(var, 0)] = current_row[var]
            prediction = dbn_inference.forward_inference([('uncertain', 1)], evidence=evidence)
            most_confident_prediction = np.argmax(prediction[('uncertain', 1)].values)
            predictions.append(most_confident_prediction)
    
        print("DBN_evaluate prediction done")
        accuracy = accuracy_score(actual_values, predictions)
        precision = precision_score(actual_values, predictions, labels=classes, average='weighted')
        recall = recall_score(actual_values, predictions, labels=classes, average='weighted')
        f1 = f1_score(actual_values, predictions, labels=classes, average='weighted')
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        qzy.plot_confusion_matrix(actual_values, predictions, classes, model_name)
        qzy.plot_ground_truth(actual_values, predictions, model_name)

        
        html_filename = f'{model_name}_results.html'
        print("Writing report to " + html_filename + "...")
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

    def DBN_acc(network, test_data_name = "test_data.csv", model_name = "trained_model.pkl", variables_to_add = ['Time_Helpful_SignSeen', 'Num_intersection']):
        with open(model_name, "rb") as file:
            dbn = pickle.load(file)
        test_data = pd.read_csv(test_data_name)
        dbn_inference = DBNInference(dbn)
        predictions = []
        true_labels = []
        classes = [0, 1]
        available_vars = [var for var in variables_to_add if var in test_data.columns]
        print(f"avaliable vars:{available_vars}")
        print("DBN_acc prediction in progress:")
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
            # print(str(i+1) + "/" + str(len(test_data)), end = "\r")
        
        print("DBN_evaluate prediction done")
        return(accuracy_score(true_labels, predictions))
        
    def DBN_acc_and_sensitivity(network, test_data_name = "test_data.csv", model_name = "trained_model.pkl", variables_to_add = ['Time_Helpful_SignSeen', 'Num_intersection']):
        """
        Returns the accuracy and sensitivity based on reading data.
        """
        with open(model_name, "rb") as file:
            dbn = pickle.load(file)
        test_data = pd.read_csv(test_data_name)
        dbn_inference = DBNInference(dbn)
        predictions = []
        true_labels = []
        classes = [0, 1]
        available_vars = [var for var in variables_to_add if var in test_data.columns]
        print(f"avaliable vars:{available_vars}")
        print("DBN_acc_and_sensitivity prediction in progress:")
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
            # print(str(i+1) + "/" + str(len(test_data)), end = "\r")

        accuracy = accuracy_score(true_labels, predictions)

        cm = confusion_matrix(true_labels, predictions)
        TP = cm[1, 1]
        FN = cm[1, 0]

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0

        return accuracy, sensitivity

    def DBN_fast_acc_and_sensitivity(network, test_data, variables_to_add = ['Time_Helpful_SignSeen', 'Num_intersection']):
        """
        Fast Methods does not read from permenant files, it accepts variables passed in.
        """

        local_network = network
        local_data = test_data
        local_variables = variables_to_add
        dbn_inference = DBNInference(local_network)
        available_vars = [var for var in local_variables if var in local_data.columns]
        print(f"available vars:{available_vars}")
        print("DBN_fast_acc_and_sensitivity prediction in progress:")
        
        actual_values = local_data['uncertain'].iloc[1:].values

        predictions = []
        for i, (_, current_row) in enumerate(local_data.iloc[:-1].iterrows()):

            evidence = {}
            for var in local_variables:
                if var in current_row:
                    evidence[(var, 0)] = current_row[var]
            prediction = dbn_inference.forward_inference([('uncertain', 1)], evidence=evidence)
            most_confident_prediction = np.argmax(prediction[('uncertain', 1)].values)
            predictions.append(most_confident_prediction)
            if((i % 1000) == 0):
                print(f"{i} / {len(local_data-1)}")

        accuracy = accuracy_score(actual_values, predictions)
        sensitivity = recall_score(actual_values, predictions) 

        return accuracy, sensitivity

    def DBN_T1(network, number_of_iterations = 20, predictors=['Time_Helpful_SignSeen', 'Num_intersection'], variables=['Time_Helpful_SignSeen', 'Num_intersection', 'uncertain'], bins=[4,3,3]):
        start_time = time.time()
        i = 1001
        accs = []
        senses = []
        sums = []
        maxn = 1000+number_of_iterations
        highest_accuracy = -999
        highest_sensitivity = -999
        highest_sum = -999
        highest_accuracy_seed = -999
        highest_sensitivity_seed = -999
        highest_sum_seed = -999
        
        while i <= maxn:
            network.clear()
            network = qzy.DBN_ini(predictors)
            test_data, network = qzy.DBN_train(network, columns=variables, data_name="Agent_UpdatedTra_Simulation.csv", bins = bins, shuffled=True, seed=i, fast=True)
            print(str(i-1000) + "/" + str(number_of_iterations))
            accuracy, sensitivity = qzy.DBN_fast_acc_and_sensitivity(network, test_data, variables_to_add = predictors)
            print(f"Accuracy: {accuracy} Sensitivity: {sensitivity} Seed: {i}")
            if (accuracy > highest_accuracy):
                highest_accuracy = accuracy
                highest_accuracy_seed = i

            if (sensitivity > highest_sensitivity):
                highest_sensitivity = sensitivity
                highest_sensitivity_seed = i

            if (accuracy+sensitivity > highest_sum):
                highest_sum = sensitivity+accuracy
                highest_sum_seed = i

            accs.append(accuracy)
            senses.append(sensitivity)
            sums.append(accuracy+sensitivity)
            i += 1

        print(f"starting with seed 1001 and end with seed {i}")
        print(f"highest accuracy: {highest_accuracy}")
        print(f"seed with highest accuracy: {highest_accuracy_seed}")
        qzy.plot_normal_distribution(accs)
        qzy.plot_normal_distribution(senses)
        qzy.plot_normal_distribution(sums)

        end_time = time.time()
        print(f"Time consumed: {(end_time - start_time) / 3600:.4f} hours")

    def DBN_T1_worker(start_seed, end_seed, network_template, predictors, variables, bins, results_queue):
        local_network = network_template.copy()
        local_predictors = predictors
        local_variables = variables
        for i in range(start_seed, end_seed):
            local_network.clear()
            local_network = qzy.DBN_ini(local_predictors)
            print(f"{i-start_seed+1} / {end_seed-start_seed}")
            test_data, local_network = qzy.DBN_train(local_network, columns=local_variables, data_name="Agent_UpdatedTra_Simulation.csv", bins=bins, shuffled=True, seed=i, fast=True)
            accuracy, sensitivity = qzy.DBN_fast_acc_and_sensitivity(local_network, test_data, variables_to_add=local_predictors)
            results_queue.put({
                'seed': i,
                'accuracy': accuracy,
                'sensitivity': sensitivity,
                'sum': accuracy + sensitivity
            })

    def DBN_T1_multithreaded(network, number_of_iterations=20, predictors=['Time_Helpful_SignSeen', 'Num_intersection'], variables=['Time_Helpful_SignSeen', 'Num_intersection', 'uncertain'], bins=[4,3,3], num_threads=4):
        start_time = time.time()

        threads = []
        results_queue = Queue()
        
        accs = []
        senses = []
        sums = []

        iterations_per_thread = number_of_iterations // num_threads
        for t in range(num_threads):
            local_network = network
            start_seed = 1001 + t * iterations_per_thread
            if t == num_threads - 1:
                end_seed = 1001 + number_of_iterations
            else:
                end_seed = start_seed + iterations_per_thread
            thread = threading.Thread(target=qzy.DBN_T1_worker, args=(start_seed, end_seed, local_network, predictors, variables, bins, results_queue))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()

        highest_accuracy = -999
        highest_sensitivity = -999
        highest_sum = -999
        highest_accuracy_seed = -999
        highest_sensitivity_seed = -999
        highest_sum_seed = -999

        while not results_queue.empty():
            result = results_queue.get()
            if result['accuracy'] > highest_accuracy:
                highest_accuracy = result['accuracy']
                highest_accuracy_seed = result['seed']
            if result['sensitivity'] > highest_sensitivity:
                highest_sensitivity = result['sensitivity']
                highest_sensitivity_seed = result['seed']
            if result['sum'] > highest_sum:
                highest_sum = result['sum']
                highest_sum_seed = result['seed']

            accs.append(result['accuracy'])
            senses.append(result['sensitivity'])
            sums.append(result['sum'])

        print(f"Highest Accuracy: {highest_accuracy}, Seed: {highest_accuracy_seed}")
        print(f"Highest Sensitivity: {highest_sensitivity}, Seed: {highest_sensitivity_seed}")
        print(f"Highest Sum: {highest_sum}, Seed: {highest_sum_seed}")

        qzy.plot_normal_distribution(accs, name="accuracies")
        qzy.plot_normal_distribution(senses, name="sensitivity")
        qzy.plot_normal_distribution(sums, name="sums")

        end_time = time.time()
        print(f"Time consumed: {(end_time - start_time) / 3600:.6f} hours")

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


    def DBN_hyper(network, data_name="Agent_UpdatedTra_Simulation.csv", columns=['Time_Helpful_SignSeen', 'Num_intersection', 'uncertain', 'participant'], predictors=['Time_Helpful_SignSeen', 'Num_intersection'], bins=[4,3,3], ran = 11, seed=1056):
        start_time = time.time()

        best_sum = 0
        best_accuracy=0
        best_sensitivity=0
        accuracy = 0
        sensitivity = 0
        best_sum_combination = None
        best_accuracy_combination = None
        best_sensitivity_combination = None
        
        local_network = qzy.DBN_ini(predictors)
        # Exclude 'uncertain' and 'participant' from the variable bin range generation
        variable_columns = [col for col in columns if col not in ['uncertain', 'participant']]

        # Generate all possible bin combinations within the range for each variable column
        all_bin_combinations = list(itertools.product(*(range(3, ran+1) for _ in variable_columns)))

        for bin_combination in all_bin_combinations:
            local_network.clear()
            local_network = qzy.DBN_ini(predictors)
            # Construct the full bin list including the fixed bin count for 'uncertain' as 3
            full_bins = list(bin_combination) + [3]  # Appending fixed bin count for 'uncertain'
            
            print("Training with bins:", full_bins)
            qzy.DBN_train(local_network, columns=columns, data_name=data_name, bins=full_bins, shuffled=True, seed=seed)
            
            accuracy, sensitivity = qzy.DBN_acc_and_sensitivity(local_network, variables_to_add=predictors)
            if accuracy+sensitivity > best_sum:
                best_sum = accuracy+sensitivity
                best_sum_combination = full_bins  # Store the full bin combination including 'uncertain'

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_accuracy_combination = full_bins

            if sensitivity > best_sensitivity:
                best_sensitivity = sensitivity
                best_sensitivity_combination = full_bins
            
            print("Iteration done with bins:", full_bins, "Accuracy:", accuracy, "Sensitivity:", sensitivity)
        
        # Ensure the best combination is displayed correctly
        if best_sum_combination is not None:
            print(f"Best sum bin combination: {best_sum_combination} with sum: {best_sum}")
            print(f"Best accuracy bin combination: {best_accuracy_combination} with accuracy: {best_accuracy}")
            print(f"Best sensitivity bin combination: {best_sensitivity_combination} with sensitivity: {best_sensitivity}")
        else:
            print("No best combination found.")

        end_time = time.time()
        print(f"Time consumed: {(end_time - start_time) / 3600:.4f} hours")


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

    def DBN_train_and_evaluate_by_task(network, data_name="Agent_UpdatedTra_Simulation.csv", columns=['Time_Helpful_SignSeen', 'Num_intersection', 'uncertain', 'task'], predictors=['Time_Helpful_SignSeen', 'Num_intersection'], bins=[4, 3, 3], shuffled=True, model_name="trained_model.pkl", seed=123):
        main_directory = "evaluation_results"
        os.makedirs(main_directory, exist_ok=True)
        test_data_directory = os.path.join(main_directory, f"seed_{seed}")
        os.makedirs(test_data_directory, exist_ok=True)
        
        np.random.seed(seed)
        local_columns = columns.copy()  # Fix: Make a copy of the columns list to prevent modifying the input list
        local_columns.append('task')
        data = qzy.read_data(data_name, local_columns, bins)

        evaluation_results = []

        if shuffled:
            np.random.seed(seed)
            groups = [df for _, df in data.groupby('participant')]
            np.random.shuffle(groups)
            shuffled_data = pd.concat(groups).reset_index(drop=True)
        else:
            shuffled_data = data
        
        split_index = int(len(shuffled_data) * 0.8)
        train_data = shuffled_data.iloc[:split_index]
        test_data = shuffled_data.iloc[split_index:]
        train_data = train_data.drop(columns=['participant', 'task'])
        data_t0 = train_data.rename(columns={col: (col, 0) for col in train_data.columns})
        data_t1 = train_data.shift(-1).rename(columns={col: (col, 1) for col in train_data.columns})
        complete_data = pd.concat([data_t0, data_t1], axis=1).dropna()

        network.fit(complete_data, estimator='MLE')

        for (participant, task), group in test_data.groupby(['participant', 'task']):
            group = group.drop(columns=['participant', 'task'])
            filename = os.path.join(test_data_directory, f"participant_{participant}_task_{task}.csv")
            group.to_csv(filename, index=False)
            accuracy, sensitivity = qzy.DBN_fast_acc_and_sensitivity(network, group, variables_to_add=predictors)
            evaluation_results.append([seed, participant, task, accuracy, sensitivity])

        results_df = pd.DataFrame(evaluation_results, columns=['Seed', 'Participant', 'Task', 'Accuracy', 'Sensitivity'])

        # # Save the trained model
        # with open(model_name, "wb") as file:
        #     pickle.dump(network, file, protocol=pickle.HIGHEST_PROTOCOL)
        
        return results_df
        
    def DBN_evaluate_over_seed_range(network, seed_start, seed_end, **kwargs):

        futures_with_seeds = []

        with ThreadPoolExecutor() as executor:
            for seed in range(seed_start, seed_end + 1):
                local_network = copy.deepcopy(network)
                future = executor.submit(qzy.DBN_train_and_evaluate_by_task, local_network, seed=seed, **kwargs)
                futures_with_seeds.append((future, seed))

        results_with_seeds = []
        for future, seed in futures_with_seeds:
            try:
                result = future.result()
                results_with_seeds.append((seed, result))
            except Exception as exc:
                print(f"An error occurred for seed {seed}: {exc}")

        results_with_seeds.sort(key=lambda x: x[0])  # Sort by seed, which is the first element in each tuple

        sorted_results = [result for _, result in results_with_seeds]

        combined_results_df = pd.concat(sorted_results)
        combined_results_df.to_csv("combined_evaluation_results.csv", index=False)

        print("Done.")

    def DBN_predictor_impact(data_name="Agent_UpdatedTra_Simulation.csv", columns=['Time_Helpful_SignSeen', 'Num_intersection', 'uncertain', 'participant'], predictors=['Time_Helpful_SignSeen', 'Num_intersection'], bins=[4, 3, 3], seed=123):
        """
        Evaluates the impact of each predictor on the DBN model by calculating the change in the sum of accuracy and sensitivity
        when excluding each predictor one at a time.
        
        Parameters:
        - data_name: CSV file containing the data.
        - columns: List of all columns to be included in the model.
        - predictors: List of predictor variables.
        - bins: The bin counts for discretization of each variable.
        - seed: Seed for random number generation, used in data shuffling.
        """
        impact_results = {}
        dbn_baseline = qzy.DBN_ini(variables_to_add=predictors)
        _, dbn_baseline = qzy.DBN_train(dbn_baseline, columns=columns, data_name=data_name, bins=bins, shuffled=True, seed=seed, fast=True)
        baseline_accuracy, baseline_sensitivity = qzy.DBN_fast_acc_and_sensitivity(dbn_baseline, _, variables_to_add=predictors)
        baseline_sum = baseline_accuracy + baseline_sensitivity

        for predictor in predictors:
            temp_predictors = [p for p in predictors if p != predictor]
            dbn_modified = qzy.DBN_ini(variables_to_add=temp_predictors)
            _, dbn_modified = qzy.DBN_train(dbn_modified, columns=columns, data_name=data_name, bins=bins, shuffled=True, seed=seed, fast=True)
            modified_accuracy, modified_sensitivity = qzy.DBN_fast_acc_and_sensitivity(dbn_modified, _, variables_to_add=temp_predictors)
            modified_sum = modified_accuracy + modified_sensitivity
            impact = baseline_sum - modified_sum
            impact_results[predictor] = impact

        predictors = list(impact_results.keys())
        impacts = list(impact_results.values())

        plt.figure(figsize=(12, 8))
        bars = plt.bar(predictors, impacts, color='skyblue')
        plt.xlabel('Predictors')
        plt.ylabel('Impact on Model Performance')
        plt.title('Impact of Each Predictor on the DBN Model Performance')
        plt.xticks(rotation=45, ha="right")

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, f"{yval:.2f}", va='bottom', ha='center')

        plt.tight_layout()
        plt.savefig("impact_results.png")
        return impact_results
