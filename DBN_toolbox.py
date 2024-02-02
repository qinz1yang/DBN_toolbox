# v1.1
# 2024.2.2, Friday
# Ithaca, Cloudy

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
# plot
import matplotlib.pyplot as plt
import scipy.stats as stats


class qzy():
    """
    qzy's tools for training and evaluating DBN networks
    """
    def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
        """
        Method to plot confusion matrix
        """
        confusion_matrix_filename = "comfusion_matrix.png"
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
    
    def plot_normal_distribution(data):
        median = np.median(data)
        sd = np.std(data)
        print(f"Median: {median}")
        print(f"Standard Deviation: {sd}")
        plt.figure(figsize=(10, 6))
        density = stats.gaussian_kde(data)
        xs = np.linspace(min(data), max(data), 200)
        plt.plot(xs, density(xs), label='Density')
        plt.title('Normal Distribution of Data')
        plt.xlabel('Data')
        plt.ylabel('Density')
        plt.legend()
        plt.show()

    def DBN_train(network, data_name = "Agent_UpdatedTra_Simulation.csv", shuffled = False, seed = 123):
        """
        Method to train a dbn network
        """

        data = pd.read_csv(data_name, usecols=['Time_Helpful_SignSeen', 'Num_intersection', 'uncertain', 'participant'])
        data['Time_Helpful_SignSeen'] = pd.cut(data['Time_Helpful_SignSeen'], bins=[0, 8, 85, 336, 9999], labels=[0, 1, 2, 3], right=False)
        data['uncertain'] = pd.cut(data['uncertain'], bins=[0, 0.52646873, 1.05293746], labels=[0, 1], right=False)
        data['Num_intersection'] = pd.cut(data['Num_intersection'], bins=[0, 2, 5, 6], labels=[0, 1, 2], right=True)
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
            # print(train_data)
            # np.random.seed(seed)
            # shuffled_data = data.sample(frac=1).reset_index(drop=True)
            # split_index = int(len(shuffled_data) * 0.8)
            # train_data = shuffled_data[:split_index]
            # test_data = shuffled_data[split_index:]
            # data_t0 = train_data.rename(columns={col: (col, 0) for col in train_data.columns})
            # data_t1 = train_data.shift(-1).rename(columns={col: (col, 1) for col in train_data.columns})
            # complete_data = pd.concat([data_t0, data_t1], axis=1).dropna()
        else:
            shuffled_data = data
            split_index = int(len(shuffled_data) * 0.8)
            train_data = shuffled_data[:split_index]
            test_data = shuffled_data[split_index:]
            data_t0 = train_data.rename(columns={col: (col, 0) for col in train_data.columns})
            data_t1 = train_data.shift(-1).rename(columns={col: (col, 1) for col in train_data.columns})
            complete_data = pd.concat([data_t0, data_t1], axis=1).dropna()

        network.fit(complete_data, estimator='MLE')
        with open("trained_model.pkl", "wb") as file:
            pickle.dump(network, file, protocol=pickle.HIGHEST_PROTOCOL)
        test_data.to_csv("test_data.csv", index=False)
        train_data.to_csv("train_data.csv", index=False)
        print("Training completed and model saved.")

    def DBN_evaluate(network, test_data_name = "test_data.csv", model_name = "trained_model.pkl"):
        """
        Method to evalue the DBN model based on test data given
        """
        confusion_matrix_filename = "comfusion_matrix.png"

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
        precision = precision_score(true_labels, predictions, average='weighted')
        recall = recall_score(true_labels, predictions, average='weighted')
        f1 = f1_score(true_labels, predictions, average='weighted')

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        qzy.plot_confusion_matrix(true_labels, predictions, classes)

        html_filename = 'model_evaluation_results.html'
        with open(html_filename, 'w') as f:
            f.write(f"<h1>Model Evaluation Results</h1>")
            f.write(f"<p>Model Name: <b>{model_name}</b></p>")
            f.write(f"<p>Test data name: <b>{test_data_name}</b></p>")
            f.write(f"<p>Accuracy: {accuracy}</p>")
            f.write(f"<p>Confusion Matrix:</p>")
            f.write(f'<img src="{confusion_matrix_filename}"><br>')
            f.write(f"<p>Accuracy: {accuracy}</p>")
            f.write(f"<p>Precision: {precision}</p>")
            f.write(f"<p>Recall: {recall}</p>")
            f.write(f"<p>F1 Score: {f1}</p>")
        webbrowser.open('file://' + os.path.realpath(html_filename))

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
