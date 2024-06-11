The `DBN_toolbox` library facilitates the training and evaluation of Dynamic Bayesian Networks (DBNs) with a focus on handling time-series data and complex variable relationships. This documentation provides detailed guidance on how to utilize the library, including example usage for various functions.
### Installation
Ensure Python is installed along with necessary packages: `pandas`, `pgmpy`, `numpy`, `matplotlib`, `scipy`, `pickle`, and `sklearn`. Install these using pip:
```bash
pip install pandas pgmpy numpy matplotlib scipy pickle sklearn
```
### Core Functionalities
#### Training a DBN
Training involves initializing and fitting a DBN model with given data. This process may include shuffling and setting seeds for reproducibility.

```python
import multiprocessing
from DBN_toolbox import qzy

# Detect the number of CPU cores available
cores = multiprocessing.cpu_count()
print(f"Core number: {cores}")

# Define variables and their bins
variables = ['Num_intersection', 'Time_Helpful_SignSeen', 'sbsod', 'framecount', 'uncertain']
predictors = ['Num_intersection', 'Time_Helpful_SignSeen', 'sbsod', 'framecount']
local_bins = [3, 4, 4, 5, 3]

# Initialize the DBN with predictors
network = qzy.DBN_ini(predictors)

# Train the DBN
test_data, network = qzy.DBN_train(network, columns=variables, data_name="Agent_UpdatedTra_Simulation.csv", bins=local_bins, shuffled=True, seed=3524, fast=False)
```
#### Evaluating the Model
After training, the user can evaluate the DBN's performance using metrics such as accuracy and sensitivity.

```python
# Evaluate the trained DBN
qzy.DBN_evaluate(network=network, test_data=test_data, variables_to_add=predictors)
```
### Advanced Usage

#### Multithreading for Intensive Computations
Leverage multiple CPU cores to handle intensive computations across several iterations or tasks.

```python
# Multithreaded T1 experiment
qzy.DBN_T1_multithreaded(network, number_of_iterations=500, num_threads=cores, predictors=predictors, variables=variables, bins=local_bins)
```
#### Specialized Training and Evaluation Functions
Utilize tailored functions for specific scenarios like participant-specific models or hyperparameter tuning.

```python
# Task-specific model training and evaluation
qzy.DBN_train_and_evaluate_by_task(network, columns=variables, data_name="Agent_UpdatedTra_Simulation.csv", predictors=predictors, bins=local_bins, seed=3524)

# Hyperparameter tuning across different seed ranges
qzy.DBN_evaluate_over_seed_range(network=network, seed_start=1001, seed_end=1999, columns=variables, data_name="Agent_UpdatedTra_Simulation.csv", predictors=predictors, bins=local_bins)
```
### Visualization and Comparison
#### Visualizing Model Accuracy and Sensitivity
Generate visual representations of the model's performance to better understand its behavior.
```python
# Plot distribution of model accuracy and sensitivity
qzy.compare_models()
```
#### Analyzing Predictor Impact
Evaluate how different predictors impact the overall performance of the DBN model.

```python
# Analyze the impact of each predictor on model performance
impact_results = qzy.DBN_predictor_impact(data_name="Agent_UpdatedTra_Simulation.csv", columns=variables, predictors=predictors, bins=local_bins, seed=3524)
```
# Function-specific explanations

## DBN_ini
The `DBN_ini` function is designed to initialize a Dynamic Bayesian Network (DBN) within the `DBN_toolbox` library. This function sets up the structure of the DBN based on the input variables specified by the user, defining how these variables are connected across different time slices.
### Purpose
The primary purpose of the `DBN_ini` function is to create a new DBN model that can later be trained with data. This involves:
- Setting up the nodes (variables) and edges (relationships between nodes) within the network.
- Ensuring that the network adheres to temporal dynamics, i.e., how variables at one time point influence variables at the next time point.
### Parameters
- **variables_to_add**: A list of variable names that are included in the DBN. These are the variables for which the relationships will be defined within the network.
### Process
1. **Initialize the DBN**: The function starts by creating an instance of `DynamicBayesianNetwork`, which is provided by the `pgmpy` library. This instance serves as the base for adding nodes and edges.
2. **Clear Existing Network**: If there's an existing DBN structure, it is cleared to ensure no residual data or structures interfere with the new network setup.
3. **Define Edges**:
    - The function iteratively adds edges between the variables listed in `variables_to_add` and a uncertain to be predicted.
    - For each variable, edges are defined across two time slices, T0 (current time) and T1 (next time). This means for a variable `X`, the function sets up edges from `(X, 0)` to `(uncertain, 0)` and `(X, 1)` to `(uncertain, 1)`, where `0` and `1` denote the time slices.
4. **Check Model Validity**: After setting up the edges, the function checks if the DBN model is valid using `dbn.check_model()`. This method verifies if the defined structure adheres to the principles of Bayesian networks, such as having no cyclic dependencies and maintaining consistent parental information across time slices.
5. **Return the DBN**: If the model is valid, it returns the initialized DBN model. If there are issues (e.g., an invalid structure), it raises an error or returns a message indicating the problem.



## plot_confusion_matrix

### Purpose
The `plot_confusion_matrix` function in the `DBN_toolbox` library is designed to visualize the performance of a classification model by showing the confusion matrix in a graphical format.
### Parameters
- **y_true**: The true labels from the test set.
- **y_pred**: The predictions made by the model.
- **classes**: A list of class names, representing the possible outcomes.
- **model_name**: A string specifying the name of the model, used to name the output file.
- **title** (optional): The title for the plot; defaults to "Confusion Matrix".
- **cmap** (optional): The colormap used for the plot; defaults to `plt.cm.Blues`.
### Process
1. **Generate Confusion Matrix**: Calculate the confusion matrix using `confusion_matrix(y_true, y_pred)` from `sklearn.metrics`. This matrix quantifies the number of times each expected class was predicted correctly or incorrectly.
2. **Create Plot**:
    - Set up a matplotlib figure and use `imshow` to display the confusion matrix as an image.
    - Apply the chosen colormap to differentiate between different values clearly.
3. **Add Plot Enhancements**:
    - Include color bar for reference.
    - Set x and y labels to class names, adjusting ticks accordingly.
    - Annotate the matrix with actual data counts. Adjust text color for readability based on the background.
4. **Save Plot**:
    - Save the generated plot as an image file named using the `model_name` to easily reference back to the specific model evaluated.


## plot_normal_curve
### Purpose
The `plot_normal_curve` function in the `DBN_toolbox` is used to plot the normal distribution curve for a given dataset. This helps visualize how the data is distributed around the mean and can be used to assess the normality of the data.
### Parameters
- **data**: Array or list of numerical values to plot.
- **label**: Description or label for the dataset.
- **color**: Color of the plot line.
### Process
1. **Calculate Statistics**: Compute the mean and standard deviation of the data.
2. **Generate Points**: Create a range of x values from mean minus three times the standard deviation to mean plus three times the standard deviation.
3. **Compute Normal Distribution**: Calculate the y values using the normal distribution formula based on the mean and standard deviation.
4. **Plot**: Use `matplotlib` to plot the curve, with the line labeled according to the specified `label` and colored as per the `color` parameter.
5. **Display Mean and SD Lines**: Optionally add lines to indicate the mean and one standard deviation away from the mean on both sides.


## plot_normal_distribution
### Purpose
The `plot_normal_distribution` function in the `DBN_toolbox` is designed to visualize the probability density function of a dataset, highlighting its distribution characteristics, including the median and standard deviation. This visualization helps in assessing the data's distribution pattern, particularly how it conforms to a normal distribution.
### Parameters
- **data**: The dataset to plot, typically an array of numerical values.
- **name** (optional): A label for the dataset, used to title the plot and name the output file.
### Process
1. **Prepare the Data**: Convert the data into a NumPy array for statistical calculations.
2. **Calculate Statistics**: Compute the median and standard deviation of the dataset.
3. **Set up Plot**: Initialize a figure using `matplotlib.pyplot` and set dimensions for clarity.
4. **Compute and Plot Density**:
    - Use `scipy.stats.gaussian_kde` to estimate the density function of the data.
    - Generate a range of x values covering the data range for plotting the density.
    - Plot the density curve using these x values.
5. **Add Statistical Annotations**:
    - Mark the median on the plot with a vertical line.
    - Add additional lines for the median ± standard deviation to illustrate the spread of the data.
6. **Finalize Plot**:
    - Add a legend explaining the annotations.
    - Title the plot using the provided `name`.
    - Label axes as appropriate.
7. **Save the Plot**: The plot is saved as an image file, named based on the `name` parameter, facilitating easy reference and use in reports.


## plot_ground_truth
### Purpose
The `plot_ground_truth` function in the `DBN_toolbox` is designed to compare actual data labels (ground truth) to predicted labels from a model, visually representing the correctness of predictions over a dataset. This function is particularly useful for temporal or sequential data where the progression of predictions can be assessed against the actual values.
### Parameters
- **true_labels**: An array of the actual labels from the dataset.
- **predictions**: An array of predicted labels by the model.
- **model_name**: A string that identifies the model, used for titling the plot and naming the output file.
### Process
1. **Setup the Plot**: Initialize a figure using `matplotlib.pyplot` and set its dimensions to accommodate the sequence of labels.
2. **Plot True Labels**: Plot the `true_labels` as a line or series of points. This line serves as a reference for evaluating the predictions.
3. **Mark Predictions**:
    - Iterate over the sequence of true labels and predictions.
    - Where predictions correctly match the true labels, mark these points on the plot, typically with a distinct color or marker.
4. **Enhance Visualization**:
    - Optionally, distinguish correct predictions from incorrect ones using different colors or markers.
    - Include a legend that identifies what each symbol or color represents.
5. **Label and Save the Plot**:
    - Title the plot to reflect its content and purpose, incorporating the `model_name`.
    - Label axes to indicate the sample index (or time, if applicable) and the class labels.
    - Save the plot as an image file named using the `model_name` to facilitate easy referencing.


## read_data
### Purpose
The `read_data` function in the `DBN_toolbox` is specifically designed to load and preprocess data for use with Dynamic Bayesian Networks (DBNs). It reads specified columns from a CSV file, discretizes the data into bins, and labels these bins numerically, setting up the data for effective modeling in a DBN.
### Parameters
- **data_name**: String specifying the file path or name of the CSV file to be read.
- **columns**: List of strings representing the column names to extract and process from the CSV.
- **bins**: List of integers defining the number of bins to use for each column during discretization.
### Process
1. **Read CSV File**: Use `pandas.read_csv` to load specified columns from the CSV file. This includes any additional columns needed for indexing or grouping that are not specified explicitly in the `columns` list but are necessary for subsequent operations.
2. **Discretize Data**: For each column (except any specified indices or grouping columns like 'participant' or 'task'), use `pandas.qcut` to discretize the data into the specified number of bins. `qcut` attempts to divide the data into quantiles based on the number provided in `bins`, which means each bin will have roughly the same number of data points:
    - Generate labels for these bins starting from 0 upwards.
    - Store the thresholds or bin edges used, which can be important for interpreting the bins or applying the same binning to new data.
3. **Handle Special Columns**: If columns like 'participant' or 'task' are used for grouping or splitting the data but are not among the columns to be discretized, they are processed differently or simply retained without changes.
4. **Return Processed Data and Bin Info**: The function returns the discretized DataFrame and optionally any metadata such as bin edges or other transformations applied to the data.


## DBN_train
### Purpose
The `DBN_train` function in the `DBN_toolbox` is responsible for training the Dynamic Bayesian Network (DBN) on the data provided. This function handles data preprocessing, model fitting, and optional data shuffling, providing a robust framework for probabilistic learning.
### Parameters
- **network**: An initialized DBN instance, ready to be trained.
- **columns**: List of columns that should be included in the training data.
- **bins**: List of integers that specify the number of bins for discretizing each column in `columns`.
- **data_name**: Name of the CSV file containing the data.
- **shuffled** (optional): Boolean indicating whether the data should be shuffled. Useful for breaking up time-based patterns that might not be relevant.
- **seed** (optional): Seed for random number generation, ensuring reproducibility when shuffling data.
- **model_name** (optional): Name under which the trained model will be saved.
- **fast** (optional): Boolean indicating whether a fast version of training should be used. If `True`, certain steps like model saving might be skipped.
### Process
1. **Data Preparation**:
    - Use the `read_data` function to load and discretize the specified `columns` from `data_name` into the number of `bins` provided.
    - If `shuffled` is `True`, shuffle the data to randomize the order. This involves grouping by `participant` if present, shuffling these groups, and then recombining them into a single DataFrame.
2. **Split Data**:
    - Split the data into training and test datasets. Typically, this is an 80-20 split but can be adjusted based on specific needs.
    - Separate the data into two time slices (`t=0` and `t=1`) for the DBN, which is crucial for learning transitions between states in sequential data.
3. **Fit the Model**:
    - Call `network.fit` with the prepared and formatted training data to adjust the parameters of the DBN based on the input data.
4. **Save Model and Data** (optional):
    - If `fast` is `False`, serialize the trained DBN model using `pickle` and save it to a file specified by `model_name`.
    - Also, save the test and training data to CSV files for later use or evaluation.
5. **Output**:
    - Return the test dataset along with the trained network, allowing for immediate evaluation or further processing.


## DBN_evaluate
### Purpose
The `DBN_evaluate` function is pivotal in assessing the performance of a trained Dynamic Bayesian Network (DBN). It uses test data to generate predictions and then evaluates these predictions against the actual outcomes using various metrics.
### Parameters
- **network** (optional): The trained DBN model. If not provided, the function will load a model from a file.
- **test_data** (optional): Data on which to evaluate the model. If not provided, the function will load data from a specified file.
- **test_data_name** (optional): The name of the CSV file from which to load the test data if it is not passed directly.
- **model_name** (optional): The filename of the trained model to load if `network` is not directly provided.
- **variables_to_add**: List of variables used in the evaluation phase, typically matching or similar to those used in training.
### Process
1. **Load Model and Data**:
    - If the `network` is not provided, it will be loaded from a file specified by `model_name`.
    - Similarly, if `test_data` is not provided, it is loaded from a file specified by `test_data_name`.
2. **Prepare the Test Data**:
    - If required, preprocess the test data to align with the DBN's expectations, such as ensuring the correct columns are present and in the appropriate format.
3. **Generate Predictions**:
    - Use DBN inference methods to predict the outcome for each instance in the test dataset. This typically involves propagating evidence through the network to compute the probabilities of different outcomes.
4. **Calculate Performance Metrics**:
    - Compute various metrics such as accuracy, precision, recall, and F1-score to quantify the model's performance.
    - These metrics are particularly important to understand the effectiveness of the model across different classes or scenarios.
5. **Plot and Save the Confusion Matrix**:
    - Generate a confusion matrix to visually assess how well the model is predicting each class.
    - Save the confusion matrix as an image file for easy reference.
6. **Generate Evaluation Report**:
    - Optionally, generate a detailed evaluation report that includes all metrics and visualizations, and save it to an HTML file. This report contains Model Name, Test data name, Accuracy, Confusion Matrix, Precision, Recall, F1 Score, and Ground Truth vs Predicted Diagram.


## DBN_acc_and_sensitivity
### Purpose
The `DBN_acc_and_sensitivity` function in the `DBN_toolbox` is a specific utility designed to evaluate the accuracy and sensitivity of a Dynamic Bayesian Network (DBN). It is typically used to determine how well the model predicts true positives across classes, which is particularly important in classifications where one class might be more critical than others.
### Parameters
- **network** (optional): If not passed, the network will be loaded from a file using the `model_name`.
- **test_data_name** (optional): Specifies the CSV file from which test data should be loaded if `test_data` is not provided directly.
- **model_name**: The file path of the pre-trained DBN model, used to load the model if `network` is not passed.
- **variables_to_add**: List of variables to include in the evaluation phase, which should match those used during the model's training phase.
### Process
1. **Load Model and Test Data**:
    - If `network` is not provided, load it from the specified `model_name`.
    - If `test_data` is not provided, load it from `test_data_name`.
2. **Prepare Data for Prediction**:
    - Organize the test data to match the DBN structure, ensuring that all necessary variables are present and appropriately formatted.
3. **Perform Predictions**:
    - Iterate through the test data, using the DBN inference mechanism to predict the outcome based on the provided evidence (variables).
    - Collect both the predictions and the actual labels to compute metrics.
4. **Calculate Metrics**:
    - **Accuracy**: The proportion of total correct predictions over all test cases.
    - **Sensitivity (True Positive Rate)**: The ability of the model to correctly identify positive instances among the actual positives. This is crucial for applications where failing to detect positives can have serious consequences.
5. **Generate Outputs**:
    - Return the calculated accuracy and sensitivity as a tuple or structured output.
    - Optionally, this function could also provide a confusion matrix and other detailed reports, depending on the application's needs.


## DBN_fast_acc_and_sensitivity
## Purpose
The `DBN_fast_acc_and_sensitivity` function serves a similar purpose to `DBN_acc_and_sensitivity` but is optimized for faster performance, especially useful when quick evaluations are needed, such as during iterative testing or when working with very large datasets where the evaluation speed is critical.
### Key Differences
- **Performance Optimization**: `DBN_fast_acc_and_sensitivity` is designed to minimize I/O operations and reduce computational overhead by assuming that the model and test data are already loaded and formatted correctly, bypassing the steps of loading and processing data from files.
- **Use Case**: It is best used in situations where the environment is controlled, such as during the development phase where repeated quick tests are needed to tune parameters or during real-time system evaluations where delays caused by data loading can be prohibitive.
- **Function Parameters**: This function typically does not handle file I/O and expects the `network` and `test_data` to be passed directly, pre-loaded and pre-processed by the caller.
### Parameters
- **network**: The already loaded and trained DBN network.
- **test_data**: The pre-processed test data as a DataFrame, structured to align with the network’s requirements.
- **variables_to_add**: List of variables used in generating predictions, mirroring those used in training.
### Process
1. **Prepare Inference**:
    - Use `DBNInference` on the `network` to set up for probabilistic queries.
2. **Predictions and Label Collection**:
    - Loop through `test_data`, use DBN inference to predict the outcome for each row based on `variables_to_add`.
    - Collect both the predictions and actual labels to compute metrics.
3. **Calculate Metrics**:
    - **Accuracy**: Measure the overall correctness of predictions.
    - **Sensitivity**: Measure the model’s ability to identify true positives accurately, crucial for models where missing a positive is costly.
4. **Output Results**:
    - The function returns both accuracy and sensitivity, providing a quick snapshot of model performance.



## DBN_T1
### Purpose
The `DBN_T1` function in the `DBN_toolbox` is specifically designed to conduct a series of experiments to evaluate the performance of a Dynamic Bayesian Network (DBN) under varying conditions, particularly focusing on the impact of different random seeds on the training process. This is often used to assess the stability and robustness of the DBN model across different initializations and data shuffles.
### Purpose
`DBN_T1` aims to provide insights into how variations in the initialization and data partitioning affect the model's accuracy and sensitivity. This can be crucial for understanding model reliability in real-world applications.
### Parameters
- **network**: The initial DBN structure before training.
- **number_of_iterations**: The number of experiments to run, each with a different seed.
- **predictors**: List of predictor variables used in the model.
- **variables**: Complete list of variables including predictors and the target.
- **bins**: The binning specification for discretizing continuous variables during data preparation.
### Process
1. **Experiment Setup**: Initialize variables and set up structures to capture results across iterations.
2. **Loop Over Seeds**:
    - For each iteration, set a different seed to shuffle and split the data differently.
    - Reinitialize the network and train it with the shuffled and split data.
    - Evaluate the trained model using `DBN_fast_acc_and_sensitivity` to get accuracy and sensitivity metrics.
    - Store and log results for each seed.
3. **Statistical Analysis**:
    - Analyze the collected accuracy and sensitivity across different seeds to assess variability and average performance.
    - Optionally, plot these metrics to visualize performance trends and distribution.
4. **Summarize Results**:
    - Calculate mean, standard deviation, and other relevant statistical measures of the performance metrics.
    - Identify the seed that provided the best overall performance.



## DBN_T1_worker and DBN_T1_multithreaded
### Purpose
The `DBN_T1_worker` and `DBN_T1_multithreaded` functions in the `DBN_toolbox` extend the concept of the `DBN_T1` function by introducing multithreading to run multiple experiments concurrently. This approach significantly speeds up the process of evaluating a Dynamic Bayesian Network (DBN) under different initialization and training conditions, making it highly efficient for large-scale experiments.

### DBN_T1_worker

This function is a worker that runs a segment of the total experiments defined by `DBN_T1`, allowing it to be executed in a separate thread.
#### Parameters
- **start_seed, end_seed**: Define the range of seeds that this worker will handle.
- **network_template**: The initial DBN structure before training, which will be copied and reinitialized for each seed.
- **predictors**: Predictor variables for the DBN.
- **variables**: All variables involved in the model, including the target.
- **bins**: Discretization specifications for the variables.
- **results_queue**: A thread-safe queue where results are stored to be aggregated later.
#### Process
1. **Loop Through Seeds**: For each seed in the specified range, the function:
    - Reinitializes the network from the template.
    - Trains the network using the `DBN_train` function.
    - Evaluates the network using `DBN_fast_acc_and_sensitivity`.
    - Pushes the results (accuracy, sensitivity, and other metrics) into the `results_queue`.
### DBN_T1_multithreaded
This function coordinates multiple `DBN_T1_worker` threads to cover a range of experiments, effectively distributing the workload across available CPU cores.
#### Parameters
- **network**: The base network structure.
- **number_of_iterations**: Total number of experiments to run.
- **num_threads**: Number of worker threads to use.
- **predictors**, **variables**, **bins**: Passed to each worker for consistent model setup.
#### Process
1. **Setup Threads**: Divide the total number of iterations by the number of threads to distribute seeds evenly across workers.
2. **Start Threads**: Initialize each thread with a segment of the total seed range and start them.
3. **Collect Results**: Wait for all threads to complete and gather results from the `results_queue`.
4. **Post-Processing**: Perform statistical analysis or aggregation on the collected results to evaluate overall model performance.



## DBN_hyper
### Purpose
The goal of `DBN_hyper` is to optimize the binning strategy used in the discretization process of continuous variables, which is a crucial step in preparing data for a DBN. The right choice of bins can significantly influence the quality of the model's predictions.
### Parameters
- **network**: The initialized but untrained DBN model structure.
- **data_name**: The file path or name of the dataset.
- **columns**: List of all columns to be included in the model, typically involving both predictors and target variables.
- **predictors**: List of predictor variables used in the model.
- **bins**: Initial setting for the number of bins for discretization that might be adjusted during the process.
- **ran**: The maximum number of bins to test for each variable.
- **seed**: Seed for random number generation to ensure reproducibility.
### Process
1. **Initialize Parameter Space**:
    - Generate all possible combinations of bin numbers within the specified range (`ran`) for each variable in `columns`.
2. **Loop Over Bin Combinations**:
    - For each combination of bin counts:
        - Reinitialize the network with the current bin configuration.
        - Train the network on the dataset using these bins.
        - Evaluate the network's performance on a validation or test set.
        - Store performance metrics (accuracy, sensitivity) associated with the bin configuration.
3. **Determine Optimal Configuration**:
    - Compare the performance metrics across all tested configurations.
    - Select the configuration that yields the best performance based on predefined criteria (e.g., highest accuracy, highest sensitivity, or a balance of both).
4. **Output Results**:
    - Return detailed results of the hyperparameter tuning process, potentially including the best configuration and its associated performance metrics.



## compare_models
### Purpose

The function aims to provide a systematic approach to model comparison, highlighting the strengths and weaknesses of each model in terms of prediction capabilities and reliability.
### Key Metrics for Comparison
- **Accuracy**: Measures the overall correctness of the model across all predictions.
- **Sensitivity** (True Positive Rate): Measures the model's ability to correctly identify positive instances.
- **Precision**: Reflects the proportion of positive identifications that were actually correct.
- **F1-Score**: Balances precision and recall in a single metric, useful for models with imbalanced classes.
### Process
1. **Model Setup**:
    - Define the different models or configurations to be tested. This could involve different hyperparameters, training procedures, or entirely different algorithms (e.g., logistic regression, random forests, DBNs).
2. **Data Preparation**:
    - Prepare a consistent dataset for testing all models. This includes data cleaning, normalization, and splitting into training and test sets.
3. **Model Training and Prediction**:
    - Train each model on the same training dataset.
    - Generate predictions from each model on a common test set.
4. **Performance Evaluation**:
    - Calculate the chosen metrics (accuracy, sensitivity, precision, F1-score) for each model.
    - Optionally, generate confusion matrices or ROC curves for a more detailed performance analysis.
5. **Comparison and Visualization**:
    - Summarize the metrics in a comparative format, such as a table or bar graph.
    - Highlight significant differences and discuss potential reasons for variations in performance.
6. **Output Results**:
    - Return a comprehensive report detailing the performance of each model, including statistical measures and visualizations.
    - Provide recommendations based on the comparison results.




## DBN_train_and_evaluate_by_task
### Purpose
The primary goal of `DBN_train_and_evaluate_by_task` is to provide a granular view of how a DBN performs across different tasks, which might have unique characteristics or require different strategies for optimal performance. This can be crucial in domains like healthcare, finance, or manufacturing, where models might need to be robust across diverse operational conditions.
### Parameters
- **network**: The pre-initialized but untrained DBN model.
- **data_name**: The file path or name of the dataset.
- **columns**: List of columns to include from the dataset, typically involving both predictors and the target.
- **predictors**: List of predictor variables used in the model.
- **bins**: Specifications for the number of bins for discretizing continuous variables during data preparation.
- **shuffled** (optional): Whether to shuffle the dataset before splitting it into training and testing sets, which can help in reducing model bias towards the order of data.
- **model_name** (optional): The filename under which to save the trained model.
### Process
1. **Data Preparation**:
    - Load and preprocess the data, ensuring that it is split according to tasks. Each task will be treated as a separate segment for model training and evaluation.
    - Discretize the data as specified in the `bins` parameter.
2. **Model Training**:
    - For each task in the dataset, train a separate instance of the DBN. This involves:
        - Isolating the data related to a specific task.
        - Using this data to train the model, ensuring that the model is fit on data that reflects the specific characteristics of that task.
3. **Model Evaluation**:
    - Evaluate each task-specific model on its corresponding test set. This might involve calculating various performance metrics such as accuracy, precision, recall, and F1-score.
    - Store the evaluation results for each task, allowing for a comparative analysis across tasks.
4. **Performance Aggregation**:
    - Aggregate the performance metrics across all tasks to provide an overview of the model's effectiveness.
    - Optionally, generate detailed reports or visualizations that highlight the performance of the model for each task.
5. **Output Results**:
    - Return the aggregated performance metrics and any task-specific evaluation details.
    - Save the trained models if specified, allowing for later use or further analysis.




## DBN_predictor_impact
### Purpose
The goal of `DBN_predictor_impact` is to quantify how the removal of each predictor affects model performance metrics such as accuracy and sensitivity. This analysis helps identify key drivers of the model's predictions and can guide efforts to optimize and streamline the model.
### Parameters
- **data_name**: The file path or name of the dataset.
- **columns**: List of all columns to be included in the model, typically involving both predictors and target variables.
- **predictors**: List of predictor variables whose impact is to be evaluated.
- **bins**: Specifications for the number of bins for discretizing continuous variables during data preparation.
- **seed**: Seed for random number generation to ensure reproducibility, particularly in data shuffling and splitting.
### Process
1. **Baseline Model Training and Evaluation**:
    - Train the DBN with all predictors included and evaluate its performance to establish a baseline for comparison.
    - Calculate baseline metrics such as accuracy and sensitivity.
2. **Impact Assessment for Each Predictor**:
    - For each predictor in the `predictors` list:
        - Temporarily remove the predictor from the list of variables.
        - Re-train the model on the modified dataset and evaluate its performance without the removed predictor.
        - Compare the performance metrics (accuracy, sensitivity) to the baseline to determine the impact of removing the predictor.
3. **Result Compilation**:
    - Aggregate the impact results, detailing how the removal of each predictor affects the performance metrics.
    - Optionally, visualize these impacts using charts or graphs to provide a clear, comparative view.
4. **Output Results**:
    - Return a structured report or dictionary containing the performance impact of each predictor, alongside any visualizations or summary statistics.
