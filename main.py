import os
import joblib
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from src.evaluation import calculate_errors
from src.visualization import (plot_correlation_matrics,
                               plot_all_data,
                               plot_actual_vs_predicted,
                               plot_actual_vs_predicted_using_plotly)
import configparser
import warnings
from sklearn.ensemble import RandomForestRegressor
import ast
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

parent_plot_directory = os.path.join('Plots/matplotlib/')
parent_plot_plotly_directory = os.path.join('Plots/plotly/')

# Reading Config File for the configration
config = configparser.ConfigParser()
config_file_path = 'config.ini'
config.read(config_file_path)

correlation_matrix_plot = config.get('visualization_configration', 'correlation_matrix')
data_plot = config.get('visualization_configration', 'data_plot')

training_data_evaluation_plot = config.get('visualization_configration', 'training_data_evaluation_plot')
testing_data_evaluation_plot = config.get('visualization_configration', 'testing_data_evaluation_plot')

training_data_evaluation_plot_plotly = config.get('visualization_configration', 'training_data_evaluation_plot_plotly')
testing_data_evaluation_plot_plotly = config.get('visualization_configration', 'testing_data_evaluation_plot_plotly')

training_file_path = config.get('files_path', 'training_file_path')
testing_file_path = config.get('files_path', 'testing_file_path')

polynomial_model = config.get('machine_learing_models', 'polynomial_model')
random_forest_model = config.get('machine_learing_models', 'random_forest_model')
neural_network_model = config.get('machine_learing_models', 'neural_network_model')

# Support Vector Machine
support_vector_machine_rbf = config.get('machine_learing_models', 'support_vector_machine_rbf')
support_vector_machine_linear = config.get('machine_learing_models', 'support_vector_machine_linear')
support_vector_machine_poly = config.get('machine_learing_models', 'support_vector_machine_poly')
support_vector_machine_sigmoid = config.get('machine_learing_models', 'support_vector_machine_sigmoid')
gradient_boosting_regressor = config.get('machine_learing_models', 'gradient_boosting_regressor')

model_name = config.get('machine_learing_models', 'model_name')
compiled_all_results = config.get('machine_learing_models', 'compiled_all_results')

data_independent_features = ast.literal_eval(config.get('data_features', 'data_independent_features'))
data_dependent_features = ast.literal_eval(config.get('data_features', 'data_dependent_features'))

print('dependent Data Features : ', data_dependent_features)
print('Independent Data Features : ', data_independent_features)

print(f'Parent directory for Plots : {parent_plot_directory}')
print(f'FilePath for training Data : {training_file_path}')
print(f'FilePath for Testing Data : {testing_file_path}')

# Reading Data
df_training = pd.read_excel(training_file_path)
df_testing = pd.read_excel(testing_file_path)

rows_to_add = int(0.3 * len(df_testing))
sampled_rows = df_testing.sample(n=rows_to_add, random_state=42)
df_training = pd.concat([df_training, sampled_rows], ignore_index=True)
df_testing = df_testing.drop(sampled_rows.index)

df_training.reset_index(drop=True, inplace=True)
df_testing.reset_index(drop=True,inplace=True)
# Training Data
X_train = df_training[data_independent_features]
y_train = df_training[data_dependent_features]

# Change type
X_train = X_train.astype(float)
y_train = y_train.astype(float)

# Unseen data | Validation Data | Testing Data
# 'Factor 3(H)'
X_test = df_testing[data_independent_features]
y_test = df_testing[data_dependent_features]

# change type
X_test = X_test.astype(float)
y_test = y_test.astype(float)

print(f"shape of Training Data(X) : {X_train.shape}")
print(f"shape of Training Data(Y) : {y_train.shape}")
print(f"shape of Testing Data(X) : {X_test.shape}")
print(f"shape of Testing Data(Y) : {y_test.shape}")

if correlation_matrix_plot == 'True':
    plot_correlation_matrics(df=df_training,
                             saving_path=parent_plot_directory,
                             title='training_data')
    plot_correlation_matrics(df=df_testing,
                             saving_path=parent_plot_directory,
                             title='testing_data')

if data_plot == 'True':
    plot_all_data(df=df_training,
                  saving_path=parent_plot_plotly_directory,
                  title='Training')
    plot_all_data(df=df_testing,
                  saving_path=parent_plot_plotly_directory,
                  title='Testing'
                  )
    df = pd.concat([df_training, df_testing], ignore_index=True)
    plot_all_data(df=df,
                  saving_path=parent_plot_plotly_directory,
                  title='CombinedData'
                  )

from sklearn.pipeline import make_pipeline

if polynomial_model == 'True':
    print('[INFO] Training Polynomial Regression Model')

    # Modeling
    # Train polynomial regression model on the whole dataset
    pr = PolynomialFeatures(degree=4)
    X_poly_train = pr.fit_transform(X_train)
    X_poly_test = pr.fit_transform(X_test)

    model = LinearRegression()
    model.fit(X_poly_train, y_train)


import pickle
if random_forest_model == 'True':
    print('[INFO] Training Random Forest Model')
    # Modeling using Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Fit the model on the training data
    model.fit(X_train, y_train)
    # joblib.dump(model, f"trained_models/{model_name}.joblib")

    with open(f"trained_models/{model_name}.pickle",'wb' ) as file:
            pickle.dump(model, file)
    X_poly_train = X_train
    X_poly_test = X_test

if neural_network_model == 'True':
    # Initialize the scaler
    scaler = MinMaxScaler()

    # Fit the scaler on the training data and transform both training and test data
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    with open(f"trained_models/{model_name}_scaler.pickle",'wb' ) as file:
        pickle.dump(scaler, file)

    print('[INFO] Training Neural Network Model')

    model = Sequential()
    model.add(Dense(64,
                    input_dim=X_train.shape[1],
                    activation='relu')
              )
    model.add(Dense(32,
                    activation='relu')
              )
    model.add(Dense(16,
                    activation='relu')
              )
    model.add(Dense(8,
                    activation='relu')
              )
    
    model.add(Dense(1,
                    activation='linear'))
    model.compile(
            optimizer='adam',
            loss='mean_squared_error'
    )

    model.fit(X_train, y_train,
              epochs=100,
              batch_size=16,
              validation_split=0.2)
    
    with open(f"trained_models/{model_name}.pickle",'wb' ) as file:
        pickle.dump(model, file)

    X_poly_train = X_train
    X_poly_test = X_test

from sklearn import svm
import joblib

if support_vector_machine_rbf == 'True':
    print('[INFO] Training Support Vector Machine kernel==RBF')
    # Initialize the scaler
    scaler = MinMaxScaler()

    # Fit the scaler on the training data and transform both training and test data
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    with open(f"trained_models/{model_name}_scaler.pickle",'wb' ) as file:
        pickle.dump(scaler, file)

    model = svm.SVR(kernel='rbf')
    model.fit(X_train, y_train)
    
    with open(f"trained_models/{model_name}.pickle",'wb' ) as file:
        pickle.dump(model, file)

    X_poly_train = X_train
    X_poly_test = X_test

if support_vector_machine_linear == 'True':
    print('[INFO] Training Support Vector Machine kernel = linear')
    # Initialize the scaler
    scaler = MinMaxScaler()

    # Fit the scaler on the training data and transform both training and test data
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    with open(f"trained_models/{model_name}_scaler.pickle",'wb' ) as file:
        pickle.dump(scaler, file)

    model = svm.SVR(kernel='linear')
    model.fit(X_train, y_train)
    with open(f"trained_models/{model_name}.pickle",'wb' ) as file:
        pickle.dump(model, file)

    X_poly_train = X_train
    X_poly_test = X_test

if support_vector_machine_poly == 'True':
    print('[INFO] Training Support Vector Machine kernel = poly')
    # Initialize the scaler
    scaler = MinMaxScaler()

    # Fit the scaler on the training data and transform both training and test data
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    with open(f"trained_models/{model_name}_scaler.pickle",'wb' ) as file:
        pickle.dump(scaler, file)

    
    model = svm.SVR(kernel='poly')
    model.fit(X_train, y_train)
    with open(f"trained_models/{model_name}.pickle",'wb' ) as file:
        pickle.dump(model, file)

    X_poly_train = X_train
    X_poly_test = X_test

if support_vector_machine_sigmoid == 'True':
    print('[INFO] Training Support Vector Machine kernel = sigmoid')
    # Initialize the scaler
    scaler = MinMaxScaler()

    # Fit the scaler on the training data and transform both training and test data
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    with open(f"trained_models/{model_name}_scaler.pickle",'wb' ) as file:
        pickle.dump(scaler, file)

    
    with open(f"trained_models/{model_name}_scaler.pickle",'wb' ) as file:
        pickle.dump(scaler, file)

    model = svm.SVR(kernel='sigmoid')
    model.fit(X_train, y_train)
    with open(f"trained_models/{model_name}.pickle",'wb' ) as file:
        pickle.dump(model, file)

    X_poly_train = X_train
    X_poly_test = X_test

from sklearn.ensemble import GradientBoostingRegressor

if gradient_boosting_regressor == 'True':
    print('[INFO] Training gradient_boosting_regressor Model ...')
    model = GradientBoostingRegressor(random_state=0)
    model.fit(X_train, y_train)
    
    with open(f"trained_models/{model_name}.pickle",'wb' ) as file:
        pickle.dump(model, file)

    X_poly_train = X_train
    X_poly_test = X_test

if training_data_evaluation_plot == 'True':
    y_pred_train = model.predict(X_poly_train)
    plot_actual_vs_predicted(actual=y_train,
                             predicted=y_pred_train,
                             parent_saving_path=parent_plot_directory,
                             filename='results_on_training_data',
                             title='Results on Training Data'
                             )

    error_metrics = calculate_errors(y_train, y_pred_train)

    print(f'Error Metrics on Training Data : {error_metrics}')

if testing_data_evaluation_plot == 'True':
    # Predict results on training Data
    y_pred_test = model.predict(X_poly_test)  # Polynomial Regression

    plot_actual_vs_predicted(actual=y_test,
                             predicted=y_pred_test,
                             parent_saving_path=parent_plot_directory,
                             filename='results_on_testing_data',
                             title='Results on Testing Data'
                             )

    error_metrics = calculate_errors(y_test, y_pred_test)
    print(f'Error Metrics on Testing Data : {error_metrics}')

if training_data_evaluation_plot_plotly == 'True':
    # Predict results on testing Data
    y_pred_train = model.predict(X_poly_train)
    if neural_network_model == 'True':
        y_pred_train = y_pred_train.flatten()

    plot_actual_vs_predicted_using_plotly(actual=y_train,
                                          predicted=y_pred_train,
                                          parent_saving_path=parent_plot_plotly_directory,
                                          filename='results_on_training_data',
                                          title='Results on Training Data'
                                          )
    error_metrics = calculate_errors(y_train, y_pred_train)
    print(f'Error Metrics on Training Data : {error_metrics}')

    # Saving Results
    error_metrics_df = pd.DataFrame(list(error_metrics.items()), columns=['Metric', 'Value'])
    os.makedirs(f'results/{model_name}', exist_ok=True)
    error_metrics_df.to_csv(f'results/{model_name}/training_results.csv', index=False)

if testing_data_evaluation_plot_plotly == 'True':
    # Predict results on testing Data
    y_pred_test = model.predict(X_poly_test)  # Polynomial Regression

    if neural_network_model == 'True':
        y_pred_test = y_pred_test.flatten()

    plot_actual_vs_predicted_using_plotly(actual=y_test,
                                          predicted=y_pred_test,
                                          parent_saving_path=parent_plot_plotly_directory,
                                          filename='results_on_testing_data',
                                          title='Results on Testing Data')
    error_metrics = calculate_errors(y_test, y_pred_test)
    print(f'Error Metrics on Testing Data : {error_metrics}')

    # Saving Results
    error_metrics_df = pd.DataFrame(list(error_metrics.items()), columns=['Metric', 'Value'])
    os.makedirs(f'results/{model_name}', exist_ok=True)
    error_metrics_df.to_csv(f'results/{model_name}/testing_results.csv', index=False)

import pickle
random_forest_model_trained_model = config.get('load_trained_models', 'random_forest_model_trained_model')
if random_forest_model_trained_model == 'True':
    print('[INFO] Loading Trained Random forest model')
    model_results_path ='trained_models/random_forest_model.pickle'
    with open(model_results_path, 'rb') as file:
        trained_model = pickle.load(file)
    
    input_features = ["t", "Factor 1(P)", "Factor 2(T)", "Factor 4(W)", "Factor 5(R)"]
    df = pd.read_excel("datasets/DataSet-testing.xlsx")
    input_values = df[input_features]
    print(trained_model.predict(input_values))



import glob

if compiled_all_results == 'True':

    compiled_results_dict = {
            'Model': [],
            'mae'  : [],
            'mse'  : [],
            'rmse' : [],
            'mape' : [],
            'r2'   : [],
    }

    list_of_models = [model.split('/')[-2] + '_' + model.split('/')[-1].split('.')[0].split('_')[0] + '_data'
                      for model in glob.glob('results/*/*.csv')]
    list_of_results_files = glob.glob('results/*/*.csv')

    for results_file_path in list_of_results_files:

        model_name = results_file_path.split('/')[1] + '_' + results_file_path.split('/')[-1].split('.')[0].split('_')[
            0] + '_data'
        compiled_results_dict['Model'].append(model_name)

        results_file_path_df = pd.read_csv(results_file_path)
        for i in range(5):
            metrics_name, metric_value = results_file_path_df.to_dict(orient='records')[i]['Metric'], \
            results_file_path_df.to_dict(orient='records')[i]['Value']
            compiled_results_dict[metrics_name].append(metric_value)
    compiled_results_df = pd.DataFrame.from_dict(compiled_results_dict)
    compiled_results_df.to_csv('compiled_results/compiled_all_model_results.csv', index=False)
    print(compiled_results_df.sort_values(by='mae'))
