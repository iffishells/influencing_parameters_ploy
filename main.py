import os
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
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
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
    plot_correlation_matrics(df= df_testing,
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



# if polynomial_model == 'True':
#     print('[INFO] Training Polynomial Regression Model')
#
#     # Modeling
#     # Train polynomial regression model on the whole dataset
#     pr = PolynomialFeatures(degree=4)
#     X_poly_train = pr.fit_transform(X_train)
#     X_poly_test = pr.fit_transform(X_test)
#
#     model = LinearRegression()
#
#     print('[INFO] Model Parameters')
#     print('Parameter : ', model.get_params())
#     print(f'[INFO] Model Coefficients : {model.coef_}')
#     print(f'[INFO] Model Intercepts : {model.intercept_}')
#     model.fit(X_poly_train, y_train)
#
#     # Access coefficients from the LinearRegression step within the pipeline
#     coefficients = model.named_steps['linearregression'].coef_
#     intercept = model.named_steps['linearregression'].intercept_
#
#     print(f'[INFO] Model Coefficients : {coefficients}')
#     print(f'[INFO] Model Intercepts : {intercept}')
#
#
from sklearn.pipeline import make_pipeline
if polynomial_model == 'True':
    print('[INFO] Training Polynomial Regression Model')

    # Modeling
    # Train polynomial regression model on the whole dataset
    pr = PolynomialFeatures(degree=4)
    X_poly_train = pr.fit_transform(X_train)
    X_poly_test = pr.fit_transform(X_test)

    model = LinearRegression()

    print('[INFO] Model Parameters')
    print('Parameter : ', model.get_params())

    # Create a pipeline with PolynomialFeatures and LinearRegression
    pipeline = make_pipeline(pr, model)

    # Fit the model using the pipeline
    pipeline.fit(X_poly_train, y_train)

    # Access coefficients from the LinearRegression step within the pipeline
    coefficients = pipeline.named_steps['linearregression'].coef_
    intercept = pipeline.named_steps['linearregression'].intercept_

    print(f'[INFO] Model Coefficients : {coefficients}')
    print(f'[INFO] Model Intercepts : {intercept}')



if random_forest_model == 'True':
    print('[INFO] Training Random Forest Model')
    # Modeling using Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Fit the model on the training data
    model.fit(X_train, y_train)
    X_poly_train = X_train
    X_poly_test = X_test

if neural_network_model == 'True':

    # Initialize the scaler
    scaler = MinMaxScaler()

    # Fit the scaler on the training data and transform both training and test data
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print('[INFO] Training Neural Network Model')

    model = Sequential()
    model.add(Dense(8,
                    input_dim=X_train.shape[1],
                    activation='relu')
              )
    # model.add(Dense(64,
    #                 activation='relu')
    #           )
    # model.add(Dense(32,
    #                 activation='relu')
    #           )
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
    X_poly_train = X_train
    X_poly_test = X_test
if training_data_evaluation_plot == 'True':
    y_pred_train = model.predict(X_poly_train)  # Polynomial Regression
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
import matplotlib.pyplot as plt

if training_data_evaluation_plot_plotly == 'True':
    # Predict results on testing Data
    y_pred_train = model.predict(X_poly_train)  # Polynomial Regression
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
