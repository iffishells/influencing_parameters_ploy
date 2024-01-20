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

parent_plot_directory = os.path.join('Plots/')
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

print(f'Parent directory for Plots : {parent_plot_directory}')
print(f'FilePath for training Data : {training_file_path}')
print(f'FilePath for Testing Data : {testing_file_path}')

# Reading Data
df_training = pd.read_excel(training_file_path)
df_testing = pd.read_excel(testing_file_path)

# Training Data
# 't', 'Factor 1(P)', 'Factor 2(T)', 'Factor 3(H)', 'Factor 4(W)', 'Factor 5(R)'
# 'Factor 3(H)'
X_train = df_training[['Factor 1(P)', 'Factor 2(T)', 'Factor 4(W)', 'Factor 5(R)']]
y_train = df_training['Target Value(BMC)']

# Change type
X_train = X_train.astype(float)
y_train = y_train.astype(float)

# Unseen data | Validation Data | Testing Data
# 'Factor 3(H)'
X_test = df_testing[['Factor 1(P)', 'Factor 2(T)','Factor 4(W)', 'Factor 5(R)']]
y_test = df_testing['Target Value(BMC)']

# change type
X_test = X_test.astype(float)
y_test = y_test.astype(float)

print(f"shape of Training Data(X) : {X_train.shape}")
print(f"shape of Training Data(Y) : {y_train.shape}")
print(f"shape of Testing Data(X) : {X_test.shape}")
print(f"shape of Testing Data(Y) : {y_test.shape}")

if correlation_matrix_plot == 'True':
    plot_correlation_matrics(df_training, parent_plot_directory)
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


# Modeling
# Train polynomial regression model on the whole dataset
pr = PolynomialFeatures(degree=4)
X_poly_train = pr.fit_transform(X_train)
X_poly_test = pr.fit_transform(X_test)

lr_2 = LinearRegression()

lr_2.fit(X_poly_train, y_train)

if training_data_evaluation_plot == 'True':
    y_pred_train = lr_2.predict(X_poly_train)  # Polynomial Regression
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
    y_pred_test = lr_2.predict(X_poly_test)  # Polynomial Regression

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
    y_pred_train = lr_2.predict(X_poly_train)  # Polynomial Regression

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
    y_pred_test = lr_2.predict(X_poly_test)  # Polynomial Regression


    plot_actual_vs_predicted_using_plotly(actual=y_test,
                                          predicted=y_pred_test,
                                          parent_saving_path=parent_plot_plotly_directory,
                                          filename='results_on_testing_data',
                                          title='Results on Testing Data')
    y_test.to_csv('test_sample.csv',index=False)
    error_metrics = calculate_errors(y_test, y_pred_test)
    print(f'Error Metrics on Testing Data : {error_metrics}')



