# Influencing Parameters

This project addresses a regression problem that has been tackled using multiple models, including both machine learning and deep learning approaches.

## Machine Learning Models

- **Non-linear Regression Model:** Utilizes non-linear relationships between input and output variables.
- **Random Forest Model:** Ensemble learning method that combines multiple decision trees.
- **Gradient Boosting Model:** Builds a series of weak learners to create a strong predictive model.
- **Support Vector Machine with Different Kernels:** Utilizes Support Vector Machines with various kernel functions for effective regression.

## Deep Learning Model

- **Simple Neural Network:** Basic neural network architecture for regression tasks.

## The resulting directory structure


The directory structure of your new project looks like this: 

```
├── LICENSE
├── compiled_results          <- Compiled results of all models saved in this directory.
├── configs                   <- it contain the configuration of the project(optional).
├── datasets
│   ├── dataset-testing.xlsx  <- Testing Data or unseen data.
│   ├── dataset-training.xlsx <- Training Data or seen data.
├── Materials                 <- This directory contains the requirements of the project.
│
├── Plots                     <- This directory contain the `visualization of the data` made by `matplotlib` and `plotly`.
│
├── results                   <- This direcoty contains the results of testing & training for each model in seperate files.
│       
├── trained_models            <- This directory contain `.pkl` files of trained models.
├── main.py                   <- `main.py` is the main file that contain code of all project. You can handle it using `config.ini` files by runnning each operationg
├── config.ini                <- it contain the main logic for runnning `main.py` file . this file it self explainatory.
└── requirements.txt          <- it contain the reqiure packages that have to install in envrinment in order to run this project
```


## Setup

To run this project, follow these steps:

1. Clone the repository to your local machine.
2. Create a virtual environment:

    ```bash
    python -m venv venv
    ```
3. Activate the virtual environment:

    - On Windows:

        ```bash
        .\venv\Scripts\activate
        ```

    - On macOS/Linux:

        ```bash
        source venv/bin/activate
        ```

4. Install the required packages from the `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

Now, you have set up the virtual environment with the necessary dependencies to run the project.