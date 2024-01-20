import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def plot_correlation_matrics(df_training, parent_plot_directory):
    print('Plot Correlation Matrix')
    # Calculate the correlation matrix
    print(list(df_training))
    correlation_matrix = df_training.drop(columns='Date Time').corr()

    # Plot the correlation matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(f'{parent_plot_directory}correlation_plot.png')
    plt.show()
    plt.close()


def plot_all_data(
        df=None,
        saving_path=None,
        title=None
):
    # Create a Plotly figure
    fig = go.Figure()
    # Add traces for each factor
    # 'Factor 3(H)'
    for factor in ['Target Value(BMC)', 'Factor 1(P)', 'Factor 2(T)', 'Factor 4(W)', 'Factor 5(R)']:
        fig.add_trace(go.Scatter(x=df['Date Time'], y=df[factor], mode='lines+markers', name=factor))
    # Update layout
    fig.update_layout(title=f' {title} Factors Over Time',
                      xaxis_title='Date',
                      yaxis_title='Value')
    # Show the plot
    # fig.show()
    fig.write_html(f'{saving_path}all_data_plot_{title}.html')
    fig.write_image(f'{saving_path}all_data_plot.png')


def plot_actual_vs_predicted(
        actual=None,
        predicted=None,
        parent_saving_path=None,
        filename=None,
        title=None
):
    plt.figure(figsize=(20, 10))
    plt.plot(predicted, marker='o', label='predicted')
    plt.plot(actual, marker='o', label='Actual')
    plt.title(f'{title}')
    plt.legend()
    plt.savefig(f'{parent_saving_path}{filename}.png')


def plot_actual_vs_predicted_using_plotly(
        actual=None,
        predicted=None,
        parent_saving_path=None,
        filename='results_on_testing_data',
        title='Results on Testing Data'

):
    # Create a Plotly scatter plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(predicted))),
                             y=predicted,
                             mode='markers+lines',
                             name='Predicted'))
    # Add actual values if needed
    fig.add_trace(go.Scatter(x=list(range(len(actual))),
                             y=actual,
                             mode='markers+lines',
                             name='Actual'))

    fig.update_layout(title=f'{title}', xaxis_title='Data Point', yaxis_title='Predicted Values')

    # Save or show the plot
    fig.write_html(f'{parent_saving_path}{filename}.html')
    # If you want to show the plot directly in the notebook, you can use fig.show()
