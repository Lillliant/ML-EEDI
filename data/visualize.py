import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Create a histogram of the features with the data
def plot_feature_histograms(data):
    data = data.drop_duplicates(['UserId', 'QuestionId'])
    features = ['UserId', 'QuestionId', 'Confidence']
    print(features)
    correct = data[data['IsCorrect'] == 1]
    incorrect = data[data['IsCorrect'] == 0]
    for key in features:
        _, bins = np.histogram(data[key], bins=25)
        plt.hist(correct[key], bins=bins, alpha=0.5, label="correct")
        plt.hist(incorrect[key], bins=bins, alpha=0.5, label="incorrect")
        plt.xlabel("Feature magnitude")
        plt.ylabel("Frequency")
        plt.title(f"Distribution for {key}")
        plt.legend(loc="best")
        plt.savefig(f'./{key}.png')
        plt.clf() # clear the figure

# Create a scatter plot of the features with the data
def plot_feature_scatter(data):
    pass

def plot_value_count_bar(data_1: pd.DataFrame, data_2: pd.DataFrame, output_dir: str):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    features = ['UserId', 'QuestionId']
    for key in features:
        value_count_1 = data_1[key].value_counts()
        value_count_1.plot(kind='hist', bins=30, alpha=0.5, label="Original Data")
        value_count_2 = data_2[key].value_counts()
        value_count_2.plot(kind='hist', bins=30, alpha=0.5, label="Filtered Data")
        plt.ylabel("Total count")
        plt.xlabel(f"Value count of {key}")
        plt.legend(loc="best")
        plt.title(f"Value count for {key}")
        plt.savefig(f'{output_dir}/{key}_value_count.png')
        plt.clf() # clear the figure

if __name__ == "__main__":
    dataset_1 = 'eedi-full'
    dataset_2 = 'eedi'
    data_path_1 = f'./{dataset_1}/data.pkl'
    data_path_2 = f'./{dataset_2}/data.pkl'
    output_dir = f'./output/'
    if os.path.exists(data_path_1) and os.path.exists(data_path_2):
        data_1 = pd.read_pickle(data_path_1)
        data_2 = pd.read_pickle(data_path_2)
        #plot_feature_histograms(data)
        plot_value_count_bar(data_1, data_2, output_dir)
    #print(data_1['IsCorrect'].value_counts())