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
        plt.clf()

# Create a scatter plot of the features with the data
def plot_feature_scatter(data):
    pass

if __name__ == "__main__":
    data_path = './eedi/data.pkl'
    if os.path.exists(data_path):
        data = pd.read_pickle(data_path)
        plot_feature_histograms(data)