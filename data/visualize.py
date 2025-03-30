import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Create a histogram of the features with the data
def plot_feature_histograms(data):
    for column in data.columns:
        plt.hist(data[column], bins=30, alpha=0.5, label=column)
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.title(f'Histogram of {column}')
        plt.legend()
        plt.show()

# Create a scatter plot of the features with the data
def plot_feature_scatter(data):
    pass