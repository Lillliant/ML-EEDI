import os
import sys
import pandas as pd
import numpy as np
import pprint
import joblib
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from param import param_grid
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

def load(data_path: str, output_dir: str = './output/', pca: bool = False):
    # Most of the data is already preprocessed by preprocess.py in eedi-raw
    # Hence this is mostly to further reduce the data size
    if os.path.exists(f'{output_dir}/data.pkl'):
        print("Data already exists. Loading data...")
        data = pd.read_pickle(f'{output_dir}/data.pkl')
    else:
        print("Data does not exist. Fetching data...")
        data = pd.read_csv(data_path)
        print("Shape of original data:", data.shape)
        data = data.groupby('IsCorrect', group_keys=False)[list(data.keys())].apply(lambda x: x.sample(frac=0.2, random_state=0))
        print("Shape of data after stratified sampling:", data.shape)
        pd.to_pickle(data, f'{output_dir}/data.pkl')

    X = data.drop(['IsCorrect'], axis=1)
    y = data['IsCorrect']
    # If PCA is used, then transform the data
    if pca:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=10) # whiten=False by default
        X = pca.fit_transform(X)
    return X, y

# Perform GridSearch CV on a particular model
def main(algorithm: str):
    param = param_grid[algorithm] # Hyperparameters
    
    # Load the data
    X, y = load(data_path, output_path, pca=False)
    print("Shape of data:", X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

    # Initialize the model
    match algorithm:
        case 'RFS':
            model = RandomForestClassifier(random_state=0)
        case 'GBDT':
            model = XGBClassifier(random_state=0)
        case 'LR':
            model = LogisticRegression(random_state=0)
        case 'NN':
            model = MLPClassifier(random_state=0)
        case _:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    # Find the best hyperparameter with GridSearchCV
    # By default for classification, stratified CV is used
    scoring = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc'] # And then plot ROC and AUC curve
    grid_search = GridSearchCV(model, param, cv=StratifiedKFold(n_splits=3), scoring=scoring, refit='accuracy', n_jobs=-1, verbose=4)
    grid_search.fit(X_train, y_train)
    print("Best hyperparameters:")
    pprint.pprint(grid_search.best_params_)
    print("Best score:")
    pprint.pprint(grid_search.best_score_)
    joblib.dump(grid_search, f'{output_path}/gd_{algorithm}.pkl')


if __name__ == '__main__':
    output_path = './output/eedi' # output_path = './output/eedi' './output/toy
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    data_path = './data/eedi/processed_eedi.csv' # data_path = './data/eedi/processed_eedi.csv' './data/eedi-toy/toy.csv'

    # Algorithms
    algorithm = 'LR' # 'RFT', 'GBDT', 'LR', 'NN'
    main(algorithm)
