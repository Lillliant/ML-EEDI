import os, pprint, joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
import umap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from param import param_grid

def load(data_path: str, output_dir: str = './output/', pca: bool = False, um: bool = False, n_components: int = 100):
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
        print("PCA is used. Transforming the data...")
        pca = PCA(n_components=n_components, whiten=True) # whiten=False by default
        X = pca.fit_transform(X)
    if um:
        reducer = umap.UMAP(n_components=n_components)
        print("UMAP is used. Transforming the data...")
        X = reducer.fit_transform(X)
    return X, y

# Perform GridSearch CV on a particular model
def main(algorithm: str, pca: bool, um: bool, fs: bool, n_components: int):
    param = param_grid[algorithm] # Hyperparameters
    
    # Load the data
    X, y = load(data_path, output_path, pca=pca, um=um, n_components=n_components)
    print("Shape of data:", X.shape)

    if fs:
        scaler = MinMaxScaler()
        print("Feature selection is used. Transforming the data...")
        X = scaler.fit_transform(X)
        # Feature selection

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
        case 'BASE':
            model = DummyClassifier(random_state=0)
        case _:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
    # Additional code when feature selection is used
    if fs and algorithm != 'BASE':
        # Feature selection using Random Forest
        print("Shape of training data before feature selection:", X_train.shape)
        selector = SelectFromModel(model, threshold='median')
        selector.fit(X_train, y_train)
        X_train = selector.transform(X_train)
        X_test = selector.transform(X_test)
        print("Shape of training data after feature selection:", X_train.shape)

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
    pca = False # PCA is used
    um = True # Feature hashing is used
    fs = True # Feature selection is used
    n_components = 150 # Number of components to keep

    output_path = f'./output/eedi{'-pca-' if pca else ''}{'-um-' if um else ''}{f'{n_components}' if pca or um else ''}{'-fs' if fs else ''}' # output_path = './output/eedi' './output/toy
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    data_path = './data/eedi/processed_eedi.csv' # data_path = './data/eedi/processed_eedi.csv' './data/eedi-toy/toy.csv'

    # Algorithms
    algorithm = ['RFS'] # 'RFS', 'GBDT', 'LR', 'NN', 'BASE' <- Stratified baseline model
    for a in algorithm:
        main(a, pca, um, fs, n_components)