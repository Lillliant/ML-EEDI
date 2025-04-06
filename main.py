import os, pprint, joblib, umap, pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from param import param_grid
from matplotlib import pyplot as plt

def load(data_path: str, output_dir: str = './output/', pca: bool = False, um: bool = False, fs: bool = False, n_components: int = 100):
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

    X = data.drop(['IsCorrect'], axis=1) # remove the AnswerId column and the target
    if 'AnswerId' in X.columns: X = X.drop(['AnswerId'], axis=1) 
    y = data['IsCorrect']
    print("Shape of data:", X.shape)

    # Data transformation and feature selection
    if pca:
        from sklearn.decomposition import PCA
        print("PCA is used. Transforming the data...")
        pca = PCA(n_components=n_components, whiten=True) # whiten=False by default
        X = pca.fit_transform(X)
    if um:
        reducer = umap.UMAP(n_components=n_components)
        print("UMAP is used. Transforming the data...")
        X = reducer.fit_transform(X)
    if fs:
        scaler = MinMaxScaler()
        print("Feature selection is used. Transforming the data...")
        X = scaler.fit_transform(X)
    
    # Get train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0, shuffle=True)
    print("Shape of training data:", X_train.shape)
    print("Shape of testing data:", X_test.shape)
    return X_train, X_test, y_train, y_test

def train(output_path, X_train, y_train, algorithm, model):
    if os.path.exists(f'{output_path}/gd_{algorithm}.pkl'):
        print("GridSearchCV already exists. Loading model...")
        grid_search = joblib.load(f'{output_path}/gd_{algorithm}.pkl')
        return grid_search
    
    print(f"GridSearchCV does not exist at {output_path}/gd_{algorithm}.pkl. Training model...")
    param = param_grid[algorithm] # Hyperparameters
    # Find the best hyperparameter with GridSearchCV
    # By default for classification, stratified CV is used
    scoring = ['accuracy'] # For now, use accuracy only ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
    grid_search = GridSearchCV(model, param, cv=StratifiedKFold(n_splits=3), scoring=scoring, refit='accuracy', n_jobs=-1, verbose=4)
    grid_search.fit(X_train, y_train)
    joblib.dump(grid_search, f'{output_path}/gd_{algorithm}.pkl')
    return grid_search

# Perform GridSearch CV on a particular model
def main(output_path:str, algorithm: str, pca: bool, um: bool, fs: bool, n_components: int):
    # Load the data
    X_train, X_test, y_train, y_test = load(data_path, output_path, pca=pca, um=um, fs=fs, n_components=n_components)

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

    gs = train(output_path, X_train, y_train, algorithm, model)
    # Evaluate the model
    print("Evaluating the model...")
    evaluate(output_path, algorithm, gs, X_train, y_train, X_test, y_test)

# Plot the feature importances
def plot_feature_importances(model, algorithm: str, data: pd.DataFrame, output_dir: str): 
    features = list(data.keys())
    plt.figure(layout='constrained')
    # for sake of visualization, only show the top 20 features
    feature_importance = pd.Series(model.feature_importances_, index=features).nlargest(20) 
    feature_importance.sort_values(ascending=False, inplace=True)
    feature_importance.plot(kind='barh')
    plt.title(f"Feature importance for {algorithm}")
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.savefig(f'{output_dir}/{algorithm}_feature_importance.png')
    plt.clf() # clear the figure

# For the sake of comparison, as IRT and NCDM has only parameter for AUC and Accuracy
# These scores will be used to evaluate the model
def evaluate(output_path, algorithm, gs, X_train, y_train, X_test, y_test):
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import roc_auc_score
    if not os.path.exists(f'{output_path}/eval'): os.makedirs(f'{output_path}/eval')

    print("Retrieving the best model...")
    model = gs.best_estimator_ # Load the best model according to the given metric in GridSearchCV
    # Show the result of the parameter tuning
    results = pd.DataFrame(gs.cv_results_)
    results.to_csv(f'{output_path}/eval/{algorithm}-tuning.csv', index=False)

    # Retrieve the best hyperparameters
    with open(f'{output_path}/eval/{algorithm}-best-param.txt', 'w') as f:
        f.write(f"Best hyperparameters for {algorithm}:\n")
        f.write(f"{pprint.pformat(gs.best_params_)}\n")
        f.write(f"Best score for {algorithm} on validation data:\n")
        f.write(f"{gs.best_score_}\n")
        f.close()
    
    print("Evaluating the model...")
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    with open(f'{output_path}/eval/{algorithm}-accuracy.txt', 'w') as f:
        f.write(f"Train accuracy: {train_accuracy}\n")
        f.write(f"Test accuracy: {test_accuracy}\n")
        f.write(f"AUC: {auc}\n")
    
    #plot_feature_importances(model, algorithm, X_train, f'{output_path}/eval') # plot the feature importances
    
if __name__ == '__main__':
    pca = False # PCA is used
    um = False # Feature hashing is used
    fs = False # Feature selection is used
    n_components = 100 # Number of components to keep
    dataset = 'eedi-one-hot' # Dataset to use
    output_path = f'./output/{dataset}{'-pca-' if pca else ''}{'-um-' if um else ''}{f'{n_components}' if pca or um else ''}{'-fs' if fs else ''}' # output_path = './output/eedi' './output/toy
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    data_path = f'./data/{dataset}/processed_eedi.csv' # data_path = './data/eedi/processed_eedi.csv' './data/eedi-toy/toy.csv'

    # Algorithms
    algorithm = ['BASE', 'GBDT', 'RFS'] # 'RFS', 'GBDT', 'LR', 'NN', 'BASE' <- Stratified baseline model
    for a in algorithm:
        print(f"Training {a} model...")
        main(output_path, a, pca, um, fs, n_components)