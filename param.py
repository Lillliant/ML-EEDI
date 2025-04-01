param_grid = {
    'RFS' : {
        'n_estimators': [300, 500],
        'max_features': ['sqrt', 28, 100],
        'max_depth': [None, 5, 10, 20],
    },
    'GBDT' : {
        'n_estimators': [100, 300, 500, 1000],
        'learning_rate': [0.001, 0.01, 0.1, 1],
        'objective': ['binary:logistic'],
        'max_depth': [1, 3, 5, 7, 10],
    },
    'LR' : {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['sag', 'saga', 'liblinear'],
        'max_iter': [10000],
    },
    'NN' : {
        'hidden_layer_sizes': [(100,), (300,), (500,)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam'],
        'alpha': [0.01, 0.1, 1, 10],
        'max_iter': [10000],
    },
    'BASE' : {
        'strategy': ['stratified'],
    }
}