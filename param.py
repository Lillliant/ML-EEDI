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
    'BASE' : {
        'strategy': ['stratified'],
    }
}