# This is adapted from the NCDM example in EduCDM
# Modified with the dataset processed from EEDI
from EduCDM import GDIRT
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

# Create the train, validation, and test sets (EduCDM requires the target variable to be with X)
data = pd.read_csv('../data/eedi-no-one-hot/processed_eedi.csv')
data = data.groupby('IsCorrect', group_keys=False)[list(data.keys())].apply(lambda x: x.sample(frac=0.2, random_state=0))
train_data, test_data = train_test_split(data, stratify=data['IsCorrect'], random_state=0)
df_item = pd.read_csv('../data/eedi-no-one-hot/metadata/question_metadata.csv')
# VERY IMPORTANT - keeps the code from bugging out as embeddings depend on it
train_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)

# Transformation parameters
batch_size = 32
user_n = np.max([np.max(train_data['UserId']), np.max(test_data['UserId'])])*batch_size
item_n = np.max([np.max(train_data['QuestionId']), np.max(test_data['QuestionId'])])*batch_size

# EduCDM's transform function for the IRT model data
def transform(x, y, z, batch_size, **params):
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.int64),
        torch.tensor(y, dtype=torch.int64),
        torch.tensor(z, dtype=torch.float32)
    )
    return DataLoader(dataset, batch_size=batch_size, **params)

def merge(fold_1: pd.DataFrame, fold_2: pd.DataFrame):
    # Merge the two folds into one
    merged = pd.concat([fold_1, fold_2], axis=0, ignore_index=True)
    return merged

# Implement a manual 3-fold cross-validation onto the algorithm
def train(data, output_dir: str = './output', epoch: int = 10, n_lr: list[float] = [0.0002, 0.002, 0.02]) -> tuple:
    # Stratified sampling to create 3 folds
    f_12, fold_3 = train_test_split(data, stratify=data['IsCorrect'], test_size=0.33, random_state=0)
    fold_1, fold_2 = train_test_split(f_12, stratify=f_12['IsCorrect'], test_size=0.5, random_state=0)
    del f_12 # save memory space
    # transform the data into the format required by EduCDM
    folds = [fold_1, fold_2, fold_3]
    for f in folds: f.reset_index(drop=True, inplace=True) # reset the index of the folds
    best_accuracy = 0 # placeholder for the best cross-validation accuracy
    best_param = None # placeholder for the best hyperparameters (epoch, lr)
    for lr in n_lr:
        fold_accuracy = []
        for i, fold in enumerate(folds): # perform 3-fold cross-validation
            # set the respective folds for training and validation and
            # transform the data into the format required by EduCDM
            valid_set = transform(fold["UserId"], fold["QuestionId"], fold["IsCorrect"], batch_size)
            train_set = merge(folds[(i+1)%3], folds[(i+2)%3]) # Merge the other two folds
            train_set = transform(train_set["UserId"], train_set["QuestionId"], train_set["IsCorrect"], batch_size)
            cdm = GDIRT(user_n, item_n) # init the model
            cdm.train(train_set, epoch=epoch, lr=lr)
            _, accuracy = cdm.eval(valid_set)
            fold_accuracy.append(accuracy)
        accuracy = np.mean(fold_accuracy) # average the accuracy of the 3 folds
        if accuracy > best_accuracy: 
            best_accuracy = accuracy
            best_param = (epoch, lr)
        print(f"Parameter {(epoch, lr)}, Validation accuracy: {accuracy}")
    # retrain the model with the best parameters on the whole dataset
    train_set = transform(data['UserId'], data['QuestionId'], data['IsCorrect'], batch_size) # for the final training on the whole dataset
    cdm = GDIRT(user_n, item_n) # init the model
    cdm.train(train_set, epoch=best_param[0], lr=best_param[1]) # train the model on the best hyperparameters
    cdm.save(f"{output_dir}/irt_{best_param[1]}.snapshot") # save the model
    with open(f"{output_dir}/irt_{best_param[1]}.txt", "w") as f: # write the results to output file
        f.write(f"Best parameters: {best_param}\n")
        f.write(f"Best validation accuracy: {best_accuracy}\n")
        f.write(f"Train accuracy: {cdm.eval(train_set)[1]}\n")
        f.close()
    return best_param

# Transform the data into the format required by EduCDM
test_set = transform(test_data["UserId"], test_data["QuestionId"], test_data["IsCorrect"], batch_size)

# Main config variable for the run of the model
n_lr = [0.0002, 0.002, 0.02] # Learning rate: set to 0.001 by default in the library
lr = None
t = True # Train the model
e = True # Evaluate the model

# Main code
output_dir = f"./output/"
if not os.path.exists(output_dir): os.makedirs(output_dir)
print("Initializing IRT model...")
cdm = GDIRT(user_n, item_n)
if t:
    print("Training the IRT model...")
    epoch, lr = train(train_data, output_dir=output_dir, epoch=5, n_lr=n_lr) # Train the model
if e and lr is not None:
    print("Evaluating the IRT model...")
    cdm.load(f"{output_dir}/irt_{lr}.snapshot")
    auc, accuracy_test = cdm.eval(test_set)
    print(f"auc: {auc}, accuracy: {accuracy_test}")
    with open(f"{output_dir}/irt_{lr}.txt", "a") as f:
        f.write(f"test auc: {auc}, test accuracy: {accuracy_test}\n")
        f.close()
    print("Evaluation complete.")