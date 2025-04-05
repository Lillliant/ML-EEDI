# This is adapted from the NCDM example in EduCDM
# Modified with the dataset processed from EEDI
from EduCDM import NCDM
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
item2knowledge = {}
knowledge_set = set()
# VERY IMPORTANT - keeps the code from bugging out as embeddings depend on it
train_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)
# Create mapping from the subjects (knowledge codes) to the items
for i, s in df_item.iterrows():
    item_id, knowledge_codes = s['QuestionId'], list(set(eval(s['SubjectId'])))
    item2knowledge[item_id] = knowledge_codes
    knowledge_set.update(knowledge_codes)

# Transformation parameters
batch_size = 32
user_n = np.max(train_data['UserId'])
item_n = np.max([np.max(train_data['QuestionId']), np.max(test_data['QuestionId'])])
knowledge_n = np.max(list(knowledge_set))

# The original transformation code from the example
def transform(user, item, item2knowledge, score, batch_size):
    knowledge_emb = torch.zeros((len(item), knowledge_n))
    for idx in range(len(item)):
        knowledge_emb[idx][np.array(item2knowledge[item[idx]]) - 1] = 1.0

    data_set = TensorDataset(
        torch.tensor(user, dtype=torch.int64) - 1,  # (1, user_n) to (0, user_n-1)
        torch.tensor(item, dtype=torch.int64) - 1,  # (1, item_n) to (0, item_n-1)
        knowledge_emb,
        torch.tensor(score, dtype=torch.float32)
    )
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)

# Merge two folds into one (training fold)
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
            valid_set = transform(fold["UserId"], fold["QuestionId"], item2knowledge, fold["IsCorrect"], batch_size)
            train_set = merge(folds[(i+1)%3], folds[(i+2)%3]) # Merge the other two folds
            train_set = transform(train_set["UserId"], train_set["QuestionId"], item2knowledge, train_set["IsCorrect"], batch_size)
            cdm = NCDM(knowledge_n, item_n, user_n) # init the model
            cdm.train(train_set, epoch=epoch, lr=lr)
            _, accuracy = cdm.eval(valid_set)
            fold_accuracy.append(accuracy)
        accuracy = np.mean(fold_accuracy) # average the accuracy of the 3 folds
        if accuracy > best_accuracy: 
            best_accuracy = accuracy
            best_param = (epoch, lr)
        print(f"Parameter {(epoch, lr)}, Validation accuracy: {accuracy}")
    # retrain the model with the best parameters on the whole dataset
    train_set = transform(data["UserId"], data["QuestionId"], item2knowledge, data["IsCorrect"], batch_size) # for the final training on the whole dataset
    cdm = NCDM(knowledge_n, item_n, user_n) # init the model
    cdm.train(train_set, epoch=best_param[0], lr=best_param[1]) # train the model on the best hyperparameters
    cdm.save(f"{output_dir}/ncdm_{best_param[1]}.snapshot") # save the model
    with open(f"{output_dir}/ncdm_{best_param[1]}.txt", "w") as f: # write the results to output file
        f.write(f"Best parameters: {best_param}\n")
        f.write(f"Best validation accuracy: {best_accuracy}\n")
        f.write(f"Train accuracy: {cdm.eval(train_set)[1]}\n")
        f.close()
    return best_param

# Transform the data into the format required by EduCDM
test_set = transform(test_data["UserId"], test_data["QuestionId"], item2knowledge, test_data["IsCorrect"], batch_size)

# The only hyperparameter in the NCDM model is learning rate
# which is set to 0.002 by default
n_lr = [0.0002, 0.002, 0.02] # learning rates to evaluate
lr = None
t = True # Train the model
e = True # Evaluate the model

# Main code
output_dir = f"./output/{lr}"
if not os.path.exists(output_dir): os.makedirs(output_dir)
print("Initializing NCDM model...")
cdm = NCDM(knowledge_n, item_n, user_n)
if t:
    print("Training NCDM model...")
    epoch, lr = train(train_data, output_dir=output_dir, epoch=10, n_lr=n_lr) # Train the model
if e:
    print("Evaluating the NCDM model...")
    cdm.load(f"{output_dir}/ncdm_{lr}.snapshot")
    auc, accuracy_test = cdm.eval(test_set)
    print(f"auc: {auc}, accuracy: {accuracy_test}")
    with open(f"{output_dir}/ncdm_{lr}.txt", "a") as f:
        f.write(f"test auc: {auc}, test accuracy: {accuracy_test}\n")
        f.close()
    print("Evaluation complete.")