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
train_data, valid_data = train_test_split(train_data, stratify=train_data['IsCorrect'], random_state=0)
df_item = pd.read_csv('../data/eedi-no-one-hot/metadata/question_metadata.csv')
# VERY IMPORTANT - keeps the code from bugging out as embeddings depend on it
train_data.reset_index(drop=True, inplace=True)
valid_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)

# Transformation parameters
batch_size = 32
user_n = np.max([np.max(train_data['UserId']), np.max(valid_data['UserId']), np.max(test_data['UserId'])])*batch_size
item_n = np.max([np.max(train_data['QuestionId']), np.max(valid_data['QuestionId']), np.max(test_data['QuestionId'])])*batch_size

def transform(x, y, z, batch_size, **params):
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.int64),
        torch.tensor(y, dtype=torch.int64),
        torch.tensor(z, dtype=torch.float32)
    )
    return DataLoader(dataset, batch_size=batch_size, **params)

# Transform the data into the format required by EduCDM
train_set, valid_set, test_set = [
    transform(data["UserId"], data["QuestionId"], data["IsCorrect"], batch_size)
    for data in [train_data, valid_data, test_data]
]

# The only hyperparameter in the NCDM model is learning rate
# which is set to 0.001 by default
lr = 0.0002 # 0.0002 0.002, 0.02
t = True # Train the model
e = True # Evaluate the model

# Main code
output_dir = f"./output/{lr}"
if not os.path.exists(output_dir): os.makedirs(output_dir)
print("Initializing IRT model...")
cdm = GDIRT(user_n, item_n)
if t:
    cdm.train(train_set, valid_set, epoch=2, lr=lr)
    cdm.save(f"{output_dir}/irt_{lr}.snapshot")
if e:
    cdm.load(f"{output_dir}/irt_{lr}.snapshot")
    _, accuracy_train = cdm.eval(train_set)
    auc, accuracy_test = cdm.eval(test_set)
    print("train accuracy:", accuracy_train)
    print(f"auc: {auc}, accuracy: {accuracy_test}")
    with open(f"{output_dir}/irt_{lr}.txt", "w") as f:
        f.write(f"auc: {auc}, accuracy: {accuracy_test}\n")
        f.write(f"train accuracy: {accuracy_train}")
        f.close()
    print("Evaluation complete.")