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
train_data, valid_data = train_test_split(train_data, stratify=train_data['IsCorrect'], random_state=0)
df_item = pd.read_csv('../data/eedi-no-one-hot/metadata/question_metadata.csv')
item2knowledge = {}
knowledge_set = set()
# VERY IMPORTANT - keeps the code from bugging out as embeddings depend on it
train_data.reset_index(drop=True, inplace=True)
valid_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)
# Create mapping from the subjects (knowledge codes) to the items
for i, s in df_item.iterrows():
    item_id, knowledge_codes = s['QuestionId'], list(set(eval(s['SubjectId'])))
    item2knowledge[item_id] = knowledge_codes
    knowledge_set.update(knowledge_codes)

# Transformation parameters
batch_size = 32
user_n = np.max(train_data['UserId'])
item_n = np.max([np.max(train_data['QuestionId']), np.max(valid_data['QuestionId']), np.max(test_data['QuestionId'])])
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

# Transform the data into the format required by EduCDM
train_set, valid_set, test_set = [
    transform(data["UserId"], data["QuestionId"], item2knowledge, data["IsCorrect"], batch_size)
    for data in [train_data, valid_data, test_data]
]

# The only hyperparameter in the NCDM model is learning rate
# which is set to 0.002 by default
lr = 0.002 # 0.002, 0.02, 0.2
t = True # Train the model
e = True # Evaluate the model

# Main code
output_dir = f"./output/{lr}"
if not os.path.exists(output_dir): os.makedirs(output_dir)
print("Initializing NCDM model...")
cdm = NCDM(knowledge_n, item_n, user_n)
if t:
    print("Training NCDM model...")
    cdm.train(train_set, valid_set, epoch=3, lr=lr, device="cpu")
    cdm.save(f"{output_dir}/ncdm_{lr}.snapshot")
if e:
    cdm.load(f"{output_dir}/ncdm_{lr}.snapshot")
    _, accuracy_train = cdm.eval(train_set)
    auc, accuracy_test = cdm.eval(test_set)
    print("train accuracy:", accuracy_train)
    print(f"auc: {auc}, accuracy: {accuracy_test}")
    with open(f"{output_dir}/ncdm_{lr}.txt", "w") as f:
        f.write(f"auc: {auc}, accuracy: {accuracy_test}")
        f.write(f"train accuracy: {accuracy_train}")
        f.close()
    print("Evaluation complete.")