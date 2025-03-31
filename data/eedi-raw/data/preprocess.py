import numpy as np
import pandas as pd
from scipy import stats

def analysis(data: pd.DataFrame):
    # Check if a student has answered the same question more than once
    df = data[['UserId', 'QuestionId']]
    print("Any duplicate answers?", df.duplicated().any())

def filter_by_percentile(column: str, data: pd.DataFrame, top_percentile: float, bottom_percentile: float):
    counter = data[column].value_counts(dropna=True).reset_index(name='count')
    top_percentile = counter['count'].quantile(top_percentile)
    bottom_percentile = counter['count'].quantile(bottom_percentile)
    filtered_counter = counter[(counter['count'] < top_percentile) & (counter['count'] > bottom_percentile)]
    filtered_data = data[data[column].isin(filtered_counter[column])]
    return filtered_data

# [19834813 rows x 4 columns] with test and train data

def reduce_data():
    print('Starting preprocessing...')
    df = pd.read_csv('./train_data/train_task_1_2.csv')
    df.drop(columns=['CorrectAnswer', 'AnswerValue'], inplace=True) # This is used for Task 2 of the competition

    # Filtering data to reduce data size ===
    # Merge the test data with the train data
    test1 = pd.read_csv('./test_data/test_public_answers_task_1.csv')
    test2 = pd.read_csv('./test_data/test_private_answers_task_1.csv')
    df = pd.concat([df, test1], ignore_index=True)
    df = pd.concat([df, test2], ignore_index=True)
    #analysis(df) # There is no students who answered the same question more than once
    print("Shape of original test and train data:", df.shape)

    # Reduce data size by filtering
    df = filter_by_percentile('QuestionId', df, 0.8, 0.2) # Remove 'outliers' (count too far from the mean)
    df = filter_by_percentile('UserId', df, 0.8, 0.2) # Remove 'outliers' (count too far from the mean)

    # Merge data with metadata
    question_metadata = pd.read_csv('./metadata/question_metadata_task_1_2.csv')
    answer_metadata = pd.read_csv('./metadata/answer_metadata_task_1_2.csv')[['AnswerId', 'Confidence']]
    df = pd.merge(df, answer_metadata, left_on='AnswerId', right_on='AnswerId', how='left')
    df = pd.merge(df, question_metadata, left_on='QuestionId', right_on='QuestionId', how='left')

    df = df.dropna() # drop the rows with NaN values
    print(df.shape)

    # One-Hot Encoding the subjects in SubjectId
    subjects = set(s for sublist in df['SubjectId'] for s in list(map(int, sublist.strip('[]').split(', '))))
    print("Number of unique subjects:", len(subjects))
    for s in subjects:
        df[s] = df['SubjectId'].apply(lambda x: 1 if s in list(map(int, x.strip('[]').split(', '))) else 0)

    pd.to_pickle(df, './train.pkl')

if __name__ == '__main__':
    reduce_data()
    df = pd.read_pickle('./train.pkl')
    df.drop(columns=['SubjectId', 'UserId', 'QuestionId', 'AnswerId'], inplace=True)
    pd.to_pickle(df, './train_v3.pkl')
    df.to_csv('./processed_eedi.csv', index=False)