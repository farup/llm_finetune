import os

import pandas as pd
from datasets import load_dataset, load_metric, DatasetDict, load_from_disk
import datasets


def determine_label(bodyPlain_split_len):
    """ Used for plotting and partioning dataset in bins"""

    if bodyPlain_split_len <= 500: 
        return 1 
    elif 500 < bodyPlain_split_len <= 1000: 
        return 2

    elif 1000 < bodyPlain_split_len <= 1500:
        return 3
    
    elif 1500 < bodyPlain_split_len <= 2000: 
        return 4

    elif 2000 < bodyPlain_split_len <= 2500: 
        return 5
    
    elif 2500 < bodyPlain_split_len <= 3000: 
        return 6
    
    elif 3000 < bodyPlain_split_len <= 3500: 
        return 7
    
    elif 3500 < bodyPlain_split_len <= 4000: 
        return 8
    
    elif bodyPlain_split_len > 4000: 
        return 9


def remove_rows_empty_string(df):
    missing_bodyPlain_indices = df[df['bodyPlain'] == ''].index
    missing_lead_indices = df[df['lead'] == ''].index

    union = set(missing_bodyPlain_indices).union(set(missing_lead_indices))
    df = df.drop(union)
    return df

def preprocess_data(dataset_path, dataset_path_out):
    """ 
    Reads data to pandas dataframe.
    Removes rows with empty strings 
    """

    if os.path.exists(dataset_path_out) and len(os.listdir(dataset_path_out)) >= 1: 
        return 
    
    if not os.path.exists(dataset_path_out):
        os.makedirs(dataset_path_out)

    df = pd.read_json(dataset_path, lines=True)
    df = remove_rows_empty_string(df)

    df['category'] = df.apply(lambda x: determine_label(x['wordCount']), axis=1)
    df = df[['lead', 'bodyPlain', 'category']]
    data = datasets.Dataset.from_pandas(df)
    data.save_to_disk(dataset_path_out)
    print("Done!")


if __name__ == "__main__":

    dataset_path = '/cluster/home/terjenf/norwAI_All/NorGLMFinetune2/dataset/nrk-articles.jsonl'
    dataset_path_out = "/cluster/home/terjenf/norwAI_All/finetune/data/processed"
    preprocess_data(dataset_path, dataset_path_out)
