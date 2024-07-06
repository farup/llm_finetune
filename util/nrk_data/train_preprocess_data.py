import os

import pandas as pd
from datasets import load_dataset, load_metric, DatasetDict, load_from_disk
import datasets

from pynvml import *


def format_input(sample):
   """Formates the training data"""
   sample['prediction'] = f'Vennligst lag et sammendrag av artikkelen:\n{sample["bodyPlain"]}\n\nSammendrag:\n{sample["lead"]}'
   return sample

def format_input_eval(sample): 
   sample['prediction'] = f'Vennligst lag et sammendrag av artikkelen:\n{sample["bodyPlain"]}\n\nSammendrag:\n{sample["lead"]}'
   sample['eval'] = f'Vennligst lag et sammendrag av artikkelen:\n{sample["bodyPlain"]}\n\nSammendrag:'
   return sample

def tokenize_format_eval(data_eval, tokenizer): 
   data_eval = data_eval.map(format_input_eval)
   eval_col = data_eval.select_columns('eval')
   eval_col = eval_col.map(lambda samples: tokenizer(samples['eval'], padding=True, truncation=True, return_tensors='pt', max_length=20), batched=True)
   data_eval = data_eval.add_column('eval_ids', eval_col['input_ids'])
   data_eval = data_eval.add_column('eval_mask', eval_col['attention_mask'])
   return data_eval

def format_tokenize_data(tokenizer, dataset_path, dataset_path_out, test_size=0.2, stratify=False):
    """
    Returns tokenized if exits, otherwise format,split, and tokenize data. 
    """

    if os.path.exists(dataset_path_out):
        return load_from_disk(dataset_path_out)
    
    try:
        data = load_from_disk(dataset_path)
    except FileNotFoundError as e:
        print("Error", e) 
        
    
    print("CREATING TEST SPLIT ...")
    data = data.class_encode_column("category")
    if stratify:
        data = data.train_test_split(test_size=test_size, stratify_by_column='category')
    else: 
        data = data.train_test_split(test_size=test_size)
    
    data = data.remove_columns(["category"])

    train_data = data['train']
    eval_data = data['test']

    train_data = train_data.map(format_input)
    train_data = train_data.map(lambda samples: tokenizer(samples['prediction'], padding=True, truncation=True, max_length=tokenizer.model_max_length), batched=True)

    eval_data = tokenize_format_eval(eval_data)
   
    eval_data = eval_data.remove_columns(["prediction", "__index_level_0__"])
    train_data = train_data.remove_columns(["prediction","bodyPlain", "__index_level_0__"])

    data = DatasetDict({'train': train_data, 'test': eval_data}) 
    data.save_to_disk(dataset_path_out)

    return data


