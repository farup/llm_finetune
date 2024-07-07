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


def split_data(data, model_id, config): 

    test_size = config.get("test_size")
    stratify = config.get("stratify")
    model_id = model_id.split("/")[-1]
    processed_dataset_path = os.path.join(config.get("dataset_path_out"), model_id)
    
    print("CREATING TEST SPLIT ...")
    try:
        data = load_from_disk(processed_dataset_path)
    except FileNotFoundError as e:
        print("Error", e) 
    if stratify:
        data = data.class_encode_column("category")
        data = data.train_test_split(test_size=test_size, stratify_by_column='category')
    else: 
        data = data.train_test_split(test_size=test_size)
    data = data.remove_columns(["category"])

    return data['train'], data['test']


def format_tokenize_data(tokenizer, model_id, config):
    """
    Returns tokenized if exits, otherwise format,split, and tokenize data. 
    """
    model_id = model_id.split("/")[-1]

    dataset_path = config.get("input_data_processed_path")
    dataset_path_out = os.path.join(config.get("dataset_path_out"), model_id)
    
    if not os.path.exists(dataset_path_out):
        os.makedirs(dataset_path_out)
    
    if os.listdir(dataset_path_out):
        try: 
            data = load_from_disk(dataset_path_out)
            return data 
        except FileNotFoundError as e: 
            print("Error while loading file", e) 
    
    try:
        data = load_from_disk(dataset_path)
        data = data.select(range(2000))
    except FileNotFoundError as e:
        print("Error", e) 
        
    data = data.map(format_input)
    data = data.map(lambda samples: tokenizer(samples['prediction'], padding=True, truncation=True, max_length=tokenizer.model_max_length), batched=True)
  
    data = data.remove_columns(["prediction", "__index_level_0__"])
    data.save_to_disk(dataset_path_out)

    return data



