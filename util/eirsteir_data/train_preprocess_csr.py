import json
import os
import pandas as pd
import numpy as np
import time

import datasets 
from datasets import concatenate_datasets, DatasetDict

import torch

import pandas as pd
from datasets import load_dataset, load_metric, DatasetDict, load_from_disk
import datasets

from pynvml import *


def format_tokenize_data(tokenizer, model_id, config): 

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
    
    else:
        hf_data_enc = process_json_data(dataset_path)

        hf_data_enc = hf_data_enc.map(format_dataset)
        # hf_data_enc = hf_data_enc.map(lambda samples: tokenizer(samples['prediction'], padding=True, truncation=True, max_length=tokenizer.model_max_length), batched=False)
        hf_data_enc = hf_data_enc.map(lambda samples: tokenizer(samples['prediction'], padding=True, truncation=True), batched=True)

        data = custom_split(hf_data_enc, eval_test_size=0.2)

        data['eval'] = format_eval(data['eval'], tokenizer)
        
        data.save_to_disk(dataset_path_out)

        return data

def calc_add_uni(datadf): 
    uni = lambda x: x['university']

    uni_per_sample = datadf.metadata.apply(uni)

    unq = np.unique(np.asarray(uni_per_sample))

    map_uni_to_id = dict()
    for key, val in zip(unq, range(len(unq))):
        map_uni_to_id[str(key)] = val

    uni_id_per_sample = uni_per_sample.apply(lambda x: map_uni_to_id[x])

    datadf['uni'] = uni_per_sample
    
    return datadf


def format_eval(datahf, tokenizer):
    eval_prompts = []
    eval_ids = []
    eval_mask = []
    eval_gt = []

    for conv in datahf['messages']: 
        chat_seq, gt_ai_answer = create_prompts(conv)


        # out = tokenizer(chat_seq, padding=True, truncation=True, return_tensors='pt')
        # eval_ids.append(out['input_ids'])
        # eval_mask.append(out['attention_mask'])
        eval_prompts.append(chat_seq)
        eval_gt.append(gt_ai_answer)


    datahf = datahf.add_column("eval_prompts", eval_prompts)
    datahf = datahf.add_column("eval_gt", eval_gt)
    # datahf = datahf.add_column("eval_ids", eval_ids)
 
    # eval_col = datahf.select_columns('eval_prompts')
    #eval_col = eval_col.map(lambda x: tokenize_eval(x, tokenizer))

    # eval_col = eval_col.map(lambda samples: tokenizer(samples['eval_prompts'], padding=True, truncation=True, return_tensors='pt'), batched=False)
    # datahf = datahf.add_column('eval_ids', eval_col['input_ids'])
    # datahf = datahf.add_column('eval_mask', eval_col['attention_mask'])

    return datahf
    

def tokenize_eval(samples, tokenizer): 
    tokens_list = []
    for conv in samples['eval_prompts']:
        tokenizer
        tokens = tokenizer(conv['eval_prompts'], padding=True, truncation=True, return_tensors='pt')
        tokens_list.append(tokens)
    return torch.stack(tokens_list)
    

def create_prompts(conv): 
    chat_seq = []
    start = 0
    gt_ai_answer = []
    for chat_ind in range(len(conv))[::2]: 
        text = ""
        if chat_ind == 0: 
            continue
        for chat in conv[start:chat_ind-1]:
            text += f"{chat['speaker']}: {chat['content']}\n"

        # seq = [f"{chat['speaker']}: {chat['content']}" for chat in conv[start:chat]]   
        chat_seq.append(text)
        gt_ai_answer.append(conv[chat_ind-1]['content'])

    return chat_seq, gt_ai_answer
    

def process_json_data(dataset_path):
    pandas_dataframe = load_data_from_file(dataset_path)
    hf_data = calc_add_uni(pandas_dataframe)
    hf_data = datasets.Dataset.from_pandas(pandas_dataframe)
    
    hf_data_enc = hf_data.class_encode_column('uni')
    return hf_data_enc
    

def formate_conversation(conversation):
    text = "" 
    for chat in conversation: 
        text += f"{chat['speaker']}: {chat['content']}\n"

    return text

def format_dataset(sample): 
    text = formate_conversation(sample['messages'])
    sample['prediction'] = text
    return sample

def custom_split(dataset, eval_test_size):
  
    train_evaltest = dataset.train_test_split(test_size=eval_test_size, stratify_by_column="uni")
    train_data = train_evaltest['train']

    eval_test = train_evaltest['test'].train_test_split(test_size=0.5, stratify_by_column="uni")
    eval_data = eval_test['train']
    test_data = eval_test['test']

    eval_stanford = eval_data.filter(lambda x: x['uni']==3)
    eval_rest = eval_data.filter(lambda x: x['uni']!=3)

    train_eval_rest_data = concatenate_datasets([train_data, eval_rest])

    data = DatasetDict({'train':  train_eval_rest_data, 'eval': eval_stanford, 'test': test_data})
    return data
     

def load_data_from_file(dataset_path): 
    with open(dataset_path, 'r', encoding='utf-8') as file: 
        data = json.load(file)
        df = pd.DataFrame.from_dict(data)

    return df




