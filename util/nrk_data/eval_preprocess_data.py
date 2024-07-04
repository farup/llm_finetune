


def format_input_eval(sample): 
   #sample['prediction'] = f'Vennligst lag et sammendrag av artikkelen:\n{sample["bodyPlain"]}\n\nSammendrag:\n{sample["lead"]}'
   sample['eval'] = f'Vennligst lag et sammendrag av artikkelen:\n{sample["bodyPlain"]}\n\nSammendrag:'
   return sample


def tokenize_prompt_format(data_eval, tokenizer): 
   data_eval = data_eval.map(format_input_eval)
   eval_col = data_eval.select_columns('eval')
   eval_col = eval_col.map(lambda samples: tokenizer(samples['eval'], padding=True, truncation=True, return_tensors='pt', max_length=20), batched=True)
   data_eval = data_eval.add_column('eval_ids', eval_col['input_ids'])
   data_eval = data_eval.add_column('eval_mask', eval_col['attention_mask'])
   return data_eval
