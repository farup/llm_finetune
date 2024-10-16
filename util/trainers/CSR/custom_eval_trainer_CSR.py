import torch 
import torch.nn as nn
import random

import evaluate
import wandb

from transformers import Trainer, TrainerCallback, TrainingArguments, TrainerState
import torch.nn.functional as F
from torch.utils.data import DataLoader

from typing import Dict, Union, Any, Optional, List, Tuple
from tqdm import tqdm
import time
import evaluate


class CustomEvalTrainer_CSR(Trainer): 

    def __init__(self, *args, **kwargs): 
        super(CustomEvalTrainer_CSR, self).__init__(*args, **kwargs)

        self.text_table= wandb.Table(columns=["Mode","Global Update Step","Epoch",  "Eval Prompt", "Eval Generated", "Eval Ground Truth", "Bleu (%)", "RougeL (%)"])
    

    def custom_collate_fn(self, batch):
        # Extract input_ids and labels from the batch
        
        eval_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch], dim=0)
        eval_mask = torch.stack([torch.tensor(item['attention_mask']) for item in batch], dim=0)

        prediction = [item['prediction'] for item in batch]
        eval_prompts = [item['eval_prompts'] for item in batch]
        
        eval_gt = [item['eval_gt'] for item in batch]
    
        return {
            'input_ids': eval_ids,
            'attention_mask': eval_mask,
            'prediction': prediction,
            'eval_prompts': eval_prompts, 
            'eval_gt': eval_gt,
        }
    

    def evaluate(self, eval_dataset= None, ignore_keys = None, metric_key_prefix = "eval"):
       
        bleu = evaluate.load("sacrebleu")
        rouge = evaluate.load("rouge")

        bleu_greedy = evaluate.load("sacrebleu")
        rouge_greedy = evaluate.load('rouge')
        perplexity_greedy = evaluate.load('perplexity')

        bleu_beam = evaluate.load("sacrebleu")
        rouge_beam = evaluate.load('rouge')
        perplexity_beam = evaluate.load('perplexity')

        self.model.eval()
        with torch.no_grad():
            output = super().evaluate(eval_dataset=self.eval_dataset.select_columns(['input_ids', 'attention_mask']), ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

        
        eval_loader = DataLoader(dataset=self.eval_dataset, batch_size=self.args.eval_batch_size, collate_fn=self.custom_collate_fn)
        #eval_loader = self.get_eval_dataloader()

        pbar = tqdm(eval_loader, desc="Bleu and Rouge Evaluation")


   
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                eval_tokenized = [self.tokenizer(conv, padding=True, truncation=True, return_tensors='pt') for conv in batch['eval_prompts']]
            
                for conv_nr, conv in enumerate(eval_tokenized):

                    input_ids = conv['input_ids'].to(self.model.device)
                    attention_mask = conv['attention_mask'].to(self.model.device)

                    print(torch.cuda.mem_get_info())
                    print(torch.cuda.memory_allocated())
                    
                    for mode in ["greedy", "beam"]: 
                        start_time = time.time()
                        print("Start time", start_time)
                        if mode == "greedy":
                            tokens = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=150, do_sample=False)
                        else: 
                            tokens = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=150, num_beams=3, early_stopping=True)

                        generate_time = time.time() - start_time
                        print("Generated time", generate_time)

                        wandb.log({'Generation time per sample': generate_time / self.args.eval_batch_size})

                        decoded = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

                        for i in range(len(decoded)):
                            bleu_scores = bleu.compute(predictions=[decoded[i]], references=[batch['eval_gt'][conv_nr][i]])
                            rouge_scores = rouge.compute(predictions=[decoded[i]], references=[batch['eval_gt'][conv_nr][i]])
                            self.text_table.add_data(mode, self.state.global_step, self.state.epoch, batch['eval_prompts'][conv_nr][i], decoded[i], batch['eval_gt'][conv_nr][i], round(bleu_scores.get("score"),4), round(rouge_scores.get("rougeL"),4))
                        
                    
                        new_table = wandb.Table(
                            columns=self.text_table.columns, data=self.text_table.data
                        )
                        
                        wandb.log({"Evaluation samples": new_table})

                        if mode == "greedy":
                            bleu_greedy.add_batch(references=batch['eval_gt'][conv_nr], predictions=decoded)
                            rouge_greedy.add_batch(references=batch['eval_gt'][conv_nr], predictions=decoded)
                            #perplexity_greedy.add_batch(predictions=decoded)

                        else: 
                            bleu_beam.add_batch(references=batch['eval_gt'][conv_nr], predictions=decoded)
                            rouge_beam.add_batch(references=batch['eval_gt'][conv_nr], predictions=decoded)
                            #perplexity_beam.add_batch(predictions=decoded)

                        pbar.set_postfix()
                    break
                break

            bleu_res_g = bleu_greedy.compute()
            rouge_res_g = rouge_greedy.compute()
            #perplexity_res_g = perplexity_greedy.compute(model_id=args.model_id)

            bleu_res_b = bleu_beam.compute()
            rouge_res_b = rouge_beam.compute()
            #perplexity_res_b = perplexity_beam.compute(model_id=args.model_id)

            #wandb.log({'eval/loss': output})
            wandb.log({'bleau_greedy': round(bleu_res_g.get('score'),4), 'rouge_L_greedy': round(rouge_res_g.get('rougeL'),4)})
            wandb.log({'bleau_beam': round(bleu_res_b.get('score'),4), 'rouge_L_beam': round(rouge_res_b.get('rougeL'),4)})
            return output
        
            # wandb.log({'bleau_greedy': bleu_res_g, 'rouge_greedy': rouge_res_g, 'perplexity_greedy': perplexity_res_g})
            # wandb.log({'bleau_beam': bleu_res_b, 'rouge_beam': rouge_res_b, 'perplexity_beam': perplexity_res_b})


