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


class CustomEvalTrainer(Trainer): 


    def __init__(self, *args, **kwargs): 
        super(CustomEvalTrainer, self).__init__(*args, **kwargs)

        self.text_table= wandb.Table(columns=["Mode", "Eval Prompt", "Eval Generated", "Eval Ground Truth", "Bleu (%)", "RougeL (%)"])
       
    def evaluate(self, eval_dataset= None, ignore_keys = None, metric_key_prefix = "eval"):
        bleu = evaluate.load("sacrebleu")
        rouge = evaluate.load("rouge")

        print(self.state.global_step)

        bleu_greedy = evaluate.load("sacrebleu")
        rouge_greedy = evaluate.load('rouge')
        perplexity_greedy = evaluate.load('perplexity')

        bleu_beam = evaluate.load("sacrebleu")
        rouge_beam = evaluate.load('rouge')
        perplexity_beam = evaluate.load('perplexity')


        self.model.eval()
        with torch.no_grad():
            output = super().evaluate(eval_dataset=self.eval_dataset.select_columns(['input_ids', 'attention_mask']), ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)


        eval_loader = DataLoader(dataset=self.eval_dataset, batch_size=self.args.eval_batch_size)

        pbar = tqdm(eval_loader, desc="Bleu and Rouge Evaluation")
   
        with torch.no_grad():
            for batch in pbar:
        
                print(torch.cuda.mem_get_info())
                print(torch.cuda.memory_allocated())
                
                for mode in ["greedy", "beam"]: 
                    start_time = time.time()
                    print("Start time", start_time)
                    if mode == "greedy":
                        tokens = self.model.generate(input_ids=batch['eval_ids'], attention_mask=batch['eval_mask'].to(device), max_new_tokens=50, do_sample=False)
                    else: 
                        tokens = self.model.generate(input_ids=batch['eval_ids'], attention_mask=batch['eval_mask'].to(device), max_new_tokens=50, num_beams=3, early_stopping=True)

                    generate_time = time.time() - start_time
                    print("Generated time", generate_time)

                    wandb.log({'Generation time per sample': generate_time / self.args.eval_batch_size})

                    decoded = self.tokenizer.batch_decode(tokens)

                    for i in range(self.args.eval_batch_size):
                        print([batch['lead'][i]])
                        bleu_scores = bleu.compute(predictions=[decoded[i]], references=[batch['lead'][i]])
                        rouge_scores = rouge.compute(predictions=[decoded[i]], references=[batch['lead'][i]])
                        self.text_table.add_data(mode, batch['eval'][i], decoded[i], batch['lead'][i], round(bleu_scores.get("score"),4), round(rouge_scores.get("rougeL"),4))
                    
                
                    new_table = wandb.Table(
                        columns=self.text_table.columns, data=self.text_table.data
                    )
                    
                    wandb.log({"Evaluation samples": new_table})

                    if mode == "greedy":
                        bleu_greedy.add_batch(references=batch['lead'], predictions=decoded)
                        rouge_greedy.add_batch(references=batch['lead'], predictions=decoded)
                        #perplexity_greedy.add_batch(predictions=decoded)

                    else: 
                        bleu_beam.add_batch(references=batch['lead'], predictions=decoded)
                        rouge_beam.add_batch(references=batch['lead'], predictions=decoded)
                        #perplexity_beam.add_batch(predictions=decoded)

                    pbar.set_postfix()
                break

            bleu_res_g = bleu_greedy.compute()
            rouge_res_g = rouge_greedy.compute()
            #perplexity_res_g = perplexity_greedy.compute(model_id=args.model_id)

            bleu_res_b = bleu_beam.compute()
            rouge_res_b = rouge_beam.compute()
            #perplexity_res_b = perplexity_beam.compute(model_id=args.model_id)

            #wandb.log({'eval/loss': output})
            wandb.log({'bleau_greedy': bleu_res_g, 'rouge_greedy': rouge_res_g})
            wandb.log({'bleau_beam': bleu_res_b, 'rouge_beam': rouge_res_b})
            return output
        
            # wandb.log({'bleau_greedy': bleu_res_g, 'rouge_greedy': rouge_res_g, 'perplexity_greedy': perplexity_res_g})
            # wandb.log({'bleau_beam': bleu_res_b, 'rouge_beam': rouge_res_b, 'perplexity_beam': perplexity_res_b})


