import argparse
import torch
import time 
import os
import sys
import torch.nn as nn
import evaluate
import wandb

from distutils.util import strtobool
from dotenv import load_dotenv
from pathlib import Path

from accelerate import Accelerator
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from torch.utils.data import TensorDataset, DataLoader, Dataset
from datasets import load_dataset, load_from_disk
from tqdm.auto import tqdm


CUDA_LAUNCH_BLOCKING=1
sys.path.append("/cluster/home/terjenf/norwAI_All/finetune")

load_dotenv()

os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

bleu = evaluate.load("sacrebleu")
rouge = evaluate.load("rouge")

bleu_greedy = evaluate.load("sacrebleu")
rouge_greedy = evaluate.load('rouge')
perplexity_greedy = evaluate.load('perplexity')

bleu_beam = evaluate.load("sacrebleu")
rouge_beam = evaluate.load('rouge')
perplexity_beam = evaluate.load('perplexity')

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
#device = torch.device("cpu")

def parse_args(): 
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiement')
    
    parser.add_argument('--model-id', type=str, default=None,
                        help='the model id to the base model')
    
    parser.add_argument('--peft-path', type=str, default=None,
                        help='path to peft checkpoints')

    parser.add_argument('--dataset-type', type=str, default="nrk",
                        help='dataset to evaluate on')
    
    parser.add_argument('--input-data-path', type=str, default=None,
                        help='path to processed data')
    
    parser.add_argument('--data-size', type=int, default=None,
                        help='limit dataset size')
    
    parser.add_argument('--batch-size', type=int, default=2,
                        help='batch size for evaluation')
    
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    
    parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=True,
                        help='if toggled, this experiment will be tracked with weights and Biases')
    parser.add_argument('--wandb-project-name', type=str, default="eval_norwai",
                        help="the wandb's roject name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")
    
    args = parser.parse_args()
    return args

def load_model_tokenizer(model_id, pathPeftModel=None):

    tokenizer = AutoTokenizer.from_pretrained(model_id,device_map=device,torch_dtype=torch.bfloat16)                                         
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, torch_dtype=torch.bfloat16)
    print("LOADED PRETRAINED MODEL AND TOKENIZER")

    # Load the Lora adapter to base model.
    if pathPeftModel is not None: 
        model = PeftModel.from_pretrained(model, pathPeftModel)
        print("Peft Model : ", model.device)
        print(f"Running merge_and_unload")
        model = model.merge_and_unload()
    tokenizer.pad_token = tokenizer.eos_token
    print("CONNECTED LORA ADAPATERS.")
    return model, tokenizer


def load_prepare_data(data_path):
    if os.path.exists(path=data_path):
        data = load_from_disk(data_path)
    else:
        raise"Data does not exits!"
    
    return data['test']

def custom_collate_fn(batch):
    # Extract input_ids and labels from the batch
    
    input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch], dim=0)
    attention_mask = torch.stack([torch.tensor(item['attention_mask']) for item in batch], dim=0)

    eval_ids = torch.stack([torch.tensor(item['eval_ids']) for item in batch], dim=0)
    eval_mask = torch.stack([torch.tensor(item['eval_mask']) for item in batch], dim=0)

    #bodyPlain = [item['bodyPlain'] for item in batch]
    eval = [item['eval'] for item in batch]
    lead = [item['lead'] for item in batch]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'eval': eval, 
        'lead': lead,
        'eval_ids': eval_ids,
        'eval_mask': eval_mask
    }


if __name__ == "__main__": 
    args = parse_args()

    run_name = f"{args.model_id.split('/')[-1]}_{args.exp_name}_{int(time.time())}"

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name, 
            entity=args.wandb_entity,  
            config=vars(args),
            name=run_name, 
            save_code=True, 
        )
        text_table= wandb.Table(columns=["Mode", "Eval Prompt", "Eval Generated", "Eval Ground Truth", "Bleu", "RougeL"])
    
    data_eval = load_prepare_data(args.input_data_path)

    if args.data_size is not None: 
        data_eval = data_eval.select(range(args.data_size))

    model, tokenizer = load_model_tokenizer(args.model_id, args.peft_path)

    if args.dataset_type == "nrk":
        from util.nrk_data.eval_preprocess_data import tokenize_prompt_format
        data_eval = tokenize_prompt_format(data_eval, tokenizer)
    else: 
        raise Exception("Only nrk dataset supported")


    print(torch.cuda.mem_get_info())
    print(torch.cuda.memory_allocated())

    data_loader = DataLoader(dataset=data_eval, batch_size=args.batch_size, collate_fn=custom_collate_fn)
    
    loss_fn = nn.CrossEntropyLoss()


    pbar = tqdm(data_loader, desc="Evaluation")
    running_eval_loss =0

    with torch.no_grad():
        for batch in pbar:
    
            print(torch.cuda.mem_get_info())
            print(torch.cuda.memory_allocated())
            
            for mode in ["greedy", "beam"]: 
                start_time = time.time()
                print("Start time", start_time)
                if mode == "greedy":
                    tokens = model.generate(input_ids=batch['eval_ids'].to(device), attention_mask=batch['eval_mask'].to(device), max_new_tokens=50, do_sample=False)
                else: 
                    tokens = model.generate(input_ids=batch['eval_ids'].to(device), attention_mask=batch['eval_mask'].to(device), max_new_tokens=50, num_beams=3, early_stopping=True)

                generate_time = time.time() - start_time
                print("Generated time", generate_time)

                if args.track:
                    wandb.log({'Generation time per sample': generate_time / args.batch_size})

                decoded = tokenizer.batch_decode(tokens)

                for i in range(args.batch_size):
                    print([batch['lead'][i]])
                    bleu_scores = bleu.compute(predictions=[decoded[i]], references=[batch['lead'][i]])
                    rouge_scores = rouge.compute(predictions=[decoded[i]], references=[batch['lead'][i]])
                    text_table.add_data(mode, batch['eval'][i], decoded[i], batch['lead'][i], round(bleu_scores.get("score"),4), round(rouge_scores.get("rougeL"),4))
                
                if args.track:
                    new_table = wandb.Table(
                        columns=text_table.columns, data=text_table.data
                    )
                    
                    wandb.log({"Evaluation samples": new_table})

                if mode == "greedy":
                    bleu_greedy.add_batch(references=batch['lead'], predictions=decoded)
                    rouge_greedy.add_batch(references=batch['lead'], predictions=decoded)
                    perplexity_greedy.add_batch(predictions=decoded)

                else: 
                    bleu_beam.add_batch(references=batch['lead'], predictions=decoded)
                    rouge_beam.add_batch(references=batch['lead'], predictions=decoded)
                    perplexity_beam.add_batch(predictions=decoded)

                pbar.set_postfix()
            break
        
        bleu_res_g = bleu_greedy.compute()
        rouge_res_g = rouge_greedy.compute()
        perplexity_res_g = perplexity_greedy.compute(model_id=args.model_id)

        bleu_res_b = bleu_beam.compute()
        rouge_res_b = rouge_beam.compute()
        perplexity_res_b = perplexity_beam.compute(model_id=args.model_id)

        if args.track: 
            wandb.log({'bleau_greedy': bleu_res_g, 'rouge_greedy': rouge_res_g, 'perplexity_greedy': perplexity_res_g})
            wandb.log({'bleau_beam': bleu_res_b, 'rouge_beam': rouge_res_b, 'perplexity_beam': perplexity_res_b})


# now = datetime.datetime.now()
# dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

# folder = "/cluster/home/terjenf/norwAI_All/NorwAI_finetune/NorGLM_spring2024/evaluate/results"
# folder_path = os.path.join(folder, model_id.split("/")[-1])

# if not os.path.exists(folder_path):
#     os.makedirs(folder_path)

# df = pd.DataFrame({
#     'Model_id': model_id, 
#     "bleu_res": bleu_res, 
#     "perplexity_res": perplexity_res, 
#     "rouge_score":  rouge_res})

# df.to_csv(os.path.join(folder_path, f"{dt_string}.csv"))


   

       







