import os
import sysconfig
import gc
import sys
import logging
import argparse

from pynvml import *
from dotenv import load_dotenv

from accelerate import Accelerator
import torch
import torch.nn as nn

import evaluate
import wandb
import transformers

from transformers import AutoTokenizer, AutoModelForCausalLM, EvalPrediction
from peft import LoraConfig, get_peft_model 

from util.nrk_data.train_preprocess_data import preprocess_data, format_tokenize_data

load_dotenv()
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

logger = logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

bleu = evaluate.load("sacrebleu")
rouge = evaluate.load('rouge')

print('START GPU INFO')
print_gpu_utilization()


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

def load_model_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16, 
    )
    tokenizer.pad_token = tokenizer.eos_token

    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

def freeze_pretrained(model):
    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)

    model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float16)

def set_up_model(model, rank,lora_alpha=32, lora_dropout=0.05):
    config = LoraConfig(
    r=rank, #attention heads, rank of the attention matrix, i think
    lora_alpha= lora_alpha, #alpha scaling, scaling factor for the weight matrices
    # target_modules=["q_proj", "v_proj"], #will be set after i know the names
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM" # set this for CLM or Seq2Seq
    )

    model = get_peft_model(model, config)
    return model


def train(model, config=None):
        print("SETTING MODEL")
        model_peft = set_up_model(model, config["lora_rank"],config["lora_alpha"],config["lora_dropout"])

        print('LOADED PEFT MODEL')
        print_gpu_utilization()
        trainable_params, all_param, percentage = print_trainable_parameters(model_peft)

        print("SETTING TRAINER")
        trainer = transformers.Trainer(
            model = model_peft, 
            tokenizer=tokenizer,
            train_dataset=data['train'],
            args=transformers.TrainingArguments(
                per_device_train_batch_size=config["batch_size"], 
                gradient_accumulation_steps=2,
                evaluation_strategy='epoch',
                num_train_epochs=config["epochs"],
                warmup_steps=100, 
                learning_rate= config["lr"],
                fp16=True,
                logging_steps=1, 
                output_dir=output_dir,
                gradient_checkpointing= True,
                report_to='wandb',
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
            eval_dataset=data['test'],    
        )
        
        model_peft.config.use_cache = False  # silence the warnings. Please re-enable for inference!
        print("STARTING TRAINING:...")
        result = trainer.train()
        print_summary(result)
        print_gpu_utilization()
        del model_peft, result
        print('After Del')
        print_gpu_utilization()
        gc.collect()
        print('After gc collect')
        print_gpu_utilization()

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


    preprocess_data()
    

    model, tokenizer = load_model_tokenizer(model_id=model_id)

    print('LOADED MODEL AND TOKENIZER')
    print_gpu_utilization()
    freeze_pretrained(model)
    model.lm_head = CastOutputToFloat(model.lm_head)

    if len_data: 
        dataset_path_out = f'./data_{model_id}_hf/tokenized_and_formatted_{len_data}_rows'
    else:
        dataset_path_out = f'./data_{model_id}_hf/tokenized_and_formatted_all_rows'

    data = check_and_load(dataset_path, tokenizer, dataset_path_out, len_data=len_data, force_new=False) # dataset_path, tokenizer, dataset_path_out, len_data

    #data['train'] = data['train'].remove_columns(['input_ids_eval', 'attention_mask_eval', 'lead'])

        
    data_eval = load_prepare_data(args.input_data_path)

    if args.data_size is not None: 
        data_eval = data_eval.select(range(args.data_size))

    model, tokenizer = load_model_tokenizer(args.model_id, args.peft_path)
     

    print("Where python looks for packages: ", sysconfig.get_paths()["purelib"])

    dataset_path = '/cluster/home/terjenf/norwAI_All/NorGLMFinetune2/dataset/nrk-articles.jsonl'

    model_id = "NorLLM-AI/NorMistral-7B"
    len_data = False
    project_name = 'NorMistral_7B_finetune'
    num_sweeps = 1
    output_dir = './results/Checkpoints_NRK_Peft_NorMistral'
    peft_model_id="./results_final/Final_model_NorMistral"
    sampling_modes = ['greedy']



# sweep_config = {
# 'method': 'random',
# 'metric': {
#   'name': 'loss',
#   'goal': 'minimize'}, 
# 'parameters': {
#   'lr': {
#     'values': [9e-6]},
#   'batch_size': {
#     'values': [32]
#     },
#   'epochs': {
#     'value': 3
#     },
#   'lora_rank': {
#     'values': [16]
#     },
#   'lora_alpha':{
#     'values': [16]
#     },
#   'lora_dropout': {
#     'values': [0.05]
#     }
#     }
# }

sweep_config = {
  'lr': 1e-4,
  'batch_size': 4,
  'epochs': 2,
  'lora_rank': 16,
  'lora_alpha': 16,
  'lora_dropout': 0.05,
}


     # Initialize the accelerator
accelerator = Accelerator()
# Check if the current process is the main process
if accelerator.is_main_process:
    # Initialize wandb only for the main process
    wandb.init(
        project=project_name, 
        entity="tnf"
    )

train(model, sweep_config)