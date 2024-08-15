import os
import sysconfig
import sys
import logging
import argparse
import time
import yaml

from pynvml import *
from dotenv import load_dotenv
from distutils.util import strtobool
from functools import partial

from accelerate import Accelerator
import torch
import torch.nn as nn
import evaluate
import wandb
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer
from peft import LoraConfig, get_peft_model 
import wandb

sys.path.append("/cluster/home/terjenf/norwAI_All/llm_training")
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(torch.cuda.device_count()))

from util.nrk_data.train_preprocess_data import format_tokenize_data, split_data, tokenize_format_eval
#from llm_training.util.nrk_data.train_preprocess_data import format_tokenize_data, split_data, tokenize_format_eval

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

def parse_args(): 
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--exp-config-path', type=str, default="",
                    help='path to config file for hyperparameter')
    
    parser.add_argument('--sweep-yaml-path', type=str, default="",
                    help='path to config file for hyperparameter')
    
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--num-sweep', type=int, default=6,
                        help='number of runs in the sweep')
    
    parser.add_argument('--wandb-project-name', type=str, default="sweep_norwai",
                        help="the wandb's roject name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")
    
    args = parser.parse_args()
    return args

def load_model_tokenizer(model_id):
    if "llama" in model_id.lower(): 
        tokenizer = LlamaTokenizer.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float32)
        model = LlamaForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float32)

    else: 
        tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float32)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float32)

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

def setup_peft_model(model, config):
    lora_config = LoraConfig(
    r=config['rank'], #attention heads, rank of the attention matrix, i think
    lora_alpha= config['lora_alpha'], #alpha scaling, scaling factor for the weight matrices
    # target_modules=["q_proj", "v_proj"], #will be set after i know the names
    lora_dropout=config['lora_dropout'],
    bias="none",
    task_type="CAUSAL_LM" # set this for CLM or Seq2Seq
    )
    model = get_peft_model(model, lora_config)
    return model


def train(model, tokenizer,train_data, eval_data, run_name, checkpoint_output_dir, peft_model_output_dir):
    with wandb.init(project=run_name):

        config = wandb.config

        peft_model = setup_peft_model(model, config)
 
        train_parms,total_parms = peft_model.get_nb_trainable_parameters()

        wandb.log({"train data-size": len(train_data)})
        wandb.log({"eval data-size": len(eval_data)})
        wandb.log({"trainable params":train_parms, "all params": total_parms, "trainable%": round(int(train_parms)/int(total_parms), 4)})
    
        trainer = transformers.Trainer(
                model = peft_model, 
                tokenizer=tokenizer,
                train_dataset=train_data,
                args=transformers.TrainingArguments(
                    per_device_train_batch_size=config["batch_size"], 
                    gradient_accumulation_steps=config['gradient_accumulation_steps'],
                    save_strategy="no",
                    eval_strategy='steps',
                    eval_steps=config['eval_steps'],
                    num_train_epochs=config["epochs"],
                    warmup_steps=config['warmup_steps'], 
                    learning_rate= config["lr"],
                    fp16=config["fp16"],
                    logging_steps=config['logging_steps'],
                    output_dir=checkpoint_output_dir,
                    save_total_limit=5,
                    save_steps=0.1,
                    gradient_checkpointing=config["gradient_checkpointing"],
                    report_to='wandb',
                ),
                data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
                eval_dataset=eval_data
            )

        peft_model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
        print("STARTING TRAINING:...")
        trainer.train()
        print("Done!")

if __name__ == "__main__":

    print("Where python looks for packages: ", sysconfig.get_paths()["purelib"])
    args = parse_args()

    torch.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = args.torch_deterministic
    try:
        with open(args.exp_config_path, 'r') as file:
            config = yaml.safe_load(file)
            exp_name = file.name.split("/")[-1].split(".")[0]
            
    except FileNotFoundError as e: 
        print("Error while loading yaml config",  e)

    try:
        with open(args.sweep_yaml_path, 'r') as file:
            sweep_config = yaml.safe_load(file)     
    except FileNotFoundError as e: 
        print("Error while loading yaml config",  e)

    run_name = f"{exp_name}_{int(time.time())}"

    model, tokenizer = load_model_tokenizer(model_id=config.get("model_id"))
    data = format_tokenize_data(tokenizer,config.get("model_id"), config.get("data"))

    if config['data'].get("dataset_size") is not None:
        size = config['data'].get("dataset_size") 
        if isinstance(int(size), int):
            data = data.select(range(int(size)))
    #data = data.select(range(5000))
 
    train_data, eval_data = split_data(data, config.get("model_id"), config.get("data"))
    eval_data = tokenize_format_eval(eval_data, tokenizer)

    
    freeze_pretrained(model)
    model.lm_head = CastOutputToFloat(model.lm_head)

    checkpoint_output_dir = os.path.join(config.get("output_dir"), "checkpoints", run_name)
    peft_model_output_dir = os.path.join(config.get("output_dir"), "final", run_name)

    sweep_id = wandb.sweep(sweep=sweep_config, project=run_name) #  wind up a Sweep Controller by calling
    wandb_train = partial(train, model, tokenizer,train_data, eval_data, run_name, checkpoint_output_dir, peft_model_output_dir)
    wandb.agent(sweep_id=sweep_id, function=wandb_train, count=args.num_sweep)


