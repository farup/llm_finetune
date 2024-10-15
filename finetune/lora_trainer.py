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

from accelerate import Accelerator
import torch
import torch.nn as nn
import evaluate
import wandb
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, EvalPrediction
from peft import LoraConfig, get_peft_model 

sys.path.append("/cluster/home/terjenf/norwAI_All/llm_training")

from util.nrk_data.train_preprocess_data import format_tokenize_data, split_data, tokenize_format_eval, find_train_eval_split, load_train_eval_split, save_train_eval_split

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

def load_model_tokenizer(model_id, torch_dtype):

    torch_dtype = eval(torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch_dtype)
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
    r=config['rank'], #attention heads, rank of the attention matrix, 
    lora_alpha= config['lora_alpha'], #alpha scaling, scaling factor for the weight matrices
    # target_modules=["q_proj", "v_proj"],
    lora_dropout=config['lora_dropout'],
    bias="none",
    task_type="CAUSAL_LM" # set this for CLM or Seq2Seq
    )
    model = get_peft_model(model, lora_config)
    return model

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

    run_name = f"{exp_name}_{int(time.time())}"

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name, 
            entity=args.wandb_entity,  
            config=vars(args),
            name=run_name, 
            save_code=True, 
        )

    model, tokenizer = load_model_tokenizer(model_id=config.get("model_id"), torch_dtype=config.get("torch_dtype"))
    data = format_tokenize_data(tokenizer,config.get("model_id"), config.get("data"))

    if not find_train_eval_split(config.get("model_id"), config.get("data")): 
        train_data, eval_data = split_data(data, config.get("model_id"), config.get("data"))
        eval_data = tokenize_format_eval(eval_data, tokenizer)
        save_train_eval_split(train_data, eval_data, config.get("model_id"), config.get("data"))

    else:
        train_data, eval_data = load_train_eval_split(config.get("model_id"), config.get("data"))

    if config['data'].get("dataset_size") is not None:
        data_size = config['data'].get("dataset_size")
        if data_size != "full":
            train_data = train_data.select(range(int(int(data_size)*0.8)))
            eval_data = eval_data.select(range(int(int(data_size)*0.2)))

    if isinstance(config['data'].get("dataset_size"), int): 
        data = data.select(range(config['data'].get("dataset_size")))
        
    train_data, eval_data = split_data(data, config.get("model_id"), config.get("data"))
    eval_data = tokenize_format_eval(eval_data, tokenizer)

    freeze_pretrained(model)
    model.lm_head = CastOutputToFloat(model.lm_head)

    checkpoint_output_dir = os.path.join(config.get("output_dir"), "checkpoints", run_name)
    peft_model_output_dir = os.path.join(config.get("output_dir"), "final", run_name)

    peft_model = setup_peft_model(model, config['lora_parameters'])
    peft_model.print_trainable_parameters()

    train_parms,total_parms = peft_model.get_nb_trainable_parameters()

    if args.track: 
        wandb.log({"trainable params":train_parms, "all params": total_parms, "trainable%": round(int(train_parms)/int(total_parms), 4)})

    config_parm = config['parameters']
    
    trainer = transformers.Trainer(
            model = peft_model, 
            tokenizer=tokenizer,
            train_dataset=train_data,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=config_parm["batch_size"], 
                gradient_accumulation_steps=config_parm['gradient_accumulation_steps'],
                evaluation_strategy='steps',
                eval_steps=config_parm['eval_steps'],
                num_train_epochs=config_parm["epochs"],
                warmup_steps=config_parm['warmup_steps'], 
                learning_rate= config_parm["lr"],
                bf16=config_parm["bf16"],
                fp16=config_parm["fp16"],
                logging_steps=config_parm['logging_steps'],
                output_dir=checkpoint_output_dir,
                save_total_limit=5,
                save_steps=0.01,
                gradient_checkpointing=config_parm["gradient_checkpointing"],
                report_to='wandb' if args.track else 'none',
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
            eval_dataset=eval_data
        )

    peft_model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    print("STARTING TRAINING:...")
    trainer.train()
    trainer.model.save_pretrained(peft_model_output_dir)
    print("Done!")