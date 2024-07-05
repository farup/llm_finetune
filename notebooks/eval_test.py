import torch
import deepspeed

from accelerate import Accelerator
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from torch.utils.data import TensorDataset, DataLoader, Dataset
from datasets import load_dataset, load_from_disk
from tqdm.auto import tqdm
import deepspeed


pathPeftModel = "/cluster/home/terjenf/norwAI_All/results/Checkpoints_NRK_Peft_NorMistral/checkpoint-28000"
model_id = "NorLLM-AI/NorMistral-7B"

tokenizer = AutoTokenizer.from_pretrained(model_id,device_map='auto',torch_dtype=torch.bfloat16)                                         
model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.bfloat16)
tokenizer.pad_token = tokenizer.eos_token
model = PeftModel.from_pretrained(model, pathPeftModel)
model = model.merge_and_unload()


model_engine = deepspeed.initialize(model=model)
tup = model_engine.load_checkpoint(load_dir=pathPeftModel)

tokens = tokenizer("hei, jeg heter terje", max_length=20, return_tensors='pt')

print(model.hf_device_map)
#out_f = model(input_ids=tokens['input_ids'].to("cuda:0"), attention_mask=tokens['attention_mask'].to("cuda:0"), labels=tokens['input_ids'].to("cuda:1"))

out = model.generate(input_ids=tokens['input_ids'].to("cuda:0"), attention_mask=tokens['attention_mask'].to("cuda:0"))

# Initialize the DeepSpeed-Inference engine
ds_engine = deepspeed.init_inference(model,
                                 tensor_parallel={"tp_size": 1},
                                 dtype=torch.half,

                                 replace_with_kernel_inject=True)

print(type(ds_engine))

ds_engine.load_checkpoint(deepspeed_path)

out_ds = ds_engine.module.generate(input_ids=tokens['input_ids'].to("cuda:0"), attention_mask=tokens['attention_mask'].to("cuda:0"))


print("hei")