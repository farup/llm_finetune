from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, Trainer
from peft import PeftModel, PeftConfig
from huggingface_hub import login
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


def load_model_tokenizer(model_id, torch_dtype):

    torch_dtype = eval(torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)
    tokenizer.pad_token = tokenizer.eos_token

    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def merge_model(model, pathPeft): 

    new_model = PeftModel.from_pretrained(model, pathPeft).to(device)

    merged_model = new_model.merge_and_unload()
    print("MERGED")
    return merged_model



peft_path = "/cluster/home/terjenf/norwAI_All/results/final/mistral_7B_exp_1_1722629952"
model_id = "NorwAI/NorwAI-Mistral-7B"

model, tokenizer = load_model_tokenizer(model_id, "torch.float32")
merged_path = "/cluster/home/terjenf/norwAI_All/llm_training/merged_models/merged_mistral_7B_exp_1_1722629952"
tokenizer.save_pretrained(merged_path)
print("hei")