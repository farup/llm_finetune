## Accelerate and Trainer


Trainer from huggingface uses accelerate under the hood. 


Train with config (customize) 

```
accelerate config [arguments]
```

optional args example (path to config file):
```
accelerate config --config_file ./accelerate_config.yaml
```

Launch: 

```
accelerate launch [arguments] {training_script} --{training_script-argument-1
```


mistral 7B runs out of memory on 2 V100 32GB cards, with device_map="auto", batch_size=4, lora_alpha=16, lora_rank=16

Cant train model in torch.float16
https://github.com/huggingface/transformers/issues/23165


Bfloat16 not supported on V100? 
https://github.com/salesforce/LAVIS/issues/91


## Training options and quantization 

- Facing problems loading model in in BFloat16 and using mixed precision. Might be solved be setting up mixed precision with accelerate config. 
- Working with loading model in fp32 and using mixed precision "fp16=True"



## Parameter efficient fine-tuning (PEFT)

A method that aims to reduce the size of models making it possible to perform calculations on less powerful GPUs

LoRA (low-rank adaption) is a method in PEFT. Adds a small number of new weights to the model for training, rather than retraining the entire parameter space of the model. If a weight matrix conists of `m x n` weigths, then two lora adapters often reffered to as lora_A and lora_B reduce the amount of parameters with a lower rank `r`, `m x n = m x r + r x n`

Example: 
If m and n are 1000, results in 1 000 000 parameters in orignal, but with a rank of r = 16 this is results in 32 000 trainable parameters. 

```
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

state_dict = get_fp32_state_dict_from_zero_checkpoint(pathPeftModel)
```



### Problems: 

"_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'



Belive error while saving tokenizer in torch_dtype=torch.float32
```
  File "/cluster/home/terjenf/.conda/envs/vgdebatt/lib/python3.11/site-packages/transformers/trainer.py", line 3437, in _save
    self.tokenizer.save_pretrained(output_dir)
  File "/cluster/home/terjenf/.conda/envs/vgdebatt/lib/python3.11/site-packages/transformers/tokenization_utils_base.py", line 2511, in save_pretrained
    out_str = json.dumps(tokenizer_config, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
        ^^^^^^^^^^^
  File "/cluster/home/terjenf/.conda/envs/vgdebatt/lib/python3.11/json/encoder.py", line 180, in default
    raise TypeError(f'Object of type {o.__class__.__name__} '

```
