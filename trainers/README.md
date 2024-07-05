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


