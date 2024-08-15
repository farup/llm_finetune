# Evaluation 


### How to run file:


Example script
```
python evaluate_mulit_gpu_auto.py --model-id <model id> --peft-path <path to peft adapters> --input-data-path <path to processed data> --data-size <size or full> --track <True: wandb>

```





Checkpoints saved with DeepSpeed Stage 3 seems not to be suported to be converted to universial(?)
https://github.com/microsoft/DeepSpeed/issues/5405

Model wrapped with deepspeed not able to to use model.generate()?


`
code
`