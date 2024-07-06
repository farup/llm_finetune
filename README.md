# Finetuning and Parallelism 

Training large lanugage models are not necessarily straight forward, however there exists tools and frameworks to simplify this, enabling training of models such as Megatron-2 with one trillion parameters, with limiting computing capacity.

This file gives an introduction to different parallelism techniques. 

## Data Parallelism

Copy model weights across multiple devices, each being fed a slice of the data. Forwardpass and gradient caluclations done in parallel. Devices synchronized by "reducing" gradients (averaged). Updatas weights on onde deivce (?) then sent back to each device. **Requires the model to fit on one GPU.** 

DistributedDataParallel (DDP) is typically faster than DataParallel (DP).


## Model Parallel

Model is split and distrbuted over devices. Generally two types tensor parallelism and pipeline parallelism. 

Tensor parallelism is to parallelize computation within an operation such as matrix-matrix multiplication. Pipeline parallelism is to parallelize computation between layers.



### Tensor Parallelism (intra layer parallelism)
---

Split a tensor into ```N``` chunks, where each device holds ```1/N``` of the whole tensor without affecting correctness of computation graph. 

<img src="https://s2.loli.net/2022/01/28/2ZwyPDvXANW4tMG.png" alt="tp" width="300"/>

Figure shows tensor parallelism by the coloumns, ```B``` split between different devices. 

### Pipeline Parallelism (inter layer parallelism)
---

Model split by layer into several chunks, each chunk is given to a device. Intermediate activations passed to the next (forward pass) or previous (backward pass) layer. 

<img src="https://s2.loli.net/2022/01/28/at3eDv7kKBusxbd.png" alt="pp" width="300"/>

One draw back of pipline parallel training is waste of compitational resources as devices needs to wait on previous. 

### ZeRO - DeepSpeed 
---
ZERO (zero redundancy optimizer) by DeepSpeed a technique created by Microsoft. 

Sharding (partioning) a model’s parameters, gradients and optimizer states across data parallel workers (gpus/tpus). 

ZERO has 3 stages:

1. Optimizer states are partitioned across processes.
2. Gradients are partitioned across processes.
3. Model parameters are partitioned across the processes.

Users can decide between which one to use. 

ZERO makes it also possible to offload the sharded model parameters to CPUs or disk, as has much larger memory compared to GPU. Tensors offlad to CPU memory or disk when they are not used. Enables the possiblity to accomodate huge models on a single machine. 

https://arxiv.org/abs/1910.02054

### FSDP 
---

Fully-sharded data-parallel (FSDP) is Meta’s version of sharding, inspired by DeepSpeed (stage 3).


links:

https://huggingface.co/docs/transformers/v4.24.0/en/perf_train_gpu_many

https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/

https://colossalai.org/docs/concepts/paradigms_of_parallelism/

