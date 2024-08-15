#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=400:05:00
#SBATCH --nodes=1              # 1 compute node
#SBATCH --ntasks-per-node=4    # 4 mpi process each node
#SBATCH --gres=gpu:2
#SBATCH --constraint="gpu80g"
#SBATCH --mem=70G
#SBATCH --job-name="Train NorMistral7B"
#SBATCH --output=/cluster/home/terjenf/norwAI_All/llm_training/finetune/jobs/slurm_outputs/trainer_mistral7B.out
#SBATCH --mail-user=terjenf@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}

echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of the job is: $SLURM_JOB_NAME"

module purge
module load  Anaconda3/2023.09-0

conda activate vgdebatt

python /cluster/home/terjenf/norwAI_All/llm_training/finetune/lora_trainer.py --wandb-project-name normistral_no_custom_7B --exp-config-path /cluster/home/terjenf/norwAI_All/llm_training/configs/experiments/mistral_7B_exp_1.yaml --track True