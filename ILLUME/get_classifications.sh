#! /bin/bash
#SBATCH -p h200
#SBATCH --gres gpu:h200:1
#SBATCH --time 06:00:00

export TORCH_LOGS="-dynamo,-inductor,-aot"
# export WANDB_MODE="offline"
# export HF_HUB_OFFLINE=1

source /export/home/ra48gaq/anaconda3/etc/profile.d/conda.sh
conda activate illume
export LD_LIBRARY_PATH="/export/home/ra63ral/miniconda3/envs/pytorch2.4/lib/python3.1/site-packages/nvidia/cuda_nvrtc/lib/:$LD_LIBRARY_PATH"

export MASTER_PORT=$((25000+RANDOM%5000))
export HYDRA_FULL_ERROR=1
# srun deepspeed --master_port $MASTER_PORT train.py deepspeed.local_world_size=2 experiment=gen_cap_vislex wandb.enabled=True
export MASTER_IP='127.0.0.1' # fine for single node training on MVL

# export LAUNCH='export TRITON_CACHE_DIR=$(mktemp -d); export TORCH_EXTENSIONS_DIR=$(mktemp -d); accelerate launch --debug --num_processes $NUM_PROCS --num_machines $COUNT_NODE --multi_gpu --mixed_precision no --machine_rank $SLURM_PROCID --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT train.py deepspeed.local_world_size=4'


# srun apptainer exec $CONTAINER bash -c 'export SSL_CERT_FILE=/home/hpc/v104dd/v104dd24/code/diffusion/cacert.pem; export WANDB_NAME="$SLURM_JOBID-gen_cap_gumble_sd21_frozen_img_enc_no_temp_decay_helma"; pip install --no-cache-dir --upgrade pip wandb==0.18.3; pip install sentencepiece; eval "$LAUNCH experiment=gen_cap_gumble_sd21_frozen_img_enc_helma"'


# srun accelerate launch --debug --num_processes 4 --num_machines 1 --multi_gpu --mixed_precision no --machine_rank $SLURM_PROCID --main_process_ip $MASTER_IP --main_process_port $MASTER_PORT compute_class_predictions.py --model_type illume --bs 20 --data_path /export/group/datasets/imagenet_shards_raw_shuffled/ --no_classification --out_path /export/home/ra48gaq/code/ILLUME_plus/outputs

srun accelerate launch --debug --num_processes 1 --num_machines 1 --mixed_precision no --machine_rank $SLURM_PROCID --main_process_ip $MASTER_IP --main_process_port $MASTER_PORT compute_class_predictions.py --model_type illume --bs 100 --data_path /export/group/datasets/imagenet_shards_raw_shuffled/ --pregenerated_imgs --out_path /export/home/ra48gaq/code/ILLUME_plus/outputs