#!/bin/sh

N_GPUS=1 #$(lspci|grep -i nvidia | grep -e VGA -e 3D | wc -l)
# if not usind nvidia gpus, nede to set manually
N_THREADS=12 #$(nproc --all)
THREADS_PER_GPU=$((N_THREADS / N_GPUS))
export NUMEXPR_MAX_THREADS=$THREADS_PER_GPU
export OMP_NUM_THREADS=$THREADS_PER_GPU

if [ $N_GPUS -gt 1 ]
then
  USE_DDP=1
else
  USE_DDP=0
fi

if [ $N_GPUS = 0 ]
then
  DEVICE=cpu
  NGPUS=1
  # uses as number of devices
else
  DEVICE=cuda
fi

prefix="--standalone --nnodes=1 --nproc-per-node=${N_GPUS} main.py"

general_params="--seed 1895 --n_workers ${THREADS_PER_GPU} --device ${DEVICE} --use_ddp ${USE_DDP} --first_k none --first_k_eval_test none --dataset_name Wikitext_processed"
hyperparams='--early_stop_metric lm_metric --discriminative true --loss_alpha 0.12 --batch_size 8 --epochs 2 --early_stop_after none --eval_interval 16000 --use_steps 1 --max_steps none --n_embd 812 --dropout 0.065 --learning_rate 1.1e-3'

core="${general_params} train ${hyperparams}"

# python main.py --first_k none --first_k_eval_test none dataprep
torchrun ${prefix} \
    --name exp_discriminative_baseline \
    ${core} \
    --dependency_mode 'standard'

torchrun ${prefix} \
    --name exp_discriminative_supervised \
    ${core} \
    --dependency_mode 'supervised'
