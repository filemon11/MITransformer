#!/bin/sh

N_GPUS=$(lspci|grep -i nvidia | grep -e VGA -e 3D | wc -l)
# if not usind nvidia gpus, nede to set manually
N_THREADS=$(nproc --all)
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

general_params="--n_workers ${THREADS_PER_GPU} --device ${DEVICE} --use_ddp ${USE_DDP} --first_k none --first_k_eval_test none"
general_hyperopt_params='--batch_size 128 --epochs 100 --early_stop_after 5 --n_trials=50 --eval_interval 1000 --n_warmup_steps 5 --n_startup_trials 5 --use_steps 1 --max_steps none'
hyperopt_selection='--n_embd 200:1000 --dropout 0.0:0.6 --learning_rate 1e-6:1e-2 '

core="${general_params} hyperopt ${hyperopt_selection} ${general_hyperopt_params}"

# python main.py --first_k none --first_k_eval_test none dataprep

torchrun ${prefix} \
    --name exp1_dep-supervised \
    ${core} \
    --dependency_mode 'supervised' \
    --loss_alpha 0.0:1.0

torchrun ${prefix} \
    --name exp1_dep0 \
    ${core} \
    --dependency_mode 'standard'

torchrun ${prefix} \
    --name exp1_dep-in \
    ${core} \
    --dependency_mode input 
