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
general_hyperopt_params='--batch_size 128 --epochs 100 --early_stop_after 3 --n_trials=50 --eval_interval 500 --n_warmup_steps 3 --n_startup_trials 3 --use_steps 1 --max_steps none --n_embd 812 --dropout 0.065 --learning_rate 1.1e-3'
hyperopt_selection='--loss_alpha 0.0:1.0'

core="${general_params} hyperopt ${hyperopt_selection} ${general_hyperopt_params}"

# python main.py --first_k none --first_k_eval_test none dataprep

torchrun ${prefix} \
    --name exp1_dep-supervised \
    ${core} \
    --dependency_mode 'supervised_det'
