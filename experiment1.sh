#!/bin/sh

N_GPUS=8 # $(lspci|grep -i nvidia | grep -e VGA -e 3D | wc -l)
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
  N_GPUS=1
  # uses as number of devices
else
  DEVICE=cuda
fi

prefix="--standalone --nnodes=1 --nproc-per-node=${N_GPUS} main.py"

general_params="--n_workers ${THREADS_PER_GPU} --device ${DEVICE} --use_ddp ${USE_DDP}"
general_hyperopt_params='--batch_size 256 --epochs 5 --early_stop_after 3 --n_trials=100 --eval_interval 500 --n_warmup_steps 3 --n_startup_trials 1 --use_steps 1 --max_steps none'
hyperopt_selection='--n_embd 200:1000 --dropout 0.0:0.4 --learning_rate 1e-6:1e-2 --pos_enc embedding,sinusoidal --bias 1,0 --use_dual_fixed 1,0'

core="${general_params} hyperopt ${hyperopt_selection} ${general_hyperopt_params}"

# python main.py --first_k none --first_k_eval_test none dataprep

torchrun ${prefix} \
    --name test_hyperopt \
    ${core} \
    --dependency_mode 'supervised' \
    --loss_alpha 1.0