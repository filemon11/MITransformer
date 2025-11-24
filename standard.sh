#!/bin/sh

N_GPUS=$(lspci|grep -i nvidia | grep -e VGA -e 3D | wc -l)
# if not using nvidia gpus, need to set manually

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
    # used as number of devices
else
    DEVICE=cuda
fi

prefix="--standalone --nnodes=1 --nproc-per-node=${N_GPUS} -m mitransformer.__main__"

general_params="--n_workers ${THREADS_PER_GPU} --device ${DEVICE} --use_ddp ${USE_DDP}"
general_hyperopt_params='--batch_size 256 --epochs 10  --early_stop_after none --eval_interval 2000 --use_steps 1 --max_steps none --masks_setting current --use_dual_fixed 0'
hyperopt_selection='--n_embd 886 --dropout_attn 0.0 --dropout_resid 0.219 --dropout_ff 0.026 --dropout_embd 0.083 --dropout_lstm 0.305 --learning_rate 1.21e-3 --d_ff_factor 7 --bias 0'

core="${general_params} train ${hyperopt_selection} ${general_hyperopt_params}"

torchrun ${prefix} \
    --name standard \
    ${core} \
    --dependency_mode standard