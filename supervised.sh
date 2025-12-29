#!/bin/sh

N_GPUS=$(lspci|grep -i nvidia | grep -e VGA -e 3D | wc -l)
# if not using nvidia gpus, need to set manually

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
    # used as number of devices
else
    DEVICE=cuda
fi

N_THREADS=$(nproc --all)
THREADS_PER_GPU=$((N_THREADS / N_GPUS))
export NUMEXPR_MAX_THREADS=$THREADS_PER_GPU
export OMP_NUM_THREADS=$THREADS_PER_GPU


prefix="--standalone --nnodes=1 --nproc-per-node=${N_GPUS} -m mitransformer.__main__"

general_params="--n_workers ${THREADS_PER_GPU} --device ${DEVICE} --use_ddp ${USE_DDP} --use_amp 0"
general_hyperopt_params='--batch_size 256 --epochs 10 --early_stop_after none --eval_interval 2000 --use_steps 1 --max_steps none --masks_setting current --use_dual_fixed 0'
hyperopt_selection='--n_embd 464 --dropout_attn 0.0 --dropout_resid 0.597 --dropout_ff 0.454 --dropout_embd 0.133 --dropout_lstm 0.211 --learning_rate 1.13e-4 --d_ff_factor 4 --bias 0 --loss_alpha 0.05'

core="${general_params} train ${hyperopt_selection} ${general_hyperopt_params}"

torchrun ${prefix} \
    --name supervised \
    ${core} \
    --dependency_mode supervised