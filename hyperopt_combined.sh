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

general_params="--n_workers ${THREADS_PER_GPU} --device ${DEVICE} --use_ddp ${USE_DDP}"
general_hyperopt_params='--optimise loglik --masked 0 --first_k 10000 --first_k_eval_test 10000 --batch_size 100 --gradient_acc 2 --epochs 1 --early_stop_after 4 --n_trials=10 --eval_interval 2 --n_warmup_steps 2 --n_startup_trials 3 --use_steps 1 --max_steps none'
hyperopt_selection='--layer_design (0,) --width 4 --depth 4 --n_embd 400;800 --dropout_attn 0.0 --dropout_resid 0.0;0.6 --dropout_ff 0.0;0.6 --dropout_embd 0.0;0.6 --dropout_lstm 0.0;0.6 --learning_rate 1e-4;1e-2 --d_ff_factor 4;10 --bias 0 --use_dual_fixed 0'

core="${general_params} hyperopt ${hyperopt_selection} ${general_hyperopt_params}"

torchrun ${prefix} \
    --name hyperopt_combined \
    ${core} \
    --dependency_mode 'standard'

