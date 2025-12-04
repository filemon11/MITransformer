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
general_hyperopt_params='--masked 0 --use_lstm False --first_k_eval_test 1000 --combined_loss True --distr_mode att-n --global_distr False --length_weighted True --include_current False --batch_size 10 --epochs 1000  --early_stop_after none --eval_interval 100 --use_steps 1 --max_steps none --masks_setting current --use_dual_fixed 0'
hyperopt_selection="--depth 4 --width 2 --losses {lm:0.9,attention_entropy:0.1} --n_embd 400 --dropout_attn 0.0 --dropout_resid 0.219 --dropout_ff 0.026 --dropout_embd 0.083 --dropout_lstm 0.305 --learning_rate 1.21e-3 --d_ff_factor 7 --bias 0"

# --layer_design (h1,h2)|(h3,h4) --losses {lm:0.33;1.0,attention_entropy:0.33|0.44,distance:0.33}|{lm:1.0}

core="${general_params} train ${hyperopt_selection} ${general_hyperopt_params}"

torchrun ${prefix} \
    --name standard \
    ${core} \
    --dependency_mode standard