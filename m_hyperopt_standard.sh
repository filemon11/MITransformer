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
general_hyperopt_params='--psyling_eval 1 --average_psyling 1 --psyling_dataset geco_train,zuco2_1_train,zuco1_1_train,zuco1_2_train,meco1_train,meco2_train,frank_ET_train --shift 1 --lme_formula GPT~length+position+length.1+frequency+frequency.1+surprisal+surprisal.1+(position+length+length.1+frequency+frequency.1+surprisal+surprisal.1|WorkerId) --min_len_train 3 --min_len_eval_test 3 --max_len_train 60 --max_len_eval_test 60 --optimise perplexity --k_negatives None|(49;499) --masked 0 --combined False --first_k None --first_k_eval_test None --batch_size 496 --gradient_acc 2 --epochs 10 --early_stop_after 4 --early_stop_metric perplexity --n_trials=100 --eval_interval 1000 --n_warmup_steps 4 --use_steps 1 --max_steps none'
hyperopt_selection='--pos_enc sinusoidal|embedding --use_lstm 0 --sampler tpe --pruner hyperband --length_weighted 1 --layer_design (0,) --width 16 --depth 24 --n_embd 1024 --dropout_attn 0.0 --dropout_resid 0.0;0.5 --dropout_ff 0.0;0.5 --dropout_embd 0.0;0.5 --dropout_lstm 0.0;0.5 --learning_rate 1e-4;1e-3 --d_ff_factor 4;8 --bias 0|1 --use_dual_fixed 0'

core="${general_params} hyperopt ${hyperopt_selection} ${general_hyperopt_params}"

# geco_train,zuco2_1_train,zuco1_1_train,zuco1_2_train,meco1_train,meco2_train,frank_ET_train
# naturalstories_train,frank_SP_train

# TODO: allow sentences up to length 60
# python -m mitransformer dataprep --min_len_train 3 --min_len_eval_test 3 --max_len_train 40 --max_len_eval_test 40 --first_k none --first_k_eval_test none --masked 0

torchrun ${prefix} \
    --name standard_test2 \
    ${core} \
    --dependency_mode 'standard'

# --load_psyling_mmap temp_mmap_dataset
