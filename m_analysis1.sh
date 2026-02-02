#!/bin/sh

N_GPUS=1 #$(lspci|grep -i nvidia | grep -e VGA -e 3D | wc -l)
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
export HSA_OVERRIDE_GFX_VERSION="11.0.2"


prefix="--standalone --nnodes=1 --nproc-per-node=${N_GPUS} -m mitransformer.__main__"

general_params="--n_workers ${THREADS_PER_GPU} --device ${DEVICE} --use_ddp False"

core="${general_params} rt --lme False --model_name exp1_4 --batch_size 2 --to_add surprisal,attention_entropy --dataset_name provo_train"

# geco_train,zuco2_1_train,zuco1_1_train,zuco1_2_train,meco1_train,meco2_train,frank_ET_train
# naturalstories_train,frank_SP_train

# TODO: allow sentences up to length 60
# python -m mitransformer dataprep --min_len_train 3 --min_len_eval_test 3 --max_len_train 40 --max_len_eval_test 40 --first_k none --first_k_eval_test none --masked 0

torchrun ${prefix} \
    --name RT_test \
    ${core}

# --load_psyling_mmap temp_mmap_dataset
