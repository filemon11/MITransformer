#!/bin/sh

ALPHA=$1
ALPHA=$((${ALPHA}00/10))
ALPHA=${ALPHA:0:-2}.${ALPHA: -2}

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

general_params="--n_workers ${THREADS_PER_GPU} --device ${DEVICE} --use_ddp ${USE_DDP}"
hyperparams='--first_k 2 --first_k_eval_test 2 --dataset_name Wikitext_processed --masks_setting current --batch_size 2 --epochs 10 --early_stop_after none --eval_interval 1 --use_steps 1 --max_steps none --n_embd 812 --dropout 0.065 --learning_rate 1.1e-3'

core="${general_params} train ${hyperparams}"

# python main.py --first_k none --first_k_eval_test none dataprep
# torchrun ${prefix} \
#     --name test_${ALPHA} \
#     ${core} \
#     --dependency_mode 'supervised' \
#     --use_lstm 1 \
#     --depth 1 \
#     --loss_alpha ${ALPHA}

# python main.py --name test_1.0 --n_workers 12 --device cpu --use_ddp 0 train --first_k 10 --first_k_eval_test 10 --dataset_name Wikitext_processed --masks_setting current --batch_size 5 --epochs 10 --gradient_acc 2 --early_stop_after none --eval_interval 1 --use_steps 1 --max_steps none --n_embd 812 --dropout 0.065 --learning_rate 1.1e-3 --dependency_mode 'supervised' --loss_alpha 1.0
python main.py --name test_1.0_0 --n_workers 12 --device cpu --use_ddp 0 test --first_k 10 --first_k_eval_test 10 --dataset_name Wikitext_processed --batch_size 5