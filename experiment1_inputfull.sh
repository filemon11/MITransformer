
#!/bin/sh

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

general_params="--seed 1895 --n_workers ${THREADS_PER_GPU} --device ${DEVICE} --use_ddp ${USE_DDP} --first_k 16 --first_k_eval_test 16 --dataset_name Wikitext_processed"
hyperparams='--batch_size 8 --epochs 10 --early_stop_after none --eval_interval 1 --use_steps 1 --max_steps none --n_embd 812 --dropout 0.065 --learning_rate 1.1e-3 --n_runs 3'

core="${general_params} train ${hyperparams}"

# python main.py --first_k none --first_k_eval_test none dataprep

torchrun ${prefix} \
    --name exp1_dep-inputfull2 \
    ${core} \
    --dependency_mode 'input'
