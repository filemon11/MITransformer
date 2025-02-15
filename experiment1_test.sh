#!/bin/sh

N_GPUS=1 #$(lspci|grep -i nvidia | grep -e VGA -e 3D | wc -l)
# if not usind nvidia gpus, nede to set manually
N_THREADS=16 #$(nproc --all)
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


# python main.py --first_k none --first_k_eval_test none dataprep
for i in $(LC_ALL=C seq 0.0 .1 1.0); do
  for j in $(LC_ALL=C seq 0 1 2); do
    python main.py --use_ddp 1 --device "gpu" --first_k 32 test --model_name exp1_dep-alphaselect_${i}_${j} --batch_size 32 --att_plot 1 --tree_plot 1
done