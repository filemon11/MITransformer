# python main.py --first_k none --first_k_eval_test none dataprep
NUM=$1

for i in $(LC_ALL=C seq 0.2 .1 0.2); do
  for j in $(LC_ALL=C seq 0 1 $((NUM -1))); do
    echo "Testing alpha=${i}, model=${j} ..."
    python -m mitransformer --use_ddp 0 --device "cuda" --name test_exp1_repeat_${i}_${j} test --dataset_name Wikitext_processed --first_k 4 --model_name exp1_repeat_${i}_${j} --batch_size 4 --att_plot 0 --tree_plot 0 --masks_setting current
  done
done

# for i in 0.105 0.110 0.165 0.170 0.175 0.180 0.185 0.190 0.195; do  #$(LC_ALL=C seq 0.105 .005 0.195); do
#   for j in $(LC_ALL=C seq 0 1 2); do
#     echo "Testing alpha=${i}, model=${j} ..."
#     python main.py --use_ddp 0 --device "cuda" --name test_exp1_dep-alphaselect_${i}_${j} test --first_k 8 --model_name exp1_dep-alphaselect_${i}_${j} --batch_size 8 --att_plot 0 --tree_plot 0
#   done
# done