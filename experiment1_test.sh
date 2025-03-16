# python main.py --first_k none --first_k_eval_test none dataprep
for i in $(LC_ALL=C seq 0.1 .1 1.0); do
  for j in $(LC_ALL=C seq 0 1 2); do
    echo "Testing alpha=${i}, model=${j} ..."
    python main.py --use_ddp 0 --device "cpu" --name test_exp_discriminative_baseline_0 test --dataset_name Wikitext_processed --first_k 8 --model_name exp_discriminative_baseline_0 --batch_size 8 --att_plot 0 --tree_plot 0
  done
done

# for i in 0.105 0.110 0.165 0.170 0.175 0.180 0.185 0.190 0.195; do  #$(LC_ALL=C seq 0.105 .005 0.195); do
#   for j in $(LC_ALL=C seq 0 1 2); do
#     echo "Testing alpha=${i}, model=${j} ..."
#     python main.py --use_ddp 0 --device "cuda" --name test_exp1_dep-alphaselect_${i}_${j} test --first_k 8 --model_name exp1_dep-alphaselect_${i}_${j} --batch_size 8 --att_plot 0 --tree_plot 0
#   done
# done