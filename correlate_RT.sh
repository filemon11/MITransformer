model=$1 # "exp1_dep-supervised_det_39"

python reading_times.py $model

cd RT
Rscript --vanilla preproc.R data/words_processed_${model}.csv data/preprocessed_${model}.csv
Rscript --vanilla analysis.R data/preprocessed_${model}.csv results/plot_${model}.png > results/log_${model}.log