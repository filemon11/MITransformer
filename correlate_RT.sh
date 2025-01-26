model=$1 # "exp1_dep-supervised_det_39"
baseline=$2

# python reading_times.py $model
# python reading_times.py $baseline
# 
cd RT
# Rscript --vanilla preproc.R data/words_processed_${model}.csv data/preprocessed_${model}.csv
# Rscript --vanilla preproc.R data/words_processed_${baseline}.csv data/preprocessed_${baseline}.csv
Rscript --vanilla analysis.R data/preprocessed_${model}.csv data/preprocessed_${baseline}.csv results/plot_${model}.png results/plot_${baseline}.pdf > results/log_${model}.log

# sh correlate_RT.sh exp1_dep-alphaselect_0.120_0 exp1_dep-alphaselect_1.0_0