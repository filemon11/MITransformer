model=$1 # "exp1_dep-supervised_det_39"
baseline=$2
model_count=$3

for i in $(seq 0 $((model_count -1)))
do
    python reading_times.py ${model}_${i}
    python reading_times.py ${baseline}_${i}
done
cd RT
for i in $(seq 0 $((model_count -1)))
do
    Rscript --vanilla preproc.R data/words_processed_${model}_${i}.csv data/preprocessed_${model}_${i}.csv
    Rscript --vanilla preproc.R data/words_processed_${baseline}_${i}.csv data/preprocessed_${baseline}_${i}.csv
done
Rscript --vanilla analysis.R ${model} ${baseline} ${model_count} > results/log_${model}.log

# sh correlate_RT.sh exp1_dep-alphaselect_0.120_0 exp1_dep-alphaselect_1.0_0