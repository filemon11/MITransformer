model=$1 # "exp1_dep-supervised_det_39"
baseline=$2
model_count=$3
prepare_baseline=$4

for i in $(seq 0 $((model_count -1)))
do
    python reading_times.py ${model}_${i}
    cd RT
    Rscript --vanilla preproc.R data/words_processed_${model}_${i}.csv data/preprocessed_${model}_${i}.csv
    cd ..
    if [ $prepare_baseline = 1 ]
    then
        python reading_times.py ${baseline}_${i}
        cd RT
        Rscript --vanilla preproc.R data/words_processed_${baseline}_${i}.csv data/preprocessed_${baseline}_${i}.csv
        cd ..
    fi
done

cd RT
Rscript --vanilla analysis.R ${model} ${baseline} ${model_count} > results/log_${model}.log

# sh correlate_RT.sh exp2_lstm_0.120_0 exp2_lstm_1.0_0