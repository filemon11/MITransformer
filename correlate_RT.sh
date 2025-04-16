corpus=$1
model=$2 # "exp1_dep-supervised_det_39"
baseline=$3
model_count=$4
prepare_baseline=$5

for i in $(seq 0 $((model_count -1)))
do
    python reading_times.py ${model}_${i} ${corpus}
    cd RT
    if [ $corpus = "zuco" ]
    then
    Rscript --vanilla preproc_zuco.R data/words_processed_${model}_${i}.csv data/preprocessed_${model}_${i}.csv
    else
    Rscript --vanilla preproc.R data/words_processed_${model}_${i}.csv data/preprocessed_${model}_${i}.csv
    fi
    cd ..
    if [ $prepare_baseline = 1 ]
    then
        python reading_times.py ${baseline}_${i} ${corpus}
        cd RT
        if [ $corpus = "zuco" ]
        then
        Rscript --vanilla preproc_zuco.R data/words_processed_${baseline}_${i}.csv data/preprocessed_${baseline}_${i}.csv
        else
        Rscript --vanilla preproc.R data/words_processed_${baseline}_${i}.csv data/preprocessed_${baseline}_${i}.csv
        fi
        cd ..
    fi
done

cd RT
if [ $corpus = "zuco" ]
then
    Rscript --vanilla analysis_zuco.R ${model} ${baseline} ${model_count} > results/log_${model}.log
else
    Rscript --vanilla analysis.R ${model} ${baseline} ${model_count} > results/log_${model}.log
fi
# sh correlate_RT.sh exp2_lstm_0.120_0 exp2_lstm_1.0_0