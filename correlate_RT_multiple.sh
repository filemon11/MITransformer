model_count=$1

for i in $(seq 0 $((model_count -1)))
do
     python reading_times.py exp2_lstm_1.0_${i}
     cd RT
     Rscript --vanilla preproc.R data/words_processed_exp2_lstm_1.0_${i}.csv data/preprocessed_exp2_lstm_1.0_${i}.csv
     cd ..
done

for i in $(LC_ALL=C seq 0.1 .1 1.0);
do
     echo "Correlating reading times for alpha=${i} ..."
     sh correlate_RT.sh exp2_lstm_${i} exp2_lstm_1.0 ${model_count} 0
done

# for i in $(LC_ALL=C seq 0.105 .005 0.195);
# do
#      echo "Correlating reading times for alpha=${i} ..."
#      sh correlate_RT.sh exp2_lstm_${i} exp2_lstm_1.0 ${model_count} 0
# done