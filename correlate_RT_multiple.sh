model_count=$1

# for i in $(seq 0 $((model_count -1)))
# do
#     python reading_times.py exp1_dep-alphaselect_1.0_${i}
#     cd RT
#     Rscript --vanilla preproc.R data/words_processed_exp1_dep-alphaselect_1.0_${i}.csv data/preprocessed_exp1_dep-alphaselect_1.0_${i}.csv
#     cd ..
# done

# for i in $(LC_ALL=C seq 0.1 .1 1.0);
# do
#      echo "Correlating reading times for alpha=${i} ..."
#      sh correlate_RT.sh exp1_dep-alphaselect_${i} exp1_dep-alphaselect_1.0 ${model_count} 0
# done

for i in 0.105 0.175;  #$(LC_ALL=C seq 0.105 .005 0.195);
do
     echo "Correlating reading times for alpha=${i} ..."
     sh correlate_RT.sh exp1_dep-alphaselect_${i} exp1_dep-alphaselect_1.0 ${model_count} 0
done