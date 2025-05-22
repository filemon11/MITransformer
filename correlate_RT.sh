corpus=$1
model=$2 # "exp1_dep-supervised_det_39"
baseline=$3
model_count=$4
shift=$5
cost_param=$6
left_param=$7

for i in $(seq 0 $((model_count -1)))
do
    python -m mitransformer.readingtimes ${model}_${i} ${corpus} ${shift} ${cost_param} ${left_param}
    # python reading_times.py ${model}_${i} ${corpus}
done

cd RT
Rscript --vanilla analysis_new.R ${model} ${model_count} ${corpus} > results/log_${model}.log

# sh correlate_RT.sh exp2_lstm_0.120_0 exp2_lstm_1.0_0