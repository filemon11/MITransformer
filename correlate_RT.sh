corpus=$1
model=$2
model_count=$3
shift=$4
cost_param=$5
left_param=$6
tokeniser=$7

for i in $(seq 0 $((model_count -1)))
do
    python -m mitransformer.readingtimes ${model}_${i} ${corpus} ${shift} ${cost_param} ${left_param} ${tokeniser}
done

cd RT
Rscript --vanilla analysis_new.R ${model} ${model_count} ${corpus} ${shift} > results/log_${model}.log