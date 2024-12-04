
prefix="-m torch.distributed.launch --nproc-per-node=1 main.py"

general_params='--n_workers 32 --device cuda --use_ddp 1 --first_k none --first_k_eval_test none'
general_hyperopt_params='--batch_size 128 --epochs 100 --abort_after 1 --n_trials=50'
hyperopt_selection='--n_embd 200:1000 --dropout 0.0:0.6 --learning_rate 1e-6:1e-2'

core="${general_params} hyperopt ${hyperopt_selection} ${general_hyperopt_params}"

python main.py --first_k none --first_k_eval_test none dataprep

python ${prefix} \
    --name exp1_dep-supervised \
    ${core} \
    --dependency_mode 'supervised' \
    --loss_alpha 0.0:1.0
    
python ${prefix} \
    --name exp1_dep0 \
    ${core} \
    --dependency_mode 'standard'

python ${prefix} \
    --name exp1_dep-in \
    ${core} \
    --dependency_mode input 
