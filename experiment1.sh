
prefix="--standalone --nnodes=1 --nproc-per-node=2 main.py"

general_params='--n_workers 32 --device cuda --use_ddp 1 --first_k none --first_k_eval_test none'
general_hyperopt_params='--batch_size 128 --epochs 100 --early_stop_after 5 --n_trials=50 --eval_interval 100 --n_warmup_steps 500 --n_startup_trials 5 --use_steps 1 --max_steps none'
hyperopt_selection='--n_embd 200:1000 --dropout 0.0:0.6 --learning_rate 1e-6:1e-2 '

core="${general_params} hyperopt ${hyperopt_selection} ${general_hyperopt_params}"

python main.py --first_k none --first_k_eval_test none dataprep

torchrun ${prefix} \
    --name exp1_dep-supervised \
    ${core} \
    --dependency_mode 'supervised' \
    --loss_alpha 0.0:1.0
    
torchrun ${prefix} \
    --name exp1_dep0 \
    ${core} \
    --dependency_mode 'standard'

torchrun ${prefix} \
    --name exp1_dep-in \
    ${core} \
    --dependency_mode input 
