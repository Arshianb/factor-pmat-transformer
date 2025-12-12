#!/bin/sh
env="football"
scenario="academy_counterattack_easy" # academy_pass_and_shoot_with_keeper academy_3_vs_1_with_keeper academy_counterattack_easy
n_agent=11
algo="pmat"
exp="single"
seed=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=0 "/linux-MAT-env-python3.9/bin/python" train/train_football.py --use_linear_lr_decay ${False} --n_head  4 --n_embd 128 --seed ${seed} --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario ${scenario} --n_agent ${n_agent} --lr 5e-4 --entropy_coef 0.01 --max_grad_norm 0.5 --eval_episodes 32 --n_training_threads 64 --n_rollout_threads 64 --num_mini_batch 1 --episode_length 200 --eval_interval 25 --num_env_steps 30000000 --ppo_epoch 10 --clip_param 0.05 --use_eval --use_value_active_masks --use_policy_active_masks --ranking_loss_coef 1e-4 --rank_layer_N 2
