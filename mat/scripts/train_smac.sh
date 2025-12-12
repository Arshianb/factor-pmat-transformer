#!/bin/sh
env="StarCraft2"
map="MMM2" # MMM2 10m_vs_11m
algo="pmat"
exp="single"
seed=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=0 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --seed ${seed} --n_training_threads 16 --n_rollout_threads 32 --num_mini_batch 1 --episode_length 100 --num_env_steps 10000000 --lr 5e-4 --ppo_epoch 15 --clip_param 0.05 --save_interval 100000 --use_value_active_masks --use_eval --ranking_loss_coef 1e-4 --rank_layer_N 2