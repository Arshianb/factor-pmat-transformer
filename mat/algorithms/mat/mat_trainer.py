import numpy as np
import torch
import torch.nn as nn
from mat.utils.util import get_gard_norm, huber_loss, mse_loss
from mat.utils.valuenorm import ValueNorm
from mat.algorithms.utils.util import check
import torch.nn.functional as F

class MATTrainer:
    """
    Trainer class for MAT to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 num_agents,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        self.num_agents = num_agents

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.ranking_loss_coef = args.ranking_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        self.dec_actor = args.dec_actor
        
        if self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """

        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                    self.clip_param)

        if self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        # if self._use_value_active_masks and not self.dec_actor:
        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss
    def vae_loss_function(self, pred_next_obs, real_next_obs, pred_reward, real_reward, mu, log_std):
        # 1. Reconstruction Loss (Observation)
        recon_loss = F.mse_loss(pred_next_obs, real_next_obs, reduction='mean')
        
        # 2. Reward Loss
        reward_loss = F.mse_loss(pred_reward, real_reward, reduction='mean')
        
        # 3. KL Divergence (Forces z to be normal distribution)
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + log_std - mu.pow(2) - log_std.exp())
        kl_loss = kl_loss.mean() # Normalize by batch size

        # Total Loss (You might need to tune the beta weight for kl_loss)
        beta = 0.01  # Small weight often works better for RL representations
        total_loss = recon_loss + reward_loss + (beta * kl_loss)
        
        return total_loss, recon_loss, reward_loss, kl_loss
    def ppo_update(self, sample):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        share_obs_batch, obs_batch, next_obs_batch, _real_reward, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, agent_ids_batch, old_action_log_probs_batch, \
        old_seq_log_probs_batch, seqs_batch, adv_targ, available_actions_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        old_seq_log_probs_batch = check(old_seq_log_probs_batch).to(**self.tpdv)
        seqs_batch = check(seqs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        next_obs_batch_raw = check(next_obs_batch).to(**self.tpdv)
        _real_reward = check(_real_reward).to(**self.tpdv)


        for p in self.policy.transformer.parameters():
            p.requires_grad = True
        for p in self.policy.help_encoder_model.parameters():
            p.requires_grad = False
        # Reshape to do in a single forward pass for all steps
                                                                              
        values, action_log_probs, entropy, seq_log_probs, seq_entropy, \
        pred_next_obs, pred_reward, mu, log_std = self.policy.evaluate_actions(
            agent_ids_batch,
            share_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            seqs_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch
        )
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_loss = (-torch.sum(torch.min(surr1, surr2),
                                      dim=-1,
                                      keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        # scoring network update
        seq_imp_weights = torch.exp(seq_log_probs - old_seq_log_probs_batch)
        seq_adv_targ = adv_targ.view(-1, seqs_batch.shape[1]).mean(dim=1, keepdim=True)
        seq_surr1 = seq_imp_weights * seq_adv_targ
        seq_surr2 = torch.clamp(seq_imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * seq_adv_targ
        ranking_loss = -torch.sum(torch.min(seq_surr1, seq_surr2), dim=-1, keepdim=True).mean()
        
        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        loss = policy_loss - entropy * self.entropy_coef + value_loss * self.value_loss_coef + ranking_loss * self.ranking_loss_coef \
            - seq_entropy * self.entropy_coef

        self.policy.optimizer.zero_grad()
        loss.backward()

        if self._use_max_grad_norm:
            grad_norm = nn.utils.clip_grad_norm_(self.policy.transformer.parameters(), self.max_grad_norm)
        else:
            grad_norm = get_gard_norm(self.policy.transformer.parameters())

        self.policy.optimizer.step()


        for p in self.policy.transformer.parameters():
            p.requires_grad = False
        for p in self.policy.help_encoder_model.parameters():
            p.requires_grad = True
        T = next_obs_batch_raw.numel() // (self.num_agents * self.policy.obs_dim)
        
        real_next_obs = next_obs_batch_raw.view(T * self.num_agents, -1).detach()
        _real_reward = torch.as_tensor(_real_reward, dtype=torch.float32, device=self.device)
        real_reward = _real_reward.view(T * self.num_agents, -1).detach() 
        self.policy.optimizer_encoder_helper.zero_grad()
        values, action_log_probs, entropy, seq_log_probs, seq_entropy, \
        pred_next_obs, pred_reward, mu, log_std = self.policy.evaluate_actions(
            agent_ids_batch,
            share_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            seqs_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch
        )
        pred_next_obs = pred_next_obs.view(T * self.num_agents, -1)
        pred_reward = pred_reward.view(T * self.num_agents, -1)
        mu = mu.view(T * self.num_agents, -1)
        log_std = log_std.view(T * self.num_agents, -1)


        # print(f"pred_next_obs device: {pred_next_obs.device}")
        # print(f"real_next_obs device: {real_next_obs.device}")
        # print(f"pred_reward device: {pred_reward.device}")
        # print(f"real_reward device: {real_reward.device}")
        # print(f"mu device: {mu.device}")
        # print(f"log_std device: {log_std.device}")

        
        loss_vae, recon_loss, reward_loss, kl_loss = self.vae_loss_function(
            pred_next_obs, real_next_obs, 
            pred_reward, real_reward, 
            mu, log_std
        )

        loss_vae.backward()

        if self._use_max_grad_norm:
            enc_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy.help_encoder_model.parameters(), self.max_grad_norm
            )
        else:
            enc_grad_norm = get_gard_norm(self.policy.help_encoder_model.parameters()) 

        self.policy.optimizer_encoder_helper.step()

        # --- UPDATED RETURN STATEMENT ---
        return value_loss, grad_norm, enc_grad_norm, policy_loss, \
            loss_vae, recon_loss, reward_loss, kl_loss, \
            entropy, grad_norm, imp_weights, seq_imp_weights, ranking_loss

    def train(self, buffer):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        advantages_copy = buffer.advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (buffer.advantages - mean_advantages) / (std_advantages + 1e-5)
        

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['ranking_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['enc_help_grad_norm'] = 0
        
        # --- NEW/MODIFIED VAE LOSS LOGGING ---
        train_info['vae_total_loss'] = 0     # The overall loss used for backprop
        train_info['vae_recon_loss'] = 0     # Reconstruction Loss (MSE on next_obs)
        train_info['vae_reward_loss'] = 0    # Reward Loss (MSE on reward)
        train_info['vae_kl_loss'] = 0        # KL Divergence Loss (Regularization)
        # --- END NEW/MODIFIED ---

        train_info['ratio'] = 0
        train_info['seq_ratio'] = 0

        for _ in range(self.ppo_epoch):
            data_generator = buffer.feed_forward_generator_transformer(advantages, self.num_mini_batch)

            for sample in data_generator:

                # UNPACKING NEW VAE LOSS COMPONENTS:
                value_loss, critic_grad_norm, enc_help_grad_norm, policy_loss, \
                vae_total_loss, vae_recon_loss, vae_reward_loss, vae_kl_loss, \
                dist_entropy, actor_grad_norm, imp_weights, seq_imp_weights, ranking_loss \
                    = self.ppo_update(sample)

                # LOGGING UPDATES:
                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['ranking_loss'] += ranking_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['enc_help_grad_norm'] += enc_help_grad_norm
                
                # LOGGING VAE LOSSES:
                train_info['vae_total_loss'] += vae_total_loss.item()
                train_info['vae_recon_loss'] += vae_recon_loss.item()
                train_info['vae_reward_loss'] += vae_reward_loss.item()
                train_info['vae_kl_loss'] += vae_kl_loss.item()
                
                train_info['ratio'] += imp_weights.mean()
                train_info['seq_ratio'] += seq_imp_weights.mean()
        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info

    def prep_training(self):
        self.policy.train()

    def prep_rollout(self):
        self.policy.eval()
