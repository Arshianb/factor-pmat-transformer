import torch
import numpy as np
from mat.utils.util import update_linear_schedule, update_exp_schedule
from mat.utils.util import get_shape_from_obs_space, get_shape_from_act_space
from mat.algorithms.utils.util import check


class TransformerPolicy:
    """
    MAT Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, num_agents, device=torch.device("cpu")):
        self.agent2factor=[
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1]
        ]
        self.agent2factor = torch.tensor(self.agent2factor, device = device, dtype=torch.long)
        self.device = device
        self.algorithm_name = args.algorithm_name
        self.lr = args.lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self._use_policy_active_masks = args.use_policy_active_masks
        if act_space.__class__.__name__ == 'Box':
            self.action_type = 'Continuous'
        else:
            self.action_type = 'Discrete'

        self.obs_dim = get_shape_from_obs_space(obs_space)[0]
        self.share_obs_dim = get_shape_from_obs_space(cent_obs_space)[0]
        if self.action_type == 'Discrete':
            self.act_dim = act_space.n
            self.act_num = 1
        else:
            print("act high: ", act_space.high)
            self.act_dim = act_space.shape[0]
            self.act_num = self.act_dim

        print("obs_dim: ", self.obs_dim)
        print("share_obs_dim: ", self.share_obs_dim)
        print("act_dim: ", self.act_dim)

        self.num_agents = num_agents
        self.tpdv = dict(dtype=torch.float32, device=device)

        if self.algorithm_name == "pmat":
            from mat.algorithms.mat.algorithm.pma_transformer import MultiAgentTransformer as MAT
        else:
            raise NotImplementedError

        self.transformer = MAT(self.share_obs_dim, self.obs_dim, self.act_dim, num_agents,
                               n_block=args.n_block, n_embd=args.n_embd, n_head=args.n_head,
                               n_ranking_layer=args.rank_layer_N,
                               encode_state=args.encode_state, device=device,
                               action_type=self.action_type)
                            #    dec_actor=args.dec_actor,
                            #    share_actor=args.share_actor)
        from mat.algorithms.mat.algorithm.help_encoder import LinearVAE
        # self.help_encoder_model = LinearVAE(self.obs_dim, 1, self.obs_dim, device=device).to(**self.tpdv)
        self.help_encoder_model = LinearVAE(
            features=4,
            input_size=num_agents,
            extra_decoder_input=self.obs_dim + 1,  # obs + action
            reconstruct_size=self.obs_dim + 1      # next_obs + reward
        )
        if args.env_name == "hands":
            self.transformer.zero_std()
            self.help_encoder_model.zero_std()

        # count the volume of parameters of model
        # Total_params = 0
        # Trainable_params = 0
        # NonTrainable_params = 0
        # for param in self.transformer.parameters():
        #     mulValue = np.prod(param.size())
        #     Total_params += mulValue
        #     if param.requires_grad:
        #         Trainable_params += mulValue
        #     else:
        #         NonTrainable_params += mulValue
        # print(f'Total params: {Total_params}')
        # print(f'Trainable params: {Trainable_params}')
        # print(f'Non-trainable params: {NonTrainable_params}')

        self.optimizer = torch.optim.Adam(self.transformer.parameters(),
                                        #   lr=1e-3, eps=self.opti_eps,
                                          lr=self.lr, eps=self.opti_eps,
                                          weight_decay=self.weight_decay)
        self.optimizer_encoder_helper = torch.optim.Adam(self.help_encoder_model.parameters(),
                                          lr=1e-4, eps=self.opti_eps,
                                          weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        # update_linear_schedule(self.optimizer, episode, episodes, self.lr)
        # update_linear_schedule(self.optimizer, episode, episodes, 1e-3)
        # update_linear_schedule(self.optimizer_encoder_helper, episode, episodes, self.lr)
        update_exp_schedule(self.optimizer_encoder_helper, episode, initial_lr=1e-4, decay_rate=0.97)

    def get_actions(self, agent_ids_batch, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False, help_enc_output_name=""):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """

        cent_obs = cent_obs.reshape(-1, self.num_agents, self.share_obs_dim)
        agent_ids = agent_ids_batch.reshape(-1, self.num_agents, 1)
        obs = obs.reshape(-1, self.num_agents, self.obs_dim)

        if available_actions is not None:
            available_actions = available_actions.reshape(-1, self.num_agents, self.act_dim)
        
        agent_ids_t = torch.as_tensor(agent_ids, device=self.device, dtype=torch.long)  # [B, A, 1]
        idx = agent_ids_t.squeeze(-1)   
        agent_ids = self.agent2factor[idx]
        # col_sums = agent_ids.sum(dim=0)
        # keep_mask = col_sums != 0
        # agent_ids = agent_ids[:, keep_mask]
        actions, action_log_probs, values, seq_log_probs, seqs = self.transformer.get_actions(
                                                                         agent_ids,
                                                                         cent_obs,
                                                                         obs,
                                                                         available_actions,
                                                                         deterministic)

        actions = actions.view(-1, self.act_num)
        action_log_probs = action_log_probs.view(-1, self.act_num)
        values = values.view(-1, 1)
        seq_log_probs = seq_log_probs.view(-1, 1)

        # unused, just for compatibility
        rnn_states_actor = check(rnn_states_actor).to(**self.tpdv)
        rnn_states_critic = check(rnn_states_critic).to(**self.tpdv)
        return values, actions, action_log_probs, seq_log_probs, seqs, rnn_states_actor, rnn_states_critic

    def get_values(self, agent_ids_batch, cent_obs, obs, rnn_states_critic, masks, available_actions=None):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """

        cent_obs = cent_obs.reshape(-1, self.num_agents, self.share_obs_dim)
        obs = obs.reshape(-1, self.num_agents, self.obs_dim)
        agent_ids = agent_ids_batch.reshape(-1, self.num_agents, 1)
        if available_actions is not None:
            available_actions = available_actions.reshape(-1, self.num_agents, self.act_dim)

        agent_ids_t = torch.as_tensor(agent_ids, device=self.device, dtype=torch.long)  # [B, A, 1]
        idx = agent_ids_t.squeeze(-1)   
        agent_ids = self.agent2factor[idx]

        # col_sums = agent_ids.sum(dim=0)
        # keep_mask = col_sums != 0
        # agent_ids = agent_ids[:, keep_mask]
        values = self.transformer.get_values(cent_obs, obs, agent_ids, available_actions)

        values = values.view(-1, 1)

        return values

    def evaluate_actions(self, agent_ids_batch, cent_obs, obs, rnn_states_actor, rnn_states_critic, actions, seqs, masks,
                            available_actions=None, active_masks=None):
            # ... (parameters docstring is kept)
            
            # Reshaping inputs (kept as is)
            cent_obs = cent_obs.reshape(-1, self.num_agents, self.share_obs_dim)
            obs = obs.reshape(-1, self.num_agents, self.obs_dim)
            agent_ids = agent_ids_batch.reshape(-1, self.num_agents, 1)
            actions = actions.reshape(-1, self.num_agents, self.act_num)
            if available_actions is not None:
                available_actions = available_actions.reshape(-1, self.num_agents, self.act_dim)

            agent_ids_t = torch.as_tensor(agent_ids, device=self.device, dtype=torch.long)  # [B, A, 1]
            idx = agent_ids_t.squeeze(-1)   
            agent_ids = self.agent2factor[idx]

            # col_sums = agent_ids.sum(dim=0)
            # keep_mask = col_sums != 0
            # agent_ids = agent_ids[:, keep_mask]
            action_log_probs, values, entropy, seq_log_probs, seq_entropy = self.transformer(cent_obs, obs, actions, seqs, agent_ids, available_actions)
            xp = np.concatenate([obs, actions], axis=-1)
            eye = np.eye(self.num_agents)
            agent_ids = np.repeat(eye[np.newaxis, :, :], obs.shape[0], axis=0)
            pred_next_obs, pred_reward, mu, log_std = self.help_encoder_model(agent_ids, xp)

            # Reshaping outputs (kept as is)
            action_log_probs = action_log_probs.view(-1, self.act_num)
            seq_log_probs = seq_log_probs.view(-1, 1)
            values = values.view(-1, 1)
            entropy = entropy.view(-1, self.act_num)
            seq_entropy = seq_entropy.view(-1, 1).mean()

            if self._use_policy_active_masks and active_masks is not None:
                entropy = (entropy*active_masks).sum()/active_masks.sum()
            else:
                entropy = entropy.mean()

            # UPDATED RETURN STATEMENT:
            # We now return the VAE components needed for the loss calculation.
            return values, action_log_probs, entropy, seq_log_probs, seq_entropy, \
                pred_next_obs, pred_reward, mu, log_std

    def act(self, agent_ids_batch, cent_obs, obs, rnn_states_actor, masks, available_actions=None, deterministic=True):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """

        # this function is just a wrapper for compatibility
        rnn_states_critic = np.zeros_like(rnn_states_actor)
        _, actions, _, _, _, rnn_states_actor, _ = self.get_actions(agent_ids_batch, cent_obs,
                                                              obs,
                                                              rnn_states_actor,
                                                              rnn_states_critic,
                                                              masks,
                                                              available_actions,
                                                              deterministic)

        return actions, rnn_states_actor

    def save(self, save_dir, episode):
        torch.save(self.transformer.state_dict(), str(save_dir) + "/transformer_" + str(episode) + ".pt")
        torch.save(self.help_encoder_model.state_dict(), str(save_dir) + "/enc_help_transformer_" + str(episode) + ".pt")

    def restore(self, model_dir):
        transformer_state_dict = torch.load(model_dir)
        self.transformer.load_state_dict(transformer_state_dict)
    def restore_help_enc(self, model_dir):
        transformer_state_dict = torch.load(model_dir)
        self.help_encoder_model.load_state_dict(transformer_state_dict)
        # self.transformer.reset_std()

    def train(self):
        self.transformer.train()

    def eval(self):
        self.transformer.eval()

