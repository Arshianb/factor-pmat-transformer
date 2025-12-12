import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from torch.distributions import Categorical
from mat.algorithms.utils.util import check, init, reindex_tensor, cal_restore_index
from mat.algorithms.utils.transformer_act import discrete_autoregreesive_act, another_discrete_autoregreesive_act
from mat.algorithms.utils.transformer_act import discrete_parallel_act, another_discrete_parallel_act
from mat.algorithms.utils.transformer_act import continuous_autoregreesive_act
from mat.algorithms.utils.transformer_act import continuous_parallel_act
from itertools import permutations
from torch.distributions import Categorical


def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class ScoringBlock(nn.Module):

    def __init__(self, emb_dim, hid_dim, num_layers):
        super(ScoringBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(emb_dim),
            init_(nn.Linear(emb_dim, hid_dim), activate=True), nn.GELU(), nn.LayerNorm(hid_dim),
            *[init_(nn.Linear(hid_dim, hid_dim), activate=True), nn.GELU(), nn.LayerNorm(hid_dim)] * (num_layers - 1),
            init_(nn.Linear(hid_dim, 1))
        )
        self.output = nn.Sigmoid()
    
    def forward(self, obs_rep):
        r = self.mlp(obs_rep)
        r = 9 * self.output(r) + 1
        return r


def sample_seq_by_score_batched(batched_scores, deterministic):
    batch_size, seq_length = batched_scores.size()
    
    sampled_seq_batch = torch.zeros((batch_size, seq_length), dtype=torch.long, device=batched_scores.device)    
    sampled_seq_log_prob_batch = torch.zeros(batch_size, dtype=torch.float, device=batched_scores.device)
    
    remaining = torch.ones((batch_size, seq_length), dtype=torch.bool,device=batched_scores.device)
    current_scores = batched_scores.clone()

    for item in range(seq_length):
        masked_scores = current_scores * remaining.float()
        probabilities = masked_scores / (masked_scores.sum(dim=1, keepdim=True))

        dist = Categorical(probs=probabilities)
        
        sampled_indices = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        sampled_seq_batch[:, item] = sampled_indices
        sampled_seq_log_prob_batch += dist.log_prob(sampled_indices)
        
        remaining[torch.arange(batch_size), sampled_indices] = False
    
    return sampled_seq_batch, sampled_seq_log_prob_batch

def masked_softmax(logits, mask, dim=-1, eps=1e-9):
    mask = mask.to(dtype=logits.dtype)
    logits = logits.masked_fill(mask == 0, -1e9)

    probs = torch.softmax(logits, dim=dim)
    probs = probs * mask 
    denom = probs.sum(dim=dim, keepdim=True).clamp_min(eps)
    probs = probs / denom
    return probs

def cal_seq_logprob_batched(batched_scores, batched_seqs):
    batch_size, seq_length = batched_scores.size()
    
    remaining = torch.ones((batch_size, seq_length), dtype=torch.bool,device=batched_scores.device)
    current_scores = batched_scores.clone()
    seq_log_probs = torch.zeros((batch_size, 1), dtype=torch.float, device=batched_scores.device)
    
    for item in range(seq_length):
        masked_scores = current_scores * remaining.float()
        probabilities = masked_scores / (masked_scores.sum(dim=1, keepdim=True))

        dist = Categorical(probs=probabilities)
        seq_log_probs += dist.log_prob(batched_seqs[:, item]).unsqueeze(1)

        remaining[torch.arange(batch_size), batched_seqs[:, item].long()] = False
        
    return seq_log_probs


class FactorAttention(nn.Module):

    def __init__(self, n_embd, n_head, n_agent):
        super(FactorAttention, self).__init__()

        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.n_agent = n_agent

        self.node_q = init_(nn.Linear(n_embd, n_embd))
        self.node_k = init_(nn.Linear(n_embd, n_embd))
        self.node_v = init_(nn.Linear(n_embd, n_embd))

        self.factor_q = init_(nn.Linear(n_embd, n_embd))
        self.factor_k = init_(nn.Linear(n_embd, n_embd))
        self.factor_v = init_(nn.Linear(n_embd, n_embd))

        self.proj_node = init_(nn.Linear(n_embd, n_embd))
        self.proj_factor = init_(nn.Linear(n_embd, n_embd))

    def _split_heads(self, x, L):
        B = x.size(0)
        return x.view(B, L, self.n_head, self.head_dim).transpose(1, 2)  

    def forward(self, node_x, factor_x, agent2factor_mask, factor2agent_mask):
        B, N_agent, D = node_x.size()
        # print("node_x = ", node_x.size())
        _, N_factor, _ = factor_x.size()
        # print("factor_x = ", factor_x.size())

        # --------- Step 1: factors attend to agents (compute delta_factor) ---------
        q_f = self._split_heads(self.factor_q(factor_x), N_factor)
        # print("q_f = ", q_f.size())
        k_n = self._split_heads(self.node_k(node_x), N_agent)
        # print("k_n = ", k_n.size())
        v_n = self._split_heads(self.node_v(node_x), N_agent)
        # print("v_n = ", v_n.size())
        att_f2n = (q_f @ k_n.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        # print("att_f2n = ", att_f2n.size())
        # print("factor2agent_mask = ", att_f2n.size())
        # att_f2n = att_f2n.masked_fill(
        #     factor2agent_mask[:, :, :N_factor, :N_agent] == 0,
        #     float('-inf')
        # )
        att_f2n = masked_softmax(att_f2n, factor2agent_mask[:, :, :N_factor, :N_agent], dim=-1)
        delta_factor = att_f2n @ v_n
        # print("delta_factor = ", delta_factor.size())
        delta_factor = delta_factor.transpose(1, 2).contiguous().view(B, N_factor, D)
        delta_factor = self.proj_factor(delta_factor)
        # row_is_nan = torch.isnan(delta_factor).all(dim=-1, keepdim=True)   # (B, N_factor, 1)
        # if row_is_nan.any():
        #     delta_factor = torch.where(row_is_nan, torch.zeros_like(delta_factor), delta_factor)
        
        # build UPDATED factors (for internal use in node update)
        factor_updated = factor_x + delta_factor

        # --------- Step 2: agents attend to UPDATED factors ---------
        q_n = self._split_heads(self.node_q(node_x), N_agent)
        k_f = self._split_heads(self.factor_k(factor_updated), N_factor)
        v_f = self._split_heads(self.factor_v(factor_updated), N_factor)

        att_n2f = (q_n @ k_f.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att_n2f = att_n2f.masked_fill(
            agent2factor_mask[:, :, :N_agent, :N_factor] == 0,
            float('-inf')
        )
        att_n2f = F.softmax(att_n2f, dim=-1)
        delta_node = att_n2f @ v_f
        delta_node = delta_node.transpose(1, 2).contiguous().view(B, N_agent, D)
        delta_node = self.proj_node(delta_node)

        return delta_node, delta_factor


class FactorBlock(nn.Module):

    def __init__(self, n_embd, n_head, n_agent):
        super(FactorBlock, self).__init__()

        self.ln_node1 = nn.LayerNorm(n_embd)
        self.ln_factor1 = nn.LayerNorm(n_embd)
        self.ln_node2 = nn.LayerNorm(n_embd)
        self.ln_factor2 = nn.LayerNorm(n_embd)

        self.attn = FactorAttention(n_embd, n_head, n_agent)

        self.mlp_node = nn.Sequential(
            init_(nn.Linear(n_embd, n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(n_embd, n_embd))
        )
        self.mlp_factor = nn.Sequential(
            init_(nn.Linear(n_embd, n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(n_embd, n_embd))
        )

    def forward(self, node_x, factor_x, agent2factor_mask, factor2agent_mask):
        node_h = self.ln_node1(node_x)
        factor_h = self.ln_factor1(factor_x)
        delta_node, delta_factor = self.attn(node_h, factor_h, agent2factor_mask, factor2agent_mask)
        node_x = node_x + delta_node
        factor_x = factor_x + delta_factor
        node_x = self.ln_node2(node_x + self.mlp_node(node_x))
        factor_x = self.ln_factor2(factor_x + self.mlp_factor(factor_x))
        return node_x, factor_x
class FactorDecodeBlock(nn.Module):
    def __init__(self, n_embd, n_head, n_agent):
        super(FactorDecodeBlock, self).__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.self_attn = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.factor_block = FactorBlock(n_embd, n_head, n_agent)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(n_embd, n_embd))
        )
    def forward(self, x, factor_x, agent2factor_mask, factor2agent_mask):
        x = self.ln1(x + self.self_attn(x, x, x))
        x, factor_x = self.factor_block(x, factor_x, agent2factor_mask, factor2agent_mask)
        x = self.ln3(x + self.mlp(x))

        return x, factor_x
class SelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, n_agent, masked=False):
        super(SelfAttention, self).__init__()

        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        self.proj = init_(nn.Linear(n_embd, n_embd))
        self.register_buffer("mask", torch.tril(torch.ones(n_agent + 1, n_agent + 1))
                             .view(1, 1, n_agent + 1, n_agent + 1))

        self.att_bp = None

    def forward(self, key, value, query):
        B, L, D = query.size()

        k = self.key(key).view(B, L, self.n_head, D // self.n_head).transpose(1, 2) 
        q = self.query(query).view(B, L, self.n_head, D // self.n_head).transpose(1, 2) 
        v = self.value(value).view(B, L, self.n_head, D // self.n_head).transpose(1, 2) 

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if self.masked:
            att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v 
        y = y.transpose(1, 2).contiguous().view(B, L, D) 

        y = self.proj(y)
        return y

class DecodeBlock(nn.Module):

    def __init__(self, n_embd, n_head, n_agent):
        super(DecodeBlock, self).__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.attn1 = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.attn2 = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x, rep_enc):
        x = self.ln1(x + self.attn1(x, x, x))
        x = self.ln2(rep_enc + self.attn2(key=x, value=x, query=rep_enc))
        x = self.ln3(x + self.mlp(x))
        return x

class Encoder(nn.Module):
    def __init__(self, state_dim, obs_dim, n_block, n_embd, n_head,
                 n_agent, encode_state):
        super(Encoder, self).__init__()

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.n_embd = n_embd
        self.n_agent = n_agent
        self.encode_state = encode_state
        self.VERY_LOW = -1e6

        self.state_encoder = nn.Sequential(
            nn.LayerNorm(state_dim),
            init_(nn.Linear(state_dim, n_embd), activate=True),
            nn.GELU()
        )
        self.obs_encoder = nn.Sequential(
            nn.LayerNorm(obs_dim),
            init_(nn.Linear(obs_dim, n_embd), activate=True),
            nn.GELU()
        )

        self.factor_obs_encoder = nn.Sequential(
            nn.LayerNorm(obs_dim),
            init_(nn.Linear(obs_dim, n_embd), activate=True),
            nn.GELU()
        )

        self.blocks = nn.ModuleList([
            FactorBlock(n_embd, n_head, n_agent)
            for _ in range(n_block)
        ])

        self.ln_node = nn.LayerNorm(n_embd)
        self.ln_factor = nn.LayerNorm(n_embd)

        self.head = nn.Sequential(
            init_(nn.Linear(n_embd, n_embd), activate=True),
            nn.GELU(),
            nn.LayerNorm(n_embd),
            init_(nn.Linear(n_embd, 1))
        )

    def _build_factor_obs(self, obs, agent2factor_raw, n_factor):
        B, N_agent, D = obs.size()
        mask = agent2factor_raw 
        denom = mask.sum(dim=1).clamp(min=1.0) 
        mask_exp = mask.view(B, N_agent, n_factor, 1) 
        obs_exp = obs.unsqueeze(2) 
        num = (obs_exp * mask_exp).sum(dim=1) 
        factor_obs = num / denom.view(B, n_factor, 1)
        return factor_obs

    def forward(self, state, obs, agent2factor_mask, n_factors):
        B = agent2factor_mask.size(0)
        agent2factor_mask_bool = agent2factor_mask.bool().view(
            B, 1, self.n_agent, -1
        ) 

        factor2agent_mask_bool = agent2factor_mask_bool.transpose(-1, -2)
        if self.encode_state:
            node_x = self.state_encoder(state) 
        else:
            node_x = self.obs_encoder(obs) 

        factor_obs = self._build_factor_obs(obs, agent2factor_mask, n_factors) 

        factor_x = self.factor_obs_encoder(factor_obs)

        node_x = self.ln_node(node_x)
        factor_x = self.ln_factor(factor_x)

        for block in self.blocks:
            node_x, factor_x = block(
                node_x, factor_x,
                agent2factor_mask_bool,
                factor2agent_mask_bool
            )

        v_loc = self.head(node_x) 
        return v_loc, node_x, factor_x

class Decoder(nn.Module):

    def __init__(self, obs_dim, action_dim, n_block, n_embd, n_head, n_agent,
                 action_type='Discrete', dec_actor=False, share_actor=False):
        super(Decoder, self).__init__()

        self.action_dim = action_dim
        self.n_embd = n_embd
        self.dec_actor = dec_actor
        self.share_actor = share_actor
        self.action_type = action_type

        if action_type != 'Discrete':
            log_std = torch.ones(action_dim)
            self.log_std = torch.nn.Parameter(log_std)

        if self.dec_actor:
            if self.share_actor:
                print("mac_dec!!!!!")
                self.mlp = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                         init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                         init_(nn.Linear(n_embd, action_dim)))
            else:
                self.mlp = nn.ModuleList()
                for n in range(n_agent):
                    actor = nn.Sequential(nn.LayerNorm(obs_dim),
                                          init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                          init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                          init_(nn.Linear(n_embd, action_dim)))
                    self.mlp.append(actor)
        else:
            if action_type == 'Discrete':
                self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim + 1, n_embd, bias=False), activate=True),
                                                    nn.GELU())
            else:
                self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim, n_embd), activate=True), nn.GELU())
            self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                             init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU())
            self.ln = nn.LayerNorm(n_embd)
            self.blocks = nn.Sequential(*[DecodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
            self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                      init_(nn.Linear(n_embd, action_dim)))

    def zero_std(self, device):
        if self.action_type != 'Discrete':
            log_std = torch.zeros(self.action_dim).to(device)
            self.log_std.data = log_std

    def forward(self, action, obs_rep, obs):
        if self.dec_actor:
            if self.share_actor:
                logit = self.mlp(obs)
            else:
                logit = []
                for n in range(len(self.mlp)):
                    logit_n = self.mlp[n](obs[:, n, :])
                    logit.append(logit_n)
                logit = torch.stack(logit, dim=1)
        else:
            action_embeddings = self.action_encoder(action)
            x = self.ln(action_embeddings)
            for block in self.blocks:
                x = block(x, obs_rep)
            logit = self.head(x)

        return logit

class MultiAgentTransformer(nn.Module):
    def __init__(self, state_dim, obs_dim, action_dim, n_agent,
                 n_block, n_embd, n_head,
                 n_ranking_layer, encode_state=False, device=torch.device("cpu"),
                 action_type='Discrete', dec_actor=False, share_actor=False):
        super(MultiAgentTransformer, self).__init__()

        self.n_agent = n_agent
        self.action_dim = action_dim
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_type = action_type
        self.device = device
        state_dim = 37  

        self.encoder = Encoder(
            state_dim, obs_dim, n_block, n_embd, n_head,
            n_agent, encode_state
        )
        self.decoder = Decoder(obs_dim, action_dim, n_block, n_embd, n_head, n_agent,
            self.action_type, dec_actor=dec_actor, share_actor=share_actor)
        
        self.scorer = ScoringBlock(emb_dim=n_embd, hid_dim=64, num_layers=n_ranking_layer)
        
        
        self.to(device)

    def zero_std(self):
        if self.action_type != 'Discrete':
            self.decoder.zero_std(self.device)

    def forward(self, state, obs, action, seq, agent2factor_mask, available_actions=None):
        F = agent2factor_mask.size(-1)
        ori_shape = np.shape(state)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = state.shape[0]
        v_loc, obs_rep, factor_rep = self.encoder(state, obs, agent2factor_mask, F)

        rep_scores = self.scorer(obs_rep).squeeze(-1)
        seq_logprob = cal_seq_logprob_batched(rep_scores, seq.squeeze(0))
        seq_entropy = torch.zeros_like(seq_logprob)
                
        if self.action_type == 'Discrete':
            action = action.long()
            action_log, entropy = discrete_parallel_act(self.decoder, obs_rep, obs, action, batch_size,
                                                        self.n_agent, self.action_dim, self.tpdv, available_actions)
        else:
            action_log, entropy = continuous_parallel_act(self.decoder, obs_rep, obs, action, batch_size,
                                                          self.n_agent, self.action_dim, self.tpdv)

        return action_log, v_loc, entropy, seq_logprob, seq_entropy
    def get_actions(self, agent2factor_mask, state, obs, available_actions=None, deterministic=True):
        F = agent2factor_mask.size(-1)
        ori_shape = np.shape(obs)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions)

        batch_size = obs.shape[0]
        v_loc, obs_rep, factor_rep = self.encoder(state, obs, agent2factor_mask, F)

        rep_scores = self.scorer(obs_rep).squeeze(-1)
        sampled_seq_batch, sampled_seq_log_prob_batch = sample_seq_by_score_batched(rep_scores, deterministic)
        sampled_seq = sampled_seq_batch
        sampled_seq_log_prob = sampled_seq_log_prob_batch        
        
        reindexed_obs_rep = reindex_tensor(obs_rep, sampled_seq_batch)
        if available_actions is not None:
            reindexed_available_actions = reindex_tensor(available_actions, sampled_seq_batch)
        else:
            reindexed_available_actions = None
        restore_index = cal_restore_index(sampled_seq_batch)
        
        if self.action_type == "Discrete":
            output_action, output_action_log = discrete_autoregreesive_act(self.decoder, reindexed_obs_rep, obs, batch_size,
                                                                           self.n_agent, self.action_dim, self.tpdv,
                                                                           reindexed_available_actions, deterministic)
        else:
            output_action, output_action_log = continuous_autoregreesive_act(self.decoder, reindexed_obs_rep, obs, batch_size,
                                                                             self.n_agent, self.action_dim, self.tpdv,
                                                                             deterministic)

        resumed_output_action = reindex_tensor(output_action, restore_index)
        resumed_output_action_log = reindex_tensor(output_action_log, restore_index)
        
        return resumed_output_action, resumed_output_action_log, v_loc, sampled_seq_log_prob, sampled_seq
    def get_values(self, state, obs, agent2factor_mask, available_actions=None):
        F = agent2factor_mask.size(-1)
        ori_shape = np.shape(state)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        v_tot, obs_rep, factor_rep = self.encoder(state, obs, agent2factor_mask, F)
        return v_tot
