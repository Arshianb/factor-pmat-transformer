import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path

sys.path.append("../../")

try:
    from mat.envs.football.football_env import FootballEnv
    from mat.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
    from mat.algorithms.utils.util import check
except ImportError as e:
    print(f"Import Error: {e}. Please ensure the 'mat' directory structure is correct.")
    def check(input):
        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input).float()
        return input

ENV_NAME = "football"
SCENARIO = 'academy_counterattack_easy'
N_AGENT = 11
N_ROLLOUT_THREADS = 32 
SEED = 1
CUDA = True
RUN_DIR = "./run_dir" 

PRETRAIN_EPOCHS = 10
COLLECTION_EPISODES = 50
BATCH_SIZE = 128 
LR = 3e-4
KL_WEIGHT = 1e-4

OBS_DIM = 217  
ACTION_DIM = 19 
LATENT_DIM = 4 

ENCODER_INPUT_SIZE = N_AGENT  
EXTRA_DECODER_INPUT = OBS_DIM + ACTION_DIM 
RECONSTRUCT_SIZE = OBS_DIM + 1   

def make_train_env(run_dir):
    """Creates the vectorized Football environment using global variables."""

    def get_env_fn(rank):
        def init_env():
            if ENV_NAME == "football":
                env_args = {
                    "scenario": SCENARIO,
                    "n_agent": N_AGENT,
                    "reward": "scoring",
                    "eval": False,
                    "thread_num": rank
                }

                env = FootballEnv(env_args=env_args, run_dir=run_dir)
            else:
                print("Can not support the " + ENV_NAME + " environment.")
                raise NotImplementedError

            env.seed(SEED + rank * 1000)
            return env

        return init_env

    if N_ROLLOUT_THREADS == 1: 
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(N_ROLLOUT_THREADS)]) 

class LinearVAE(nn.Module):
    def __init__(self, features, input_size, extra_decoder_input, reconstruct_size):

        super(LinearVAE, self).__init__()

        if torch.cuda.is_available() and CUDA:
            self.tpdv = dict(dtype=torch.float32, device="cuda:0")
        else:
            self.tpdv = dict(dtype=torch.float32, device="cpu")

        HIDDEN = 256
        self.features = features

        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=HIDDEN),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN, out_features=2 * features)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=features + extra_decoder_input, out_features=HIDDEN),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN, out_features=HIDDEN),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN, out_features=reconstruct_size),
        )
        self.to(self.tpdv["device"])

    def encode(self, x):
        x = check(x).to(**self.tpdv)
        x_enc = self.encoder(x)
        mu = x_enc[..., :self.features]
        log_var = x_enc[..., self.features:]
        _ = self.reparameterize(mu, log_var)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        sample = mu + (eps * std)
        return sample

    def forward(self, x, xp):
        x = check(x).to(**self.tpdv)
        xp = check(xp).to(**self.tpdv)

        x_enc = self.encoder(x)
        mu = x_enc[..., :self.features]
        log_var = x_enc[..., self.features:]

        z = self.reparameterize(mu, log_var)
        dec_input = torch.cat([z, xp], dim=-1)  

        reconstruction = self.decoder(dec_input) 

        pred_next_obs = reconstruction[..., :-1]
        pred_reward = reconstruction[..., -1:].unsqueeze(-1).squeeze(-1)
        if pred_reward.ndim == 1:
            pred_reward = pred_reward.unsqueeze(-1)

        return pred_next_obs, pred_reward, mu, log_var

class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, obs, actions, rewards, next_obs, agent_id):
        data = (obs, actions, rewards, next_obs, agent_id)
        if len(self.storage) < self.max_size:
            self.storage.append(data)
        else:
            self.storage[self.ptr] = data
            self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)

        return (np.stack([self.storage[i][0] for i in ind]), 
                np.stack([self.storage[i][1] for i in ind]),  
                np.stack([self.storage[i][2] for i in ind]), 
                np.stack([self.storage[i][3] for i in ind]), 
                np.stack([self.storage[i][4] for i in ind]))  


def vae_loss_function(pred_next_obs, real_next_obs, pred_reward, real_reward, mu, log_var):
    recon_loss = F.mse_loss(pred_next_obs, real_next_obs, reduction='mean')
    reward_loss = F.mse_loss(pred_reward, real_reward, reduction='mean')
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    kl_loss = kl_loss.mean()
    total_loss = recon_loss + reward_loss + (KL_WEIGHT * kl_loss)
    return total_loss, recon_loss, reward_loss, kl_loss

def main():
    print(f"Initializing SePS VAE training for {N_AGENT} agents in {SCENARIO}...")

    Path(RUN_DIR).mkdir(parents=True, exist_ok=True)
    Path("./enc_help_transformer").mkdir(parents=True, exist_ok=True)

    envs = make_train_env(RUN_DIR)

    model = LinearVAE(
        features=LATENT_DIM,
        input_size=ENCODER_INPUT_SIZE,
        extra_decoder_input=EXTRA_DECODER_INPUT,
        reconstruct_size=RECONSTRUCT_SIZE
    )
    optimizer = optim.Adam(model.parameters(), lr=LR)
    buffer = ReplayBuffer()

    print("Starting Data Collection (Random Policy)...")
    obs, share_obs, ava = envs.reset()

    eye_act = np.eye(ACTION_DIM, dtype=np.float32)
    eye_agents = np.eye(N_AGENT, dtype=np.float32)

    from tqdm import trange

    print(f"Starting Data Collection and VAE Training...")
    model.train()

    for episode in trange(COLLECTION_EPISODES, desc="Data Collection & Training"):
        for step in range(100):
            actions_int = []
            for t_idx in range(N_ROLLOUT_THREADS):
                thread_acts = []
                for a_idx in range(N_AGENT):
                    valid_indices = np.where(ava[t_idx][a_idx] > 0)[0]
                    act = np.random.choice(valid_indices) if len(valid_indices) > 0 else 0
                    thread_acts.append(act)
                actions_int.append(thread_acts)

            actions_int = np.array(actions_int) 

            next_obs, next_share_obs, rewards, dones, infos, next_ava, agent_id = envs.step(actions_int)

            actions_onehot = eye_act[actions_int] 

            for t in range(N_ROLLOUT_THREADS):
                r = rewards[t]
                if r.ndim == 1:
                    r = r.reshape(-1, 1) 
                buffer.add(obs[t], actions_onehot[t], r, next_obs[t], agent_id[t])

            obs = next_obs
            ava = next_ava

            if N_ROLLOUT_THREADS == 1 and np.all(dones[0]):
                obs, share_obs, ava = envs.reset() 

        if len(buffer.storage) < BATCH_SIZE:
            print(f"Buffer size: {len(buffer.storage)} (Waiting for data, need >= {BATCH_SIZE})")
            continue

        for epoch in range(PRETRAIN_EPOCHS):
            b_obs, b_acts, b_rews, b_next_obs, agent_id = buffer.sample(BATCH_SIZE)
            B = b_obs.shape[0]

            xp = np.concatenate([b_obs, b_acts], axis=-1)
            xp_flat = xp.reshape(B * N_AGENT, -1) 

            agent_id = np.eye(11)[agent_id]
            target_next_obs = b_next_obs.reshape(B * N_AGENT, -1)  
            agent_ids_flat = agent_id.reshape(B * N_AGENT, -1)
            target_rewards = b_rews.reshape(B * N_AGENT, -1)
            pred_next_obs, pred_reward, mu, log_var = model(agent_ids_flat, xp_flat)

            target_next_obs_t = check(target_next_obs).to(**model.tpdv)
            target_rewards_t = check(target_rewards).to(**model.tpdv)

            loss, recon_l, rew_l, kl_l = vae_loss_function(
                pred_next_obs, target_next_obs_t,
                pred_reward, target_rewards_t,
                mu, log_var
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print({
                "Episode": episode,
                "Epoch": epoch,
                "Total": f"{loss.item():.4f}",
                "Recon": f"{recon_l.item():.4f}",
                "Rew":   f"{rew_l.item():.4f}",
                "KL":    f"{kl_l.item():.4f}",
            })

        if episode % 10 == 0:
            torch.save(model.state_dict(), f"./enc_help_transformer/{episode}.pt")

    print(f"Collection Done. Buffer size: {len(buffer.storage)}")
    print("Training Finished.")

    envs.close()


if __name__ == "__main__":
    main()
