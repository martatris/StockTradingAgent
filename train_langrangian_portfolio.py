import os
import math
import random
import argparse
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------
# Utilities / Data download
# ---------------------------

def prepare_multi_asset_data(tickers: List[str], start="2015-01-01", end="2023-12-31", path="data/multi_asset.csv"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"Downloading tickers: {tickers} from Yahoo Finance...")
    full = pd.DataFrame()
    for t in tickers:
        df = yf.download(t, start=start, end=end, progress=False)
        if df.empty:
            raise RuntimeError(f"No data for {t}")
        full[f"{t}_raw_close"] = df["Close"].values
    # build normalized feature closes for each asset
    for t in tickers:
        raw = full[f"{t}_raw_close"]
        mean = raw.rolling(100, min_periods=1).mean()
        std = raw.rolling(100, min_periods=1).std().replace(0, 1e-8)
        full[f"{t}_close"] = (raw - mean) / std
        full[f"{t}_volume"] = 1.0
    full.dropna(inplace=True)
    full.reset_index(drop=True, inplace=True)
    full.to_csv(path, index=False)
    print(f"Saved prepared dataset -> {path} ({len(full)} rows)")
    return full

# ---------------------------
# Environment
# ---------------------------
class MultiAssetPortfolioEnv(gym.Env):
    """
    Multi-asset environment:
    - Expects DataFrame with columns: {TICKER}_raw_close and {TICKER}_close for features.
    - Action: pre-softmax logits (vector z) -> softmax -> weights (non-negative, sum=1).
    - Observation: window x (per-asset features: normalized close, returns, sma, vol) flattened.
    - Returns info with 'portfolio_value' and 'max_drawdown' each step.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, df: pd.DataFrame, assets: List[str], window: int = 50, transaction_cost: float = 0.0005):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.assets = assets
        self.N = len(self.assets)
        self.window = int(window)
        self.transaction_cost = float(transaction_cost)
        self.length = len(self.df)

        # construct feature columns and compute indicators
        self._prepare_features()

        self.features_per_asset = 4  # close(norm), returns, sma, vol
        obs_dim = self.window * self.N * self.features_per_asset
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        # Action is continuous vector (pre-softmax logits). We let actor output mean and std on this space.
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.N,), dtype=np.float32)

        self.seed()
        self.reset()

    def _prepare_features(self):
        df = self.df.copy()
        for t in self.assets:
            raw = df[f"{t}_raw_close"]
            df[f"{t}_returns"] = raw.pct_change().fillna(0.0)
            df[f"{t}_sma"] = raw.rolling(10, min_periods=1).mean()
            df[f"{t}_vol"] = df[f"{t}_returns"].rolling(20, min_periods=1).std().fillna(0.0)
            df[f"{t}_close"] = (raw - raw.rolling(100, min_periods=1).mean()) / (raw.rolling(100, min_periods=1).std().replace(0,1e-8))
        self.data = df.fillna(0.0).reset_index(drop=True)

    def reset(self, start_idx: int = None):
        if start_idx is None:
            self.t = random.randint(self.window, self.length - 2)
        else:
            self.t = int(start_idx)
        self.portfolio_value = 1.0  # normalized portfolio value
        self.weights = np.ones(self.N) / self.N
        self.peak = self.portfolio_value
        self.max_drawdown = 0.0
        return self._get_obs()

    def _get_raw_prices(self, idx: int) -> np.ndarray:
        return np.array([float(self.df.loc[idx, f"{t}_raw_close"]) for t in self.assets], dtype=float)

    def _get_obs(self) -> np.ndarray:
        start = self.t - self.window
        rows = []
        for i in range(start, self.t):
            row = []
            for t in self.assets:
                row += [
                    float(self.data.loc[i, f"{t}_close"]),
                    float(self.data.loc[i, f"{t}_returns"]),
                    float(self.data.loc[i, f"{t}_sma"]),
                    float(self.data.loc[i, f"{t}_vol"]),
                ]
            rows.append(row)
        flat = np.array(rows).flatten()
        return flat.astype(np.float32)

    def step(self, action_pre: np.ndarray):
        # action_pre: pre-softmax logits; map to weights
        z = np.array(action_pre, dtype=float)
        exp = np.exp(z - np.max(z))
        new_weights = exp / (exp.sum() + 1e-12)

        prev_prices = self._get_raw_prices(self.t - 1)
        curr_prices = self._get_raw_prices(self.t)
        asset_rets = (curr_prices - prev_prices) / (prev_prices + 1e-12)

        # compute portfolio return using new_weights (assume rebalance at current step)
        portfolio_return = float(np.dot(new_weights, asset_rets))
        trade_cost = self.transaction_cost * float(np.sum(np.abs(new_weights - self.weights)))
        net_return = portfolio_return - trade_cost

        prev_pv = self.portfolio_value
        self.portfolio_value = self.portfolio_value * (1.0 + net_return)
        self.weights = new_weights

        # update drawdown
        if self.portfolio_value > self.peak:
            self.peak = self.portfolio_value
        dd = (self.peak - self.portfolio_value) / (self.peak + 1e-12)
        if dd > self.max_drawdown:
            self.max_drawdown = float(dd)

        reward = net_return  # step-level reward (can be modified by trainer's Lagrangian wrapper/loss)
        info = {
            "portfolio_value": float(self.portfolio_value),
            "max_drawdown": float(self.max_drawdown),
            "step_return": float(net_return),
        }

        self.t += 1
        done = (self.t >= self.length - 1)
        obs = self._get_obs() if not done else np.zeros_like(self._get_obs())
        return obs, float(reward), bool(done), info

    def render(self, mode="human"):
        print(f"t={self.t} PV={self.portfolio_value:.4f} Peak={self.peak:.4f} MaxDD={self.max_drawdown:.4f}")

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        np.random.seed(seed)
        torch.manual_seed(int(seed))
        return [seed]

# ---------------------------
# Networks: Actor & Critic
# ---------------------------
class MLPActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(last, action_dim)
        # log_std as a parameter (one per action dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(obs)
        mean = self.mean_head(x)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        return mean, std

class MLPCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)

# ---------------------------
# PPO with GAE & Lagrangian
# ---------------------------
class PPO_Lagrangian:
    def __init__(
        self,
        env: gym.Env,
        assets: List[str],
        clip_eps: float = 0.2,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam_gae: float = 0.95,
        rollout_steps: int = 2048,
        ppo_epochs: int = 10,
        minibatch_size: int = 64,
        lambda_lr: float = 1e-2,
        drawdown_target: float = 0.12,
        device: str = "cpu",
        seed: int = 1,
    ):
        self.env = env
        self.assets = assets
        self.N = len(assets)
        self.obs_dim = env.observation_space.shape[0]
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.lam_gae = lam_gae
        self.rollout_steps = rollout_steps
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.device = torch.device(device)
        self.lambda_lr = lambda_lr
        self.drawdown_target = drawdown_target

        # models
        self.actor = MLPActor(self.obs_dim, self.N).to(self.device)
        self.critic = MLPCritic(self.obs_dim).to(self.device)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)

        # Lagrange multiplier (scalar)
        self.lagrange_lambda = 0.0

        # seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def sample_action(self, obs_np: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        """Return softmax(weights), log_prob (of z), and z (pre-softmax) used for storage."""
        obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        mean, std = self.actor(obs)
        mean = mean.squeeze(0)
        std = std.squeeze(0)
        # sample z
        eps = torch.randn_like(mean)
        z = mean + std * eps
        # log_prob of z under Normal(mean, std)
        log_prob = -0.5 * (((z - mean) / (std + 1e-12)) ** 2 + 2 * torch.log(std + 1e-12) + math.log(2 * math.pi))
        logp = log_prob.sum().item()
        z_np = z.detach().cpu().numpy()
        # convert to weights
        exp = np.exp(z_np - np.max(z_np))
        weights = exp / (exp.sum() + 1e-12)
        return weights.astype(np.float32), float(logp), z_np

    def deterministic_action(self, obs_np: np.ndarray) -> np.ndarray:
        obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        mean, std = self.actor(obs)
        z = mean.squeeze(0).detach().cpu().numpy()
        exp = np.exp(z - np.max(z))
        weights = exp / (exp.sum() + 1e-12)
        return weights.astype(np.float32)

    def compute_gae(self, rewards, values, dones):
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        lastgaelam = 0
        for t in reversed(range(T)):
            if t == T - 1:
                nextnonterminal = 0.0 if dones[t] else 1.0
                nextvalue = 0.0
            else:
                nextnonterminal = 0.0 if dones[t + 1] else 1.0
                nextvalue = values[t + 1]
            delta = rewards[t] + self.gamma * nextvalue * nextnonterminal - values[t]
            lastgaelam = delta + self.gamma * self.lam_gae * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + values
        return advantages, returns

    def compute_max_drawdown(self, pv_series: List[float]) -> float:
        pv = np.array(pv_series, dtype=np.float32)
        peak = np.maximum.accumulate(pv)
        dd = (peak - pv) / (peak + 1e-12)
        return float(dd.max()) if len(dd) > 0 else 0.0

    def update_lagrange(self, max_dd: float):
        violation = max_dd - self.drawdown_target
        self.lagrange_lambda = max(0.0, self.lagrange_lambda + self.lambda_lr * violation)
        return self.lagrange_lambda, violation

    def train(self, total_updates: int = 200, log_interval: int = 1, save_path: str = None):
        obs = self.env.reset()
        episode_pvs = []
        episode_rewards = []
        global_step = 0

        # main loop on updates
        for update in range(1, total_updates + 1):
            # storage for rollout
            obs_buf = []
            z_buf = []
            logp_buf = []
            act_buf = []
            rew_buf = []
            val_buf = []
            done_buf = []
            pv_buf = []

            # collect rollout_steps transitions
            for step in range(self.rollout_steps):
                # obs is flat vector from env
                obs_flat = obs.astype(np.float32)
                weights, logp, z = self.sample_action(obs_flat)
                next_obs, reward, done, info = self.env.step(z)  # env expects pre-softmax logits; we pass z
                # store base info
                obs_buf.append(obs_flat)
                z_buf.append(z)            # store pre-softmax z (action in policy space)
                logp_buf.append(logp)
                act_buf.append(weights)    # already softmaxed weights (for debugging)
                rew_buf.append(float(reward))
                pv_buf.append(float(info.get("portfolio_value", np.nan)))
                done_buf.append(done)
                # critic value
                with torch.no_grad():
                    v = self.critic(torch.tensor(obs_flat, dtype=torch.float32, device=self.device).unsqueeze(0)).item()
                val_buf.append(v)

                obs = next_obs
                global_step += 1
                if done:
                    obs = self.env.reset()

            # convert buffers to arrays
            obs_arr = np.asarray(obs_buf, dtype=np.float32)
            z_arr = np.asarray(z_buf, dtype=np.float32)
            logp_arr = np.asarray(logp_buf, dtype=np.float32)
            rew_arr = np.asarray(rew_buf, dtype=np.float32)
            val_arr = np.asarray(val_buf, dtype=np.float32)
            done_arr = np.asarray(done_buf, dtype=np.bool_)

            # compute advantages and returns using GAE
            adv_arr, ret_arr = self.compute_gae(rew_arr, val_arr, done_arr)
            adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)

            # compute rollout max drawdown from pv_buf
            max_dd = self.compute_max_drawdown(pv_buf)
            lag_lambda, violation = self.update_lagrange(max_dd)

            # prepare tensors for training
            obs_tensor = torch.tensor(obs_arr, dtype=torch.float32, device=self.device)
            z_tensor = torch.tensor(z_arr, dtype=torch.float32, device=self.device)
            old_logp_tensor = torch.tensor(logp_arr, dtype=torch.float32, device=self.device)
            adv_tensor = torch.tensor(adv_arr, dtype=torch.float32, device=self.device)
            ret_tensor = torch.tensor(ret_arr, dtype=torch.float32, device=self.device)

            # PPO update: K epochs of minibatches
            batch_size = obs_tensor.size(0)
            indices = np.arange(batch_size)
            for epoch in range(self.ppo_epochs):
                np.random.shuffle(indices)
                for start in range(0, batch_size, self.minibatch_size):
                    mb_idx = indices[start:start + self.minibatch_size]
                    mb_obs = obs_tensor[mb_idx]
                    mb_z = z_tensor[mb_idx]
                    mb_old_logp = old_logp_tensor[mb_idx]
                    mb_adv = adv_tensor[mb_idx]
                    mb_ret = ret_tensor[mb_idx]

                    # forward new policy
                    mean, std = self.actor(mb_obs)
                    # sample deterministic z' = mean for policy evaluation (use reparameterization for log_prob)
                    # for PPO we need log_prob of the actually taken action z (stored in mb_z) under new policy
                    # compute log prob of mb_z under Normal(mean, std)
                    var = std ** 2 + 1e-12
                    logp_new = -0.5 * (((mb_z - mean) ** 2) / var + 2 * torch.log(std + 1e-12) + math.log(2 * math.pi))
                    logp_new = logp_new.sum(dim=-1)

                    ratio = torch.exp(logp_new - mb_old_logp)
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # value loss
                    value_pred = self.critic(mb_obs)
                    value_loss = ((mb_ret - value_pred) ** 2).mean()

                    # Lagrangian penalty: add lambda * violation (violation scalar) distributed to minibatch as constant
                    # this effectively shifts objective penalizing policies that produced the rollout violation
                    lag_penalty = float(lag_lambda * max(0.0, violation))
                    # scale penalty by minibatch fraction
                    lag_penalty_tensor = torch.tensor(lag_penalty * (len(mb_idx) / batch_size), dtype=torch.float32, device=self.device)

                    loss = policy_loss + 0.5 * value_loss + 0.01 * (-logp_new.mean()) + lag_penalty_tensor
                    # note: entropy bonus approximate via -logp mean (not exact), adjustable

                    self.optimizer.zero_grad()
                    loss.backward()
                    # gradient clipping (helpful)
                    torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), max_norm=0.5)
                    self.optimizer.step()

            # logging
            if update % log_interval == 0:
                avg_return = np.sum(rew_arr)
                print(f"[Update {update:04d}] steps={global_step} rollout_return={avg_return:.6f} max_dd={max_dd:.4f} lambda={self.lagrange_lambda:.6f}")

            # optionally save interim model
            if save_path and (update % 50 == 0):
                self.save(save_path)

        # final save
        if save_path:
            self.save(save_path)
        print("Training complete.")

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pth"))
        meta = {"lagrange_lambda": self.lagrange_lambda, "drawdown_target": self.drawdown_target}
        pd.Series(meta).to_json(os.path.join(path, "meta.json"))
        print(f"Saved models to {path}")

    def evaluate(self, env: gym.Env = None, deterministic: bool = True, render: bool = False):
        env = env or self.env
        obs = env.reset()
        done = False
        pv = [env.portfolio_value]
        while not done:
            if deterministic:
                act = self.deterministic_action(obs)
                # convert weights back to pre-softmax z for env.step: find logits that softmax->act
                # approximate by inverse softmax: log(act) (with smoothing)
                z = np.log(act + 1e-8)
            else:
                weights, logp, z = self.sample_action(obs)
            obs, reward, done, info = env.step(z)
            pv.append(info.get("portfolio_value", pv[-1]))
            if render:
                env.render()
        return pv, info

# ---------------------------
# Main runner
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "GOOG", "AMZN"])
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="2023-12-31")
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--rollout", type=int, default=2048)
    parser.add_argument("--updates", type=int, default=120)
    parser.add_argument("--ppo_epochs", type=int, default=8)
    parser.add_argument("--minibatch", type=int, default=64)
    parser.add_argument("--drawdown_target", type=float, default=0.12)
    parser.add_argument("--lambda_lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default="ppo_lagrangian_model")
    parser.add_argument("--use_synthetic", action="store_true", help="Use synthetic GBM data instead of yfinance")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    csv_path = "data/multi_asset.csv"
    if args.use_synthetic or not os.path.exists(csv_path):
        # generate synthetic data (GBM)
        print("Generating synthetic GBM data...")
        days = max(1200, args.rollout * 10)
        rng = np.random.RandomState(args.seed)
        df = pd.DataFrame()
        for t in args.tickers:
            mu = 0.0002
            sigma = 0.01
            s0 = 100.0 + rng.randn() * 5
            increments = rng.normal(loc=mu, scale=sigma, size=days)
            prices = s0 * np.exp(np.cumsum(increments))
            df[f"{t}_raw_close"] = prices
        for t in args.tickers:
            raw = df[f"{t}_raw_close"]
            mean = pd.Series(raw).rolling(100, min_periods=1).mean()
            std = pd.Series(raw).rolling(100, min_periods=1).std().replace(0,1e-8)
            df[f"{t}_close"] = (raw - mean) / std
            df[f"{t}_volume"] = 1.0
        df.reset_index(drop=True, inplace=True)
        df.to_csv(csv_path, index=False)
        print(f"Synthetic data saved to {csv_path} ({len(df)} rows)")
    else:
        df = pd.read_csv(csv_path)
        if df.empty:
            raise RuntimeError("Loaded CSV is empty. Remove / recreate data/multi_asset.csv")

    # if missing real CSV and not synthetic, download
    if not args.use_synthetic and not os.path.exists(csv_path):
        df = prepare_multi_asset_data(args.tickers, start=args.start, end=args.end, path=csv_path)

    # build env
    env = MultiAssetPortfolioEnv(df, assets=args.tickers, window=args.window)

    # trainer
    trainer = PPO_Lagrangian(
        env=env,
        assets=args.tickers,
        clip_eps=0.2,
        lr=3e-4,
        gamma=0.99,
        lam_gae=0.95,
        rollout_steps=args.rollout,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch,
        lambda_lr=args.lambda_lr,
        drawdown_target=args.drawdown_target,
        device="cpu",
        seed=args.seed,
    )

    # train
    print("Starting PPO (Lagrangian) training...")
    trainer.train(total_updates=args.updates, log_interval=1, save_path=args.save)

    # evaluate
    pv_series, final_info = trainer.evaluate(deterministic=True, render=False)
    print("Evaluation final info:", final_info)
    plt.plot(pv_series)
    plt.title("Portfolio value (evaluation)")
    plt.xlabel("Step")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
