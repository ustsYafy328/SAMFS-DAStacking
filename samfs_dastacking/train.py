import random
import time

import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from .config import DAStackingConfig, TARGET_COL
from .experts import fit_experts_with_oof
from .metrics import regression_metrics
from .models import Actor, Critic, OUNoise, ReplayBuffer, soft_update


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _inverse_y(scaler_y: MinMaxScaler, pred):
    return scaler_y.inverse_transform(np.asarray(pred).reshape(-1, 1)).ravel()


def _scale_columns(scaler_y: MinMaxScaler, values: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [scaler_y.transform(values[:, i].reshape(-1, 1)).ravel() for i in range(values.shape[1])]
    )


def _actor_weights(actor: Actor, states, device, batch_size: int = 512):
    actor.eval()
    if isinstance(states, np.ndarray):
        states = torch.from_numpy(states.astype(np.float32)).to(device)
    out = []
    with torch.no_grad():
        for i in range(0, states.shape[0], batch_size):
            out.append(actor(states[i : i + batch_size]).cpu().numpy())
    return np.vstack(out) if out else np.zeros((0, 0), dtype=np.float32)


class DAStackingRegressor:
    """DA-Stacking: SAM-FS feature subset + heterogeneous experts + attention DDPG gate."""

    def __init__(self, config: DAStackingConfig | None = None, verbose: bool = False):
        self.config = config or DAStackingConfig()
        self.verbose = verbose
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.feature_names_: list[str] | None = None
        self.experts_ = None
        self.actor_: Actor | None = None
        self.critic_: Critic | None = None
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_seconds_: float | None = None

    def fit(self, train_df, feature_names: list[str]):
        cfg = self.config
        set_seed(cfg.random_state)
        self.feature_names_ = list(feature_names)

        X_all = train_df[self.feature_names_].values
        y_all = train_df[TARGET_COL].values
        X_train, X_val, y_train, y_val = train_test_split(
            X_all, y_all, test_size=0.2, random_state=cfg.random_state
        )

        start = time.perf_counter()

        X_train_s = self.scaler_X.fit_transform(X_train)
        X_val_s = self.scaler_X.transform(X_val)
        y_train_s = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

        self.experts_, oof_s, train_full_s, val_s, _ = fit_experts_with_oof(
            X_train_s,
            y_train_s,
            X_val_s,
            X_val_s,
            random_state=cfg.random_state,
        )
        train_full = np.column_stack([_inverse_y(self.scaler_y, train_full_s[:, i]) for i in range(train_full_s.shape[1])])
        val_preds = np.column_stack([_inverse_y(self.scaler_y, val_s[:, i]) for i in range(val_s.shape[1])])
        oof_preds = np.column_stack([_inverse_y(self.scaler_y, oof_s[:, i]) for i in range(oof_s.shape[1])])

        states_train = np.hstack([X_train_s, oof_s]).astype(np.float32)
        states_train_full = np.hstack([X_train_s, train_full_s]).astype(np.float32)
        states_val = np.hstack([X_val_s, val_s]).astype(np.float32)

        self._fit_gate(
            states_train=states_train,
            states_train_full=states_train_full,
            train_full_preds=train_full,
            y_train=y_train,
            states_val=states_val,
            val_preds=val_preds,
            y_val=y_val,
            env_states=states_val,
            env_preds=val_preds,
            y_env=y_val,
            y_std=max(1e-6, float(np.std(y_all))),
        )
        self.train_seconds_ = time.perf_counter() - start
        return self

    def _fit_gate(
        self,
        states_train,
        states_train_full,
        train_full_preds,
        y_train,
        states_val,
        val_preds,
        y_val,
        env_states,
        env_preds,
        y_env,
        y_std,
    ):
        cfg = self.config
        device = self.device_
        feature_dim = len(self.feature_names_)
        expert_dim = env_preds.shape[1]

        actor = Actor(feature_dim, expert_dim, cfg.actor_hidden, cfg.dropout).to(device)
        critic = Critic(feature_dim, expert_dim, cfg.critic_hidden, cfg.dropout).to(device)
        actor_target = Actor(feature_dim, expert_dim, cfg.actor_hidden, cfg.dropout).to(device)
        critic_target = Critic(feature_dim, expert_dim, cfg.critic_hidden, cfg.dropout).to(device)
        actor_target.load_state_dict(actor.state_dict())
        critic_target.load_state_dict(critic.state_dict())

        actor_opt = optim.Adam(actor.parameters(), lr=cfg.actor_lr, weight_decay=8e-5)
        critic_opt = optim.Adam(critic.parameters(), lr=cfg.critic_lr, weight_decay=8e-5)
        criterion = torch.nn.SmoothL1Loss()
        buffer = ReplayBuffer(cfg.buffer_size, env_states.shape[1], expert_dim)
        noise = OUNoise(expert_dim)

        n_env = env_states.shape[0]
        best_metric = float("inf")
        best_actor_state = None
        no_improve = 0
        total_steps = 0
        env_idx = 0

        for i in range(min(cfg.start_steps, n_env)):
            self._add_transition(actor, buffer, noise, env_states, env_preds, y_env, i, y_std)

        while total_steps < cfg.max_steps:
            progress = total_steps / max(1, cfg.max_steps)
            noise.sigma = 0.4 * max(0.0, 1.0 - progress / 0.6) if progress < 0.6 else 0.0
            done = self._add_transition(
                actor, buffer, noise, env_states, env_preds, y_env, env_idx, y_std
            )
            if done:
                noise.reset()

            total_steps += 1
            env_idx = (env_idx + 1) % n_env

            if buffer.size >= cfg.batch_size and total_steps % cfg.update_every == 0:
                for _ in range(cfg.updates_per_step):
                    s, a, r, s2, d = buffer.sample(cfg.batch_size)
                    s = torch.from_numpy(s).to(device)
                    a = torch.from_numpy(a).to(device)
                    r = torch.from_numpy(r).to(device)
                    s2 = torch.from_numpy(s2).to(device)
                    d = torch.from_numpy(d).to(device)

                    with torch.no_grad():
                        a2 = actor_target(s2)
                        q2 = critic_target(s2, a2)
                        target_q = r + (1.0 - d) * cfg.gamma * q2

                    q = critic(s, a)
                    critic_loss = criterion(q, target_q)
                    critic_opt.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                    critic_opt.step()

                    a_pred = actor(s)
                    actor_loss = -critic(s, a_pred).mean()
                    entropy = -(a_pred * torch.log(a_pred + 1e-8)).sum(dim=1).mean()
                    actor_loss = actor_loss - cfg.ent_coeff * entropy
                    actor_opt.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                    actor_opt.step()

                    soft_update(actor_target, actor, cfg.tau)
                    soft_update(critic_target, critic, cfg.tau)

            if total_steps % cfg.eval_every == 0:
                val_metric = self._mae_from_gate(actor, states_val, val_preds, y_val)
                if self.verbose:
                    pct = int(total_steps / cfg.max_steps * 100)
                    print(f"DDPG training {pct}% | val_mae={val_metric:.6f}", end="\r", flush=True)
                if val_metric < best_metric - 1e-8:
                    best_metric = val_metric
                    no_improve = 0
                    best_actor_state = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
                else:
                    no_improve += 1
                    if no_improve >= cfg.patience_evals:
                        break

        if best_actor_state is not None:
            actor.load_state_dict(best_actor_state)
        if self.verbose:
            print()
        self.actor_ = actor
        self.critic_ = critic

    def _add_transition(self, actor, buffer, noise, states, preds, y, idx, y_std) -> bool:
        cfg = self.config
        device = self.device_
        next_idx = (idx + 1) % states.shape[0]
        done = float(next_idx == 0)
        with torch.no_grad():
            state_t = torch.from_numpy(states[idx : idx + 1]).to(device)
            action = actor(state_t).cpu().numpy()[0]
        action = np.clip(action + noise.sample(), 1e-8, None)
        action = action / action.sum()
        pred = float((action * preds[idx]).sum())
        mse_norm = ((pred - y[idx]) ** 2) / (y_std**2)
        entropy = -float(np.sum(action * np.log(action + 1e-8)))
        reward = -cfg.reward_lambda * mse_norm + cfg.ent_coeff * entropy
        buffer.add(states[idx], action, reward, states[next_idx], done)
        return bool(done)

    def _mae_from_gate(self, actor, states, preds, y_true):
        weights = _actor_weights(actor, states, self.device_)
        pred = (weights * preds).sum(axis=1)
        return float(np.mean(np.abs(pred - y_true)))

    def predict(self, df):
        if self.actor_ is None or self.experts_ is None or self.feature_names_ is None:
            raise RuntimeError("Model must be fitted before predict.")
        X = df[self.feature_names_].values
        X_s = self.scaler_X.transform(X)
        pred_s = np.column_stack([model.predict(X_s) for _, model in self.experts_])
        pred_orig = np.column_stack([_inverse_y(self.scaler_y, pred_s[:, i]) for i in range(pred_s.shape[1])])
        states = np.hstack([X_s, pred_s]).astype(np.float32)
        weights = _actor_weights(self.actor_, states, self.device_)
        return (weights * pred_orig).sum(axis=1), weights


def train_and_evaluate(train_df, test_df, feature_names, config: DAStackingConfig, verbose=False):
    model = DAStackingRegressor(config=config, verbose=verbose)
    model.fit(train_df, feature_names)
    pred_train, _ = model.predict(train_df)
    pred_test, weights_test = model.predict(test_df)
    metrics = {
        "train": regression_metrics(train_df[TARGET_COL].values, pred_train),
        "test": regression_metrics(test_df[TARGET_COL].values, pred_test),
        "train_seconds": model.train_seconds_,
        "feature_count": len(feature_names),
    }
    return metrics, model, weights_test

