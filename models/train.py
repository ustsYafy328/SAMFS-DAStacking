import os
import numpy as np
import pandas as pd
import random
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from tabpfn import TabPFNRegressor

# 引入抽离出来的模型和辅助组件
from model import ReplayBuffer, Actor, Critic, OUNoise, soft_update

def limit_rows(df: pd.DataFrame, max_rows=9500, threshold=10000, random_state=43):
    if len(df) > threshold:
        return df.sample(n=max_rows, random_state=random_state).reset_index(drop=True)
    return df

def train_model(train_df: pd.DataFrame, test_df: pd.DataFrame,
                random_state: int = 43,
                actor_hidden=(256, 160, 96),
                critic_hidden=(512, 320, 192),
                batch_size=128,
                max_steps=40000,
                gamma=0.99,
                tau=1e-3,
                actor_lr=1e-4,
                critic_lr=5e-4,
                buffer_size=20000,
                start_steps=2048,
                update_every=5,
                updates_per_step=12,
                eval_every=200,
                patience_evals=25,
                ent_coeff: float = 0.0045,        
                reward_lambda: float = 0.8,       
                verbose: bool = False):
    
    np.random.seed(random_state); random.seed(int(random_state)); torch.manual_seed(int(random_state))

    target_col = 'speed'  # 替换了 '挤压速度'
    drop_cols = []
    feature_names = [c for c in train_df.columns if c not in set(drop_cols + [target_col])]

    X_all = train_df[feature_names].values
    y_all = train_df[target_col].values
    X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=random_state)
    X_test = test_df[feature_names].values
    y_test = test_df[target_col].values

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

    y_std = max(1e-6, float(np.std(y_all)))

    base_learners = [
        ('tabpfn', TabPFNRegressor()),
        ('et', ExtraTreesRegressor(n_estimators=300, max_depth=24, max_features='sqrt',
                                   min_samples_leaf=4, random_state=random_state, n_jobs=-1)),
        ('rf', RandomForestRegressor(n_estimators=300, max_depth=24, max_features='sqrt',
                                     min_samples_leaf=4, bootstrap=True, random_state=random_state, n_jobs=-1)),
        ('xgb', xgb.XGBRegressor(n_estimators=300, max_depth=12, learning_rate=0.03,
                                 colsample_bytree=0.8, subsample=0.8,
                                 reg_alpha=0.5, reg_lambda=0.5, verbosity=0, random_state=random_state)),
    ]

    n_experts = len(base_learners)
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    oof_preds = np.zeros((X_train_scaled.shape[0], n_experts), dtype=float)
    
    for i, (name, estimator) in enumerate(base_learners):
        preds_scaled = cross_val_predict(estimator, X_train_scaled, y_train_scaled, cv=cv, method='predict', n_jobs=1)
        preds_orig = scaler_y.inverse_transform(preds_scaled.reshape(-1,1)).ravel()
        oof_preds[:, i] = preds_orig

    fitted_experts = []
    for name, estimator in base_learners:
        est = estimator
        est.fit(X_train_scaled, y_train_scaled)
        fitted_experts.append((name, est))

    base_train_preds_full = np.column_stack([
        scaler_y.inverse_transform(est.predict(X_train_scaled).reshape(-1,1)).ravel() for _, est in fitted_experts
    ]) if X_train_scaled.shape[0] > 0 else np.zeros((0, n_experts))
    
    base_train_preds_scaled_for_gating = np.column_stack([
        scaler_y.transform(base_train_preds_full[:, i].reshape(-1,1)).ravel() for i in range(n_experts)
    ]) if base_train_preds_full.size else np.zeros((X_train_scaled.shape[0], n_experts))
    
    oof_preds_scaled_for_input = np.column_stack([
        scaler_y.transform(oof_preds[:, i].reshape(-1,1)).ravel() for i in range(n_experts)
    ])
    states_train = np.hstack([X_train_scaled, oof_preds_scaled_for_input]).astype(np.float32)
    experts_train_orig = oof_preds.astype(np.float32)

    states_train_full = np.hstack([X_train_scaled, base_train_preds_scaled_for_gating]).astype(np.float32) if base_train_preds_scaled_for_gating.size else states_train

    experts_val_preds = np.column_stack([
        scaler_y.inverse_transform(est.predict(X_val_scaled).reshape(-1,1)).ravel() for _, est in fitted_experts
    ]) if X_val_scaled.shape[0] > 0 else np.zeros((0, n_experts))
    
    experts_test_preds = np.column_stack([
        scaler_y.inverse_transform(est.predict(X_test_scaled).reshape(-1,1)).ravel() for _, est in fitted_experts
    ]) if X_test_scaled.shape[0] > 0 else np.zeros((0, n_experts))

    experts_val_preds_scaled = np.column_stack([
        scaler_y.transform(experts_val_preds[:, i].reshape(-1,1)).ravel() for i in range(n_experts)
    ]) if experts_val_preds.size else np.zeros((X_val_scaled.shape[0], n_experts))
    states_val = np.hstack([X_val_scaled, experts_val_preds_scaled]).astype(np.float32) if experts_val_preds_scaled.size else np.zeros((0, states_train.shape[1]), dtype=np.float32)

    experts_test_preds_scaled = np.column_stack([
        scaler_y.transform(experts_test_preds[:, i].reshape(-1,1)).ravel() for i in range(n_experts)
    ]) if experts_test_preds.size else np.zeros((X_test_scaled.shape[0], n_experts))
    states_test = np.hstack([X_test_scaled, experts_test_preds_scaled]).astype(np.float32)

    if states_val.shape[0] > 0:
        states_env = states_val
        experts_env = experts_val_preds.astype(np.float32)
        y_env = y_val
    else:
        states_env = states_train
        experts_env = experts_train_orig
        y_env = y_train

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = states_train.shape[1]
    action_dim = n_experts

    states_train_full_t = torch.from_numpy(states_train_full.astype(np.float32)).to(device)

    actor = Actor(state_dim, action_dim, hidden=actor_hidden, dropout=0.28).to(device)
    critic = Critic(state_dim, action_dim, hidden=critic_hidden, dropout=0.28).to(device)
    actor_target = Actor(state_dim, action_dim, hidden=actor_hidden, dropout=0.28).to(device)
    critic_target = Critic(state_dim, action_dim, hidden=critic_hidden, dropout=0.28).to(device)
    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())

    actor_opt = optim.Adam(actor.parameters(), lr=actor_lr, weight_decay=8e-5)
    critic_opt = optim.Adam(critic.parameters(), lr=critic_lr, weight_decay=8e-5)
    critic_criterion = nn.SmoothL1Loss()

    states_train_t = torch.from_numpy(states_train.astype(np.float32)).to(device)
    states_val_t = torch.from_numpy(states_val.astype(np.float32)).to(device) if states_val.size else torch.empty((0, state_dim), device=device)
    states_test_t = torch.from_numpy(states_test.astype(np.float32)).to(device) if states_test.size else torch.empty((0, state_dim), device=device)

    def actor_weights(gm, X_input, batch_size_infer=256):
        gm.eval()
        if X_input is None:
            return np.zeros((0, n_experts))
        if isinstance(X_input, torch.Tensor):
            X = X_input
        else:
            if getattr(X_input, "size", None) == 0 or (isinstance(X_input, np.ndarray) and X_input.size == 0):
                return np.zeros((0, n_experts))
            X = torch.from_numpy(np.asarray(X_input, dtype=np.float32)).to(device)
        out = []
        with torch.no_grad():
            for i in range(0, X.shape[0], batch_size_infer):
                xb = X[i:i+batch_size_infer]
                w = gm(xb).cpu().numpy()
                out.append(w)
        return np.vstack(out) if out else np.zeros((0, n_experts))

    total_steps = 0
    N_env = states_env.shape[0]
    best_val_mae = float('inf')
    no_improve_evals = 0
    best_actor_state = None

    max_train_iters = max_steps
    rb = ReplayBuffer(buffer_size, state_dim, action_dim)
    ou_noise = OUNoise(action_dim)
    ou_noise.reset()

    for i in range(min(start_steps, N_env)):
        idx = i % N_env
        next_idx = (idx + 1) % N_env
        done = 1.0 if next_idx == 0 else 0.0
        s_t = states_env[idx]
        s2 = states_env[next_idx]
        experts_orig = experts_env[idx]
        y_true = y_env[idx]
        with torch.no_grad():
            a_det = actor(torch.from_numpy(s_t[None, :]).to(device)).cpu().numpy()[0]
        noise_vec = ou_noise.sample()
        a_t = a_det + noise_vec
        a_t = np.clip(a_t, 1e-8, None)
        a_t = a_t / a_t.sum()
        weighted = (a_t * experts_orig).sum()
        err = weighted - y_true
        mse_norm = (err ** 2) / (y_std ** 2)
        r = - float(reward_lambda) * mse_norm
        rb.add(s_t, a_t, r, s2, done)

    it = 0
    env_idx = 0
    while total_steps < max_train_iters:
        s_t = states_env[env_idx]
        experts_orig = experts_env[env_idx]
        y_true = y_env[env_idx]
        next_idx = (env_idx + 1) % N_env
        s2 = states_env[next_idx]
        done = 1.0 if next_idx == 0 else 0.0

        with torch.no_grad():
            a_det = actor(torch.from_numpy(s_t[None, :]).to(device)).cpu().numpy()[0]

        progress = total_steps / max_train_iters
        if progress < 0.6:
            coef = 1.0 - progress / 0.6
            base_sigma = 0.4 * coef
        else:
            base_sigma = 0.0
        ou_noise.sigma = base_sigma
        noise_vec = ou_noise.sample()
        a_explore = a_det + noise_vec
        a_explore = np.clip(a_explore, 1e-8, None)
        a_explore = a_explore / a_explore.sum()

        weighted = (a_explore * experts_orig).sum()
        err = weighted - y_true
        mse_norm = (err ** 2) / (y_std ** 2)
        r = - float(reward_lambda) * mse_norm
        rb.add(s_t, a_explore, r, s2, done)

        if done:
            ou_noise.reset()

        total_steps += 1
        it += 1
        env_idx = next_idx

        if rb.size >= batch_size and total_steps % update_every == 0:
            for _ in range(updates_per_step):
                s_b, a_b, r_b, s2_b, d_b = rb.sample(batch_size)
                s_b = torch.from_numpy(s_b).to(device)
                a_b = torch.from_numpy(a_b).to(device)
                r_b = torch.from_numpy(r_b).to(device)
                s2_b = torch.from_numpy(s2_b).to(device)
                d_b_t = torch.from_numpy(d_b).to(device).float()
                with torch.no_grad():
                    a2 = actor_target(s2_b)
                    q2 = critic_target(s2_b, a2)
                    y_q = r_b + (1.0 - d_b_t) * gamma * q2
                q = critic(s_b, a_b)
                critic_loss = critic_criterion(q, y_q)
                critic_opt.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
                critic_opt.step()
                
                actor_opt.zero_grad()
                a_pred = actor(s_b)
                q_val_for_actor = critic(s_b, a_pred)
                actor_loss = -q_val_for_actor.mean()
                eps = 1e-8
                entropy = - (a_pred * torch.log(a_pred + eps)).sum(dim=1).mean()
                actor_loss = actor_loss - float(ent_coeff) * entropy
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
                actor_opt.step()
                
                soft_update(critic_target, critic, tau)
                soft_update(actor_target, actor, tau)

        if total_steps % eval_every == 0:
            with torch.no_grad():
                w_train_now = actor_weights(actor, states_train_full_t)
                if w_train_now.size:
                    pred_train_now = (w_train_now * base_train_preds_full).sum(axis=1)
                    tr_mae = float(mean_absolute_error(y_train, pred_train_now))
                else:
                    tr_mae = float('nan')

                if states_val_t.shape[0] > 0:
                    w_val_now = actor_weights(actor, states_val_t)
                    if w_val_now.size:
                        pred_val_now = (w_val_now * experts_val_preds).sum(axis=1)
                        val_mae = float(mean_absolute_error(y_val, pred_val_now))
                    else:
                        val_mae = float('nan')
                else:
                    val_mae = float('nan')

            if verbose:
                progress = total_steps / max_train_iters
                pct = int(progress * 100)
                print(f"{pct}%",round(pct*50*0.01)*">",end="\r", flush=True)

            metric = val_mae if np.isfinite(val_mae) else tr_mae
            if np.isfinite(metric):
                if metric < best_val_mae - 1e-8:
                    best_val_mae = metric
                    no_improve_evals = 0
                    best_actor_state = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
                else:
                    no_improve_evals += 1
                    if no_improve_evals >= patience_evals:
                        if verbose:
                            print()
                            print(f"Early stop triggered at step {total_steps}, best_val_mae={best_val_mae:.6f}")
                        break

    if best_actor_state is not None:
        if verbose: print()
        actor.load_state_dict(best_actor_state)

    w_train = actor_weights(actor, states_train_full_t)
    w_val = actor_weights(actor, states_val_t) if states_val_t.shape[0] > 0 else np.zeros((0, n_experts))
    w_test = actor_weights(actor, states_test_t)

    pred_train = (w_train * base_train_preds_full).sum(axis=1) if w_train.size else None
    pred_val = (w_val * experts_val_preds).sum(axis=1) if w_val.size else None
    pred_test = (w_test * experts_test_preds).sum(axis=1) if w_test.size else None

    if w_test.size:
        n_show = min(5, states_test.shape[0])
        idxs = np.random.choice(states_test.shape[0], size=n_show, replace=False)
        print("===== 随机测试集状态-动作（5组）=====")
        for idx in idxs:
            st = states_test[idx]
            act = w_test[idx]
            print(f"idx={idx} | state[:8]={np.round(st[:8], 4)} | action={np.round(act, 4)} | sum={act.sum():.4f}")

    def safe_mape(y_true, y_pred, eps: float = 1e-8):
        if y_pred is None or y_true is None or len(y_true) == 0: return None
        yt = np.asarray(y_true, dtype=np.float64)
        yp = np.asarray(y_pred, dtype=np.float64)
        denom = np.maximum(np.abs(yt), eps)
        return float(np.mean(np.abs((yp - yt) / denom)) * 100.0)

    def safe_metrics(y_true, y_pred):
        if y_pred is None or y_true is None or len(y_true) == 0:
            return None, None, None, None, None
        mse = float(mean_squared_error(y_true, y_pred))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        mape = safe_mape(y_true, y_pred)
        return mse, mae, r2, rmse, mape

    train_mse, train_mae, train_r2, train_rmse, train_mape = safe_metrics(y_train, pred_train)
    val_mse, val_mae, val_r2, val_rmse, val_mape = safe_metrics(y_val, pred_val)
    test_mse, test_mae, test_r2, test_rmse, test_mape = safe_metrics(y_test, pred_test)

    metrics = {
        "train_mse": train_mse, "train_mae": train_mae, "train_r2": train_r2, "train_rmse": train_rmse, "train_mape": train_mape,
        "val_mse": val_mse, "val_mae": val_mae, "val_r2": val_r2, "val_rmse": val_rmse, "val_mape": val_mape,
        "test_mse": test_mse, "test_mae": test_mae, "test_r2": test_r2, "test_rmse": test_rmse, "test_mape": test_mape,
        "feature_count": len(feature_names)
    }

    model_dict = {
        'experts': fitted_experts,
        'ddpg_actor': actor,
        'ddpg_critic': critic,
        'n_experts': n_experts
    }
    scalers = {'scaler_X': scaler_X, 'scaler_y': scaler_y, 'feature_names': feature_names}
    return metrics, model_dict, scalers

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    split_dir = os.path.join(base_dir, 'split')
    train_csv = os.path.join(split_dir, 'train.csv')
    test_csv = os.path.join(split_dir, 'test.csv')
    if not (os.path.exists(train_csv) and os.path.exists(test_csv)):
        raise FileNotFoundError(f"预划分的 split 文件不存在: {split_dir}")

    feature = [
        "Production", "PL", "BL", "MW", "DevProduction", "DevBL",
        "BLLow", "OtherST", "BLHigh", "Mould1", "BTemp",
        "SpST", "SmST", "DevBTemp", "Mould16", "ProductionHigh",
        "Alloy", "OxST", "Mould6", "BLMed", "ProductionMed", "BTempLow", "BTempHigh",
        "Mould2", "Num", "FlST", "ProductionLow", "BTempMed", "MTemp", "Diameter"
    ]
    
    train_df_all = pd.read_csv(train_csv)
    test_df_all = pd.read_csv(test_csv)

    def fmt(x): return f"{x:.4f}" if x is not None else "N/A"

    for n in range(20, len(feature) + 1):
        selected_feature = feature[:n]
        # target列已被替换为'speed'
        train_df = train_df_all[selected_feature + ['speed']]
        test_df = test_df_all[selected_feature + ['speed']]

        print(f"train by n={n},")
        metrics, model, scalers = train_model(train_df, test_df, random_state=43, verbose=True)

        print("===== 训练/测试评估结果 =====")
        print(f"Train MAE: {fmt(metrics.get('train_mae'))}, Train MAPE: {fmt(metrics.get('train_mape'))}, Train R²: {fmt(metrics.get('train_r2'))}, Train RMSE: {fmt(metrics.get('train_rmse'))}")
        print(f"Test  MAE: {fmt(metrics.get('test_mae'))}, Test  MAPE: {fmt(metrics.get('test_mape'))}, Test  R²: {fmt(metrics.get('test_r2'))}, Test  RMSE: {fmt(metrics.get('test_rmse'))}")
        print()

    print("已在内存中返回训练好的基学习器和 DDPG 门控网络（未保存到磁盘）。")