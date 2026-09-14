import os
from concurrent.futures import ProcessPoolExecutor
import pickle
import numpy as np
import pandas as pd
from scipy.stats import trim_mean
from scipy.ndimage import uniform_filter1d
from sklearn.ensemble import RandomForestRegressor

from PyALE import ale
from my_fanova import fANOVA
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

import logging
logging.disable(logging.INFO)


RET_LEN = 500
RET_NAMES = [f"ret_{i}" for i in range(RET_LEN)]
LEN_NAMES = [f"len_{i}" for i in range(RET_LEN)]
LOSS_NAMES = [f"loss_{i}" for i in range(RET_LEN)]
QVAL_NAMES = [f"qval_{i}" for i in range(RET_LEN)]
ORIG_STAT_NAMES = RET_NAMES + LEN_NAMES + LOSS_NAMES + QVAL_NAMES
STAT_NAMES = RET_NAMES + LEN_NAMES + LOSS_NAMES + QVAL_NAMES
DATA_DIR = "data/raw"

CONFIG_COLS = [
    'lr', 'gamma', 'target_net_freq', 
    'end_e', 'explore_frac', 'buffer_size', 'batch_size'
]


ENVS = ["acrobot", "cartpole", "cliffwalking", "frozenlake", "mountaincar"]


def load(env_type, step_name):
    filepath = f"data/analysis/{env_type}/{step_name}.pkl"
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    raise ValueError(f"{filepath=} Does Not Exist")


def run_or_load(env_type, step_name, func, *args, force=False, **kwargs):
    filepath = f"data/analysis/{env_type}/{step_name}.pkl"
    if os.path.exists(filepath) and not force:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    result = func(*args, **kwargs)
    with open(filepath, 'wb') as f:
        pickle.dump(result, f)
    return result


def get_data(env_type):
    filenames = os.listdir(f"{DATA_DIR}/{env_type}")
    dfs = []
    for filename in filenames:
        if "feather" in filename:
            df = pd.read_feather(f"{DATA_DIR}/{env_type}/{filename}")
            params = filename.split("_")[0].split("-")[1:]
            for param in params:
                p_name, p_val = param.split("=")
                if p_name == "buf":
                    p_name = "buffer_size"
                if p_name == "bch":
                    p_name = "batch_size"
                if p_name == "freq":
                    p_name = "train_freq"
                df[p_name] = int(p_val)
                
            df = df.drop(["agent_id", "tau"], axis=1)

            dfs.append(df)

    df = pd.concat(dfs, axis=0)
    return df


def aggregate_data(env_type, force=False):
    df = get_data(env_type)
    df_agents = run_or_load(env_type, 'agent_metrics', compute_agent_metrics, df, force=force)
    df_configs = run_or_load(env_type, 'config_metrics', aggregate_configs, df_agents, force=force)
    return df_configs


def compute_agent_metrics(df, success_percentile=.9, window_kernel=1):
    R = df[RET_NAMES].values.astype(float)
    RET_LEN = R.shape[1]
    
    W = uniform_filter1d(R, size=2*window_kernel+1, axis=1, mode='nearest')
    W_min = W.min()
    W_max = W.max()

    W = (W - W_min) / (W_max - W_min)

    auc = np.mean(W, axis=1)
    best = np.max(W, axis=1)
    
    success_score = success_percentile
    success = (best >= success_score).astype(int)
    
    reached_mask = W >= success_score
    time_to_success_raw = np.argmax(reached_mask, axis=1)
    time_to_success_raw[~np.any(reached_mask, axis=1)] = RET_LEN
    time_to_success = (RET_LEN - time_to_success_raw) / RET_LEN

    after_success_mask = np.arange(W.shape[1]) >= time_to_success_raw[:, None]
    after_success_cnt = np.maximum(np.sum(after_success_mask, axis=1), 1)
    assert np.all(after_success_cnt[time_to_success_raw == RET_LEN] == 1)
    R_zeroed = W * after_success_mask
    after_success_sum = np.sum(R_zeroed, axis=1)
    after_success_mean = after_success_sum / after_success_cnt

    res_df = df[CONFIG_COLS + ['seed']].copy()
    res_df['auc'] = auc
    res_df['success'] = success
    res_df['time_to_success_raw'] = time_to_success_raw
    res_df['time_to_success'] = time_to_success

    res_df['auc_after_success'] = after_success_mean
    
    return res_df


def aggregate_configs(df_agents: pd.DataFrame):
    def iqm(x):
        return trim_mean(x, proportiontocut=0.25)

    agg_funcs = {
        "auc": ("auc_iqm", iqm),
        "success": ("success_rate", np.mean),
        "time_to_success": ("time_to_success_iqm", iqm),
    }

    df_agg = (
        df_agents
        .groupby(CONFIG_COLS)
        .agg({col: func for col, (_, func) in agg_funcs.items()})
        .rename(columns={col: name for col, (name, _) in agg_funcs.items()})
        .reset_index()
    )

    return df_agg


def build_features(df_configs):
    eps_clip = 1e-6

    df_trans = pd.DataFrame()
    df_trans['log10_lr'] = np.log10(df_configs['lr'])
    
    gamma_clipped = np.clip(df_configs['gamma'], a_min=0.0, a_max=1.0 - eps_clip)
    df_trans['gamma_effective_horizon'] = np.log10(1 / (1 - gamma_clipped))
    
    df_trans['log10_target_freq'] = np.log10(df_configs['target_net_freq'])
    df_trans['log10_buffer'] = np.log10(df_configs['buffer_size'])
    df_trans['log10_batch'] = np.log10(df_configs['batch_size'])
    df_trans['explore'] = df_configs['explore_frac']
    df_trans['log10_end_e'] = np.log10(df_configs['end_e'])
    
    return df_trans


def evaluate_surrogate(X, y):
    if len(X) < 5:
        print(f"Not enough data for target (n={len(X)}). Skipping.")
        return None

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


def run_fanova(X_df, y_target, n_trees=30):
    num_seeds = 10
    X_df = X_df.astype(np.float64).round(6)
    cols = np.sort(X_df.columns)
    X_df = X_df[cols]
    
    nunique = X_df.nunique()
    constant_cols = nunique[nunique <= 1].index
    if len(constant_cols) > 0:
        X_df = X_df.drop(columns=constant_cols)
        
    cs = CS.ConfigurationSpace()
    for col in cols:
        if col in constant_cols:
            continue
            
        col_min = float(X_df[col].min())
        col_max = float(X_df[col].max())
        
        margin = abs(col_max - col_min) * 0.05
        if margin == 0: 
            margin = 0.1
            
        cs.add(
            CSH.UniformFloatHyperparameter(
                name=col,
                lower=col_min - margin,
                upper=col_max + margin
            )
        )

    y_np = y_target.values.astype(np.float64)
    f_anova = fANOVA(X=X_df, Y=y_np, config_space=cs, n_trees=n_trees, 
                     min_samples_split=num_seeds * 2, min_samples_leaf=num_seeds,
                     max_depth=8, seed=314)
    
    interactions = {}
    for i, col in enumerate(X_df.columns):
        res = f_anova.quantify_importance([i])
        interactions[col] = res[(i,)]['individual importance']
        
    for i in range(len(X_df.columns)):
        for j in range(i + 1, len(X_df.columns)):
            res = f_anova.quantify_importance([i, j])
            interactions[f"{X_df.columns[i]} x {X_df.columns[j]}"] = res[(i, j)]['individual importance']
            
    df_inter = pd.DataFrame([interactions])
    return df_inter


def get_ale1d_data(model, X_df, features) -> pd.DataFrame:
    ale_dfs = []
    for feature in features:
        if feature not in X_df.columns:
            continue
        ale_df = ale(X=X_df, model=model, feature=[feature], grid_size=50, include_CI=False, plot=False)
        ale_df = ale_df.reset_index()
        ale_df = ale_df.rename(columns={feature: "feature_val"})
        if "feature" in ale_df.columns:
            raise ValueError("ALE output unexpectedly already contains a 'feature' column.")
        ale_df["feature"] = feature
        ale_dfs.append(ale_df)
    return pd.concat(ale_dfs, axis=0, ignore_index=True)


def run_env_analysis(env_type):
    targets = ["success_rate", "time_to_success_iqm", "auc_iqm"]
    transformed_feature_names = ["log10_lr", "gamma_effective_horizon", "log10_target_freq", "log10_buffer", "log10_batch", "explore", "log10_end_e"]
    os.makedirs(f"data/analysis/{env_type}", exist_ok=True)

    print(f"Running analysis on {env_type}")

    df_configs = load(env_type, 'config_metrics')

    features_df = build_features(df_configs)
    
    ale_dfs = []
    fanova_dfs = []
        
    for target in targets:
        print(f"\tAnalyzing {target=}")
        filename = f"data/analysis/{env_type}/{target}_model.pkl"

        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                model = pickle.load(f)
        else:
            model = evaluate_surrogate(
                features_df, 
                df_configs[target].values.copy()
            )

            if model is None:
                continue
                
            with open(filename, 'wb') as f:
                pickle.dump(model, f)

        print(f"\t\tRetrieving ALE data")
        ale_df = get_ale1d_data(model, features_df, transformed_feature_names)
        ale_df["env"] = env_type
        ale_df["target"] = target
        ale_dfs.append(ale_df)

        print(f"\t\tRetrieving fANOVA data")
        df_inter = run_fanova(features_df, df_configs[target])
        df_inter["env"] = env_type
        df_inter["target"] = target
        fanova_dfs.append(df_inter)

    if ale_dfs:
        ale_df_all = pd.concat(ale_dfs, axis=0, ignore_index=True)
        ale_df_all.to_feather(f"data/analysis/ale_{env_type}.feather")
        
    if fanova_dfs:
        fanova_df_all = pd.concat(fanova_dfs, axis=0, ignore_index=True)
        fanova_df_all.to_feather(f"data/analysis/fanova_{env_type}.feather")


def aggregate_env_analyses_dfs():
    dfs = []
    for env_type in ENVS:
        dfs.append(pd.read_feather(f"data/analysis/ale_{env_type}.feather"))
    pd.concat(dfs, axis=0).to_feather(f"data/analysis/ale.feather")

    dfs = []
    for env_type in ENVS:
        dfs.append(pd.read_feather(f"data/analysis/fanova_{env_type}.feather"))
    pd.concat(dfs, axis=0).to_feather(f"data/analysis/fanova.feather")


def main():
    for env in ENVS:
        os.makedirs(f"data/analysis/{env}", exist_ok=True)
        print(f"Computing {env}'s agent and config metrics")
        aggregate_data(env)

    with ProcessPoolExecutor() as executor:
        executor.map(run_env_analysis, ENVS)

    aggregate_env_analyses_dfs()


if __name__ == "__main__":
    main()