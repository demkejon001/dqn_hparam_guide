import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pingouin as pg
from scipy.stats import trim_mean

from aggregate_results import build_features


def load(env_type, step_name, ):
    filepath = f"data/analysis/{env_type}/{step_name}.pkl"
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    raise ValueError(f"{filepath=} Does Not Exist")


def plot_ale(trans_feats_to_feats, trans_feats_to_pretty_names, targets_to_pretty_names, show=False) -> pd.DataFrame:
    # Helps with formatting feat_to_vals
    config_df = load("acrobot", "config_metrics")
    config_df["target_net_freq"] = config_df["target_net_freq"].astype(int)

    feat_to_vals = dict()
    for feat in trans_feats_to_feats:
        feat_to_vals[feat] = np.sort(config_df[trans_feats_to_feats[feat]].unique())
    del config_df

    ale_df = pd.read_feather("data/analysis/ale.feather")
    envs = ale_df["env"].unique()

    fig_width = 1.5 * len(trans_feats_to_pretty_names)
    fig_height = 2.0 * len(targets_to_pretty_names)
    
    fig, axes = plt.subplots(
        len(targets_to_pretty_names), 
        len(trans_feats_to_pretty_names), 
        sharey="all", 
        figsize=(fig_width, fig_height), 
        constrained_layout=True
    )
    
    fig.supylabel('Effect', fontweight='bold')

    for row, target in enumerate(targets_to_pretty_names):
        target_df = ale_df[ale_df["target"] == target]
        
        for col, feature in enumerate(trans_feats_to_pretty_names):
            ax = axes[row, col]
            for env in envs:
                df = target_df[(target_df["env"] == env) & (target_df["feature"] == feature)]
                x = df["feature_val"]
                assert np.all(x == np.sort(x))
                y = df["eff"]
                
                ax.plot(np.arange(len(x)), y, label=env)
                
            if row >= len(targets_to_pretty_names) - 1:
                if len(x) > 4:
                    xticks = np.arange(0, len(x), 2)
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(feat_to_vals[feature][xticks], rotation=45, ha='right')
                else:
                    ax.set_xticks(np.arange(len(x)))
                    ax.set_xticklabels(feat_to_vals[feature], rotation=45, ha='right')
            else:
                ax.set_xticks([])

        axes[row, 0].set_ylabel(targets_to_pretty_names[target])

    for col, feature in enumerate(trans_feats_to_pretty_names):
        axes[-1, col].set_xlabel(trans_feats_to_pretty_names[feature])

    fig.align_xlabels(axes[-1, :])
    
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper right", ncol=len(labels))

    if show:
        plt.show()
    else:
        plt.savefig(f"data/images/ale", bbox_inches="tight")
        
    plt.close(fig)


def plot_fanova(
        name_to_pretty_names,
        targets_to_pretty_names,
        show=False,
    ):
    df = pd.read_feather(f"data/analysis/fanova.feather")
    variables = list(name_to_pretty_names.keys())
    variables_pretty = [name_to_pretty_names[c] for c in variables]

    envs = np.sort(df["env"].unique())

    shape = (len(targets_to_pretty_names), len(envs))
    fig, axes = plt.subplots(*shape, constrained_layout=True, figsize=(11, 6))
    np.reshape(axes, shape)
    for ax_col, env in enumerate(envs):
        for ax_row, target in enumerate(targets_to_pretty_names):
            df_filtered = df[(df["target"] == target) & (df["env"] == env)]
            df_filtered = df_filtered.drop(["env", "target"], axis=1)
            row = df_filtered.iloc[0]

            cov_matrix = pd.DataFrame(
                np.zeros((len(variables), len(variables))),
                index=variables,
                columns=variables
            )

            for var in variables:
                cov_matrix.loc[var, var] = row[var]

            for col in row.index:
                if " x " in col:
                    v1, v2 = col.split(" x ")
                    cov_matrix.loc[v1, v2] = row[col]
                    cov_matrix.loc[v2, v1] = row[col]

            sns.heatmap(cov_matrix, ax=axes[ax_row, ax_col], vmin=0, vmax=.6, cmap="viridis", xticklabels=variables_pretty, yticklabels=variables_pretty, cbar=False)
            if ax_row < (len(targets_to_pretty_names) - 1):
                axes[ax_row, ax_col].set_xticks([])
            if ax_col > 0:
                axes[ax_row, ax_col].set_yticks([])

    for row, target in enumerate(targets_to_pretty_names.values()):
        axes[row, 0].set_ylabel(target)
    for col, val in enumerate(envs):
        axes[0, col].set_title(val, fontsize=12)
    mappable = axes[0, 0].collections[0]
    fig.colorbar(mappable, ax=axes, label="Importance")

    if show:
        plt.show()
    else:
        plt.savefig(f"data/images/fanova", bbox_inches="tight")
    plt.close(fig)


def plot_strat_hist(targets, envs, feats, targets_to_pretty_names, feats_to_pretty_names, show=False):
    for target in targets:

        fig, axes = plt.subplots(len(envs), len(feats), figsize=(10, 8), constrained_layout=True)
        for env_idx, env in enumerate(envs):

            config_df = load(env, "config_metrics")

            config_df["target_net_freq"] = config_df["target_net_freq"].astype(int)

            n_bins = 10
            bin_vals = np.round(np.linspace(0, 1, n_bins + 1), 3)
            feat_to_binned_hist = dict()

            for feat in feats:
                uniq_feat_vals = config_df[feat].unique()
                binned_hist = np.zeros((len(uniq_feat_vals), n_bins), dtype=np.float64)
                binned_hist = pd.DataFrame(binned_hist, columns=bin_vals[1:], index=uniq_feat_vals)
                feat_to_binned_hist[feat] = binned_hist

            for i in range(n_bins):
                if i == 0:
                    df_val = config_df[(config_df[target] >= bin_vals[i]) & (config_df[target] <= bin_vals[i+1])]
                else:
                    df_val = config_df[(config_df[target] >  bin_vals[i]) & (config_df[target] <= bin_vals[i+1])]

                if len(df_val) < 500:
                    continue

                for feat in feats:
                    binned_hist = feat_to_binned_hist[feat]
                    
                    cnts = df_val[feat].value_counts()
                    normalized_cnts = (cnts / cnts.sum()).reindex(binned_hist.index, fill_value=0)
                    binned_hist[bin_vals[i+1]] += normalized_cnts

            for i, feat in enumerate(feats):
                binned_hist = feat_to_binned_hist[feat]
                ax = axes[env_idx, i]
                plot_df = binned_hist.sort_index(axis=0).sort_index(axis=1)
                sns.heatmap(plot_df.T, ax=ax, vmin=0, vmax=1.0, cbar=False, cmap="viridis")
                if i > 0:
                    ax.set_yticks([])
                else:
                    ax.tick_params(axis='y', labelrotation=0)
                    for label in ax.get_xticklabels():
                        label.set_rotation(0)
                if env_idx < len(envs) - 1:
                    ax.set_xticks([])
                ax.tick_params(axis='x', labelrotation=45)

            axes[env_idx, 0].set_ylabel(env)

        for i, feat in enumerate(feats):
            axes[-1, i].set_xlabel(feats_to_pretty_names[feat])
        fig.align_xlabels(axes[-1, :])
        mappable = axes[0, 0].collections[0]
        fig.colorbar(mappable, ax=axes, label="Density")

        fig.supylabel("Performance Strata")

        plt.suptitle(targets_to_pretty_names[target])
        if show:
            plt.show()
        else:
            plt.savefig(f"data/images/strat_hist_{target}")
            plt.close(fig)
    return


def print_interaction_correlations(targets, envs, trans_feats):
    data = []

    interactions = [
        ("log10_lr", "gamma_effective_horizon", "LR x Gamma"),
        ("log10_lr", "log10_target_freq", "LR x Target Freq")
    ]

    for target in targets:
        for env in envs:
            df = load(env, "config_metrics")
            df_trans = build_features(df)
            
            for feat in trans_feats:
                v = df_trans[feat]
                df_trans[feat] = (v - v.mean()) / v.std()
                
            df_trans[target] = df[target]
            
            row_data = {"Target": target, "Environment": env}
            
            for feat1, feat2, label in interactions:
                interaction_col = f"{feat1}_x_{feat2}"
                df_trans[interaction_col] = df_trans[feat1] * df_trans[feat2]

                # Spearman
                pcorr_s = pg.partial_corr(data=df_trans, x=interaction_col, y=target, covar=trans_feats, method="spearman")
                sr = pcorr_s.r.item()
                pval_s = pcorr_s.p_val.item()

                # Pearson
                pcorr_p = pg.partial_corr(data=df_trans, x=interaction_col, y=target, covar=trans_feats, method="pearson")
                pr = pcorr_p.r.item()
                pval_p = pcorr_p.p_val.item()

                if pval_s >= 0.05:
                    sr = np.nan
                if pval_p >= 0.05:
                    pr = np.nan

                row_data[f"{label}_Pearson"] = pr
                row_data[f"{label}_Spearman"] = sr

            data.append(row_data)

    df = pd.DataFrame(data)

    multi_cols = [("Setup", "Target"), ("Setup", "Environment")]
    for _, _, label in interactions:
        multi_cols.append((label, "Pearson"))
        multi_cols.append((label, "Spearman"))

    df.columns = pd.MultiIndex.from_tuples(multi_cols)

    latex = df.to_latex(index=False, float_format="%.2f", na_rep="-")

    print(latex)


def plot_frozen_lake():
    from envs import FrozenLake
    lake = FrozenLake().desc
        #     lake = np.asarray([
        #     "S...........",
        #     ".........H..",
        #     "...H........",
        #     ".....H......",
        #     "...H........",
        #     "......H.....",
        #     "...........H",
        #     "...H........",
        #     ".....H......",
        #     "...H......H.",
        #     "......H.....",
        #     "....H.H....G"
        # ], dtype="c")
    lake_rgb = np.zeros((len(lake), len(lake), 3), np.uint8)
    for i in range(len(lake)):
        for j in range(len(lake[0])):
            if lake[i, j] == b"S":
                lake_rgb[i, j] = [255, 0, 0]
            elif lake[i, j] == b".":
                lake_rgb[i, j] = [0, 205, 255]
            elif lake[i, j] == b"H":
                lake_rgb[i, j] = [0, 0, 150]
            elif lake[i, j] == b"G":
                lake_rgb[i, j] = [0, 255, 0]
    import matplotlib.pyplot as plt
    plt.imshow(lake_rgb, )
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"data/images/frozenlake")
    # plt.show()



def plot_test():
    def agg_fn(x):
        return trim_mean(x, proportiontocut=0.25)

    envs = ["pendulum", "breakout"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    
    for i, env in enumerate(envs):
        ax = axes[i]
        data_dir = f"data/test/{env}"
        filename = os.listdir(data_dir)[0]
        df = pd.read_feather(f"{data_dir}/{filename}")

        ret_cols = [c for c in df.columns if "ret_" in c]
        
        groups = df.groupby(["lr", "gamma"])[ret_cols]
        groups_mean = groups.agg(agg_fn)
        groups_q25 = groups.quantile(0.25)
        groups_q75 = groups.quantile(0.75)

        print(env, groups_mean.mean(1).idxmax())
        
        rets = df[ret_cols].values
        W = np.arange(len(rets[0]))
        W = W / np.sum(W) 
        df["auc"] = df[ret_cols].values.mean(1)
        
        vals = groups_mean.values
        xs = (np.arange(len(vals[0])) + 1) * 5
        
        for (lr, gamma), mean_vals in groups_mean.iterrows():
            lr_str = np.format_float_scientific(np.round(lr, 5))
            gamma_str = np.round(gamma, 3)
            label = f"{lr_str}, {gamma_str}"
            line = ax.plot(xs, mean_vals.values, label=label)[0]
            q25_vals = groups_q25.loc[(lr, gamma)].values
            q75_vals = groups_q75.loc[(lr, gamma)].values
            ax.fill_between(xs, q25_vals, q75_vals, color=line.get_color(), alpha=0.2)
            
        ax.set_title(env)
        
    axes[0].set_ylabel("Return")
    fig.supxlabel("Env Steps (Thousands)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="(lr, gamma)", loc="outside right center")

    plt.savefig("data/images/test")


def main():
    os.makedirs("data/images", exist_ok=True)
    envs = ["acrobot", "cartpole", "cliffwalking", "frozenlake", "mountaincar"]
    feats = ["lr", "gamma", "target_net_freq", "buffer_size", "batch_size", "explore_frac", "end_e"]
    trans_feats_to_feats = {
        "log10_lr": "lr", "gamma_effective_horizon": "gamma", 
        "log10_target_freq": "target_net_freq", 
        "log10_buffer": "buffer_size", "log10_batch": "batch_size", 
        "explore": "explore_frac", "log10_end_e": "end_e",
    }
    trans_feats_to_pretty_names = {
        "log10_lr": r"$\alpha$", 
        "gamma_effective_horizon": r"$\gamma$", 
        "log10_target_freq": r"$U$", 
        "log10_buffer": r"$N$", 
        "log10_batch": r"$B$", 
        "explore": r"$f_{\mathrm{exp}}$", 
        "log10_end_e": r"$\epsilon_{\mathrm{final}}$",
    }
    feats_to_pretty_names = {
        "lr": r"$\alpha$", 
        "gamma": r"$\gamma$", 
        "target_net_freq": r"$U$", 
        "buffer_size": r"$N$", 
        "batch_size": r"$B$", 
        "explore_frac": r"$f_{\mathrm{exp}}$", 
        "end_e": r"$\epsilon_{\mathrm{final}}$",
    }
    targets = ['success_rate', 'time_to_success_iqm', 'auc_iqm', ]
    targets_to_pretty_names = {
        "success_rate": "Success Rate", "time_to_success_iqm": "Time to Success", "auc_iqm": "AUC",
    }

    # plot_frozen_lake()
    # plot_ale(trans_feats_to_feats, trans_feats_to_pretty_names, targets_to_pretty_names, show=False)
    # plot_fanova(trans_feats_to_pretty_names, targets_to_pretty_names, show=False)
    # plot_strat_hist(targets, envs, feats, targets_to_pretty_names, feats_to_pretty_names, show=False)
    print_interaction_correlations(targets, envs, list(trans_feats_to_feats.keys()))

    # print(f"Best performing test configs:")
    # plot_test()
    


if __name__ == "__main__":
    main()