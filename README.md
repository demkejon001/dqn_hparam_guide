# Hand Tuning DQN Hyperparameters

This repository contains the official code for reproducing the results in the paper **"Hand Tuning Deep Q-Network Hyperparameters: A Practical Guide"**.

## Installation

To ensure full reproducibility, we highly recommend using a virtual environment. The experiments were run using **Python 3.10**.

1. **Create and activate a virtual environment (using Conda):**

```bash
conda create -n dqn_hparam_guide python=3.10
conda activate dqn_hparam_guide
```

2. **Install the required dependencies:**

```bash
pip install -r requirements.txt
```

## Reproducing the Results

### Step 1: Generating Raw Data

To train the millions of agents for the results of this paper, we utilized a slurm-capable supercomputer. The script below runs a massive grid search across environments and hyperparameters. *(Note: The 7 CPUs per task are specific to our cluster and are not strictly necessary to reproduce the results).*

```bash
#!/bin/bash --login

#SBATCH --job-name=dqn_tuning
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=31G
#SBATCH --gpus=1
#SBATCH --array=0-374

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
conda activate dqn_hparam_guide

# Define all parameter arrays
envs=(acrobot cartpole cliffwalking frozenlake mountaincar)
explore_fracs=(0.1 0.3 0.5 0.7 0.9)
batch_sizes=(32 128 512)
buffer_sizes=(500 1000 5000 10000 50000)

# Calculate lengths
n_env=${#envs[@]}
n_exp=${#explore_fracs[@]}
n_batch=${#batch_sizes[@]}
n_buffer=${#buffer_sizes[@]}

task_id=$SLURM_ARRAY_TASK_ID

# Map task_id to individual indices
env_idx=$(( task_id / (n_exp * n_batch * n_buffer) ))
exp_idx=$(( (task_id / (n_batch * n_buffer)) % n_exp ))
batch_idx=$(( (task_id / n_buffer) % n_batch ))
buffer_idx=$(( task_id % n_buffer ))

env=${envs[$env_idx]}
explore_frac=${explore_fracs[$exp_idx]}
batch_size=${batch_sizes[$batch_idx]}
buffer_size=${buffer_sizes[$buffer_idx]}

python run_experiments.py --env_type ${env} --n_seeds 10 --exp_name gsfull --explore_frac ${explore_frac} --batch_size ${batch_size} --buffer_size ${buffer_size}
```

> **Warning:** This generates roughly **23 GB** of data in the `data/raw/` directory.

### Step 2: Data Aggregation & Analysis

Once the raw data is generated run the aggregation script:

```bash
python aggregate_results.py
```

**What this script does:**

1. **Computes Performance Metrics:** The first half of the script iterates over the raw data to compute the agent's performance metrics (Success Indicator, Time to Success, and AUC). Then it aggregates the agent metrics into an average success rate and interquartile means (IQM) for Time to Success and AUC. These are the configuration metrics.

2. **Hyperparameter Analysis:** The second half of the script trains random forest regressors to predict aggregated metrics based on hyperparameters. It then runs ALE and fANOVA analyses. Final computed results are saved to `data/analysis/ale.feather` and `data/analysis/fanova.feather`.

> If you are unable to generate the raw data, we have provided the pre-computed agent and config metrics located in `data/analysis/[ENV]/agent_metrics.pkl` and `data/analysis/[ENV]/config_metrics.pkl` (where `[ENV]` represents the 5 environments we trained on our agents on: acrobot, cartpole, cliffwalking, frozenlake, and mountaincar).

### Step 3: Held-Out Validation

The next step is to get the held-out validation results:

```bash
python run_experiments.py --test --env_type breakout --seed 42 --n_seeds 10 --target_net_freq 700 --buffer_size 10000 --batch_size 512 --explore_frac 0.1 --end_e .01
python run_experiments.py --test --env_type pendulum --seed 42 --n_seeds 10 --target_net_freq 700 --buffer_size 10000 --batch_size 512 --explore_frac 0.1 --end_e .01
```

### Step 4: Generating Plots & Tables

Finally, you can generate all the results, tables, and images featured in the paper by running:

```bash
python plot_results.py
```

All generated plots will be saved in the `data/images/` directory.