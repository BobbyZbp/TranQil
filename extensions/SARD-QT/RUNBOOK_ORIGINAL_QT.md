# Original QT AntMaze Runbook

All commands assume repo root: `/home/bobby/QT`.

## 1. Create Environment

```bash
MAMBA_ROOT_PREFIX=/home/bobby/QT/.micromamba/root \
  /home/bobby/TranQil/.micromamba/micromamba create -y -f environment.yml
```

## 2. Activate

```bash
source scripts/activate_env.sh
```

## 3. Install Runtime Stack

```bash
source scripts/activate_env.sh
scripts/install_original_qt_stack.sh
```

Important pins:

- `transformers==4.5.1`
- `packaging<22`
- `torch==2.4.1+cu121`

## 4. Verify Runtime

```bash
source scripts/activate_env.sh
scripts/verify_original_qt_stack.sh
```

## 5. Generate Datasets

```bash
source scripts/activate_env.sh
python D4RL/create_dataset.py antmaze-large-diverse-v0 antmaze-large-diverse-v2
```

Expected outputs:

- `D4RL/antmaze-large-diverse-v0.pkl`
- `D4RL/antmaze-large-diverse-v2.pkl`

## 6. Smoke Tests

Use tiny smoke settings on CPU in this WSL sandbox. The full default QT model with batch 256 can be killed on CPU because QT expands batches through its 50-candidate path.

v0:

```bash
source scripts/activate_env.sh
python experiment.py --seed 123 \
  --env antmaze --dataset large-diverse --d4rl_version 0 \
  --eta 0.005 --grad_norm 9.0 \
  --exp_name smoke_v0 --save_path ./save/ \
  --max_iters 1 --num_steps_per_iter 1 --num_eval_episodes 1 \
  --lr_decay --early_stop --early_epoch 0 \
  --k_rewards --use_discount --reward_tune cql_antmaze \
  --device cpu --K 5 --batch_size 2 --embed_dim 32 --n_layer 1 --n_head 1
```

v2:

```bash
source scripts/activate_env.sh
python experiment.py --seed 123 \
  --env antmaze --dataset large-diverse --d4rl_version 2 \
  --eta 0.005 --grad_norm 9.0 \
  --exp_name smoke_v2 --save_path ./save/ \
  --max_iters 1 --num_steps_per_iter 1 --num_eval_episodes 1 \
  --lr_decay --early_stop --early_epoch 0 \
  --k_rewards --use_discount --reward_tune cql_antmaze \
  --device cpu --K 5 --batch_size 2 --embed_dim 32 --n_layer 1 --n_head 1
```

## 7. Full Diagnostics

Paper-matched v0:

```bash
tmux new -s qt_antmaze_v0
source scripts/activate_env.sh
D4RL_SUPPRESS_IMPORT_ERROR=1 python experiment.py --seed 123 \
  --env antmaze --dataset large-diverse --d4rl_version 0 \
  --eta 0.005 --grad_norm 9.0 \
  --exp_name qt_antmaze_v0 --save_path ./save/ \
  --max_iters 100 --num_steps_per_iter 1000 --lr_decay \
  --num_eval_episodes 10 --early_stop --early_epoch 80 \
  --k_rewards --use_discount --reward_tune cql_antmaze
```

v2 diagnostic:

```bash
tmux new -s qt_antmaze_v2
source scripts/activate_env.sh
D4RL_SUPPRESS_IMPORT_ERROR=1 python experiment.py --seed 123 \
  --env antmaze --dataset large-diverse --d4rl_version 2 \
  --eta 0.005 --grad_norm 9.0 \
  --exp_name qt_antmaze_v2 --save_path ./save/ \
  --max_iters 100 --num_steps_per_iter 1000 --lr_decay \
  --num_eval_episodes 10 --early_stop --early_epoch 80 \
  --k_rewards --use_discount --reward_tune cql_antmaze
```

Check logs:

```bash
tmux capture-pane -pt qt_antmaze_v0:0.0 -S -120
tmux capture-pane -pt qt_antmaze_v2:0.0 -S -120
find save -name progress.csv -o -name debug.log
```
