import argparse
import collections
import pickle
from pathlib import Path

import d4rl
import gym
import numpy as np


DEFAULT_NAMES = [
    "antmaze-large-diverse-v0",
    "antmaze-large-diverse-v2",
]


def convert_dataset(env_name, output_dir):
    env = gym.make(env_name)
    print(f"[create_dataset] {env_name}")
    print(f"[create_dataset] observation_space={env.observation_space}")
    print(f"[create_dataset] action_space={env.action_space}")

    dataset = env.get_dataset()
    print(f"[create_dataset] raw_keys={sorted(dataset.keys())}")

    n_steps = dataset["rewards"].shape[0]
    data = collections.defaultdict(list)
    paths = []
    use_timeouts = "timeouts" in dataset
    episode_step = 0

    for i in range(n_steps):
        done_bool = bool(dataset["terminals"][i])
        if use_timeouts:
            final_timestep = bool(dataset["timeouts"][i])
        else:
            final_timestep = episode_step == 1000 - 1

        for key in ["observations", "actions", "rewards", "terminals"]:
            data[key].append(dataset[key][i])

        if done_bool or final_timestep:
            episode_step = 0
            paths.append({key: np.asarray(values) for key, values in data.items()})
            data = collections.defaultdict(list)
        else:
            episode_step += 1

    if data:
        paths.append({key: np.asarray(values) for key, values in data.items()})

    returns = np.asarray([np.sum(path["rewards"]) for path in paths])
    num_samples = int(np.sum([path["rewards"].shape[0] for path in paths]))
    output_path = output_dir / f"{env_name}.pkl"
    with output_path.open("wb") as f:
        pickle.dump(paths, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[create_dataset] wrote={output_path}")
    print(f"[create_dataset] trajectories={len(paths)}")
    print(f"[create_dataset] samples={num_samples}")
    print(
        "[create_dataset] returns "
        f"mean={np.mean(returns):.4f} std={np.std(returns):.4f} "
        f"max={np.max(returns):.4f} min={np.min(returns):.4f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "names",
        nargs="*",
        default=DEFAULT_NAMES,
        help="D4RL gym ids to convert, e.g. antmaze-large-diverse-v0",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("D4RL"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for env_name in args.names:
        convert_dataset(env_name, args.output_dir)


if __name__ == "__main__":
    main()
