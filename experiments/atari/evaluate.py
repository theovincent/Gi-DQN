import argparse
import json
import os
import pickle
import sys

import jax

from experiments.base.evaluate import evaluate_and_record
from slimdqn.environments.atari_recorder import AtariEnvRecorder


def _build_agent(algo_name: str, p: dict, env, key):

    obs_dim = (env.state_height, env.state_width, env.n_stacked_frames)

    common = dict(
        key=key,
        observation_dim=obs_dim,
        n_actions=env.n_actions,
        features=p["features"],
        architecture_type=p["architecture_type"],
        layer_norm=tuple(p["layer_norm"]),
        gap=p["gap"],
        learning_rate=p["learning_rate"],
        gamma=p["gamma"],
        update_horizon=p["update_horizon"],
        update_to_data=p["update_to_data"],
        target_update_period=p["target_update_period"],
        pixels=p.get("pixels", 84),
        kernel=p.get("kernel", 3),
        stride=p.get("stride", 1),
        n_conv=p.get("n_conv", 1),
        n_fc=p.get("n_fc", 2),
    )

    if algo_name == "dqn":
        from slimdqn.algorithms.dqn import DQN

        return DQN(**common)

    if algo_name == "dqnrcshared":
        from slimdqn.algorithms.dqnrcshared import DQNRCShared

        return DQNRCShared(
            **common,
            linear_heads=p.get("linear_heads", True),
            weight_decay=p.get("weight_decay", 1.0),
        )

    if algo_name == "gidqnshared":
        from slimdqn.algorithms.gidqnshared import GiDQNShared

        return GiDQNShared(
            **common,
            n_bellman_iterations=p["n_bellman_iterations"],
            linear_heads=p.get("linear_heads", True),
            weight_decay=p.get("weight_decay", 1.0),
        )

    if algo_name == "idqnshared":
        from slimdqn.algorithms.idqnshared import iDQNShared

        return iDQNShared(
            **common,
            n_bellman_iterations=p["n_bellman_iterations"],
            linear_heads=p.get("linear_heads", True),
        )

    if algo_name == "isfidqnshared":
        from slimdqn.algorithms.isdqnshared import iSDQNShared

        return iSDQNShared(
            **common,
            n_bellman_iterations=p["n_bellman_iterations"],
            linear_heads=p.get("linear_heads", True),
            iterated_shared_features=p.get("iterated_shared_features", False),
        )

    if algo_name == "mmgidqnshared":
        from slimdqn.algorithms.mmgidqnshared import MMGiDQNShared

        return MMGiDQNShared(
            **common,
            n_bellman_iterations=p["n_bellman_iterations"],
            linear_heads=p.get("linear_heads", True),
            weight_decay=p.get("weight_decay", 1.0),
            omega=p.get("omega", 5.0),
        )

    if algo_name == "mmdqnrcshared":
        from slimdqn.algorithms.mmdqnrcshared import MMDQNRCShared

        return MMDQNRCShared(
            **common,
            linear_heads=p.get("linear_heads", True),
            weight_decay=p.get("weight_decay", 1.0),
            omega=p.get("omega", 5.0),
        )

    if algo_name == "gisdqnshared":
        from slimdqn.algorithms.gisdqnshared import GiSDQNShared

        return GiSDQNShared(
            **common,
            n_bellman_iterations=p["n_bellman_iterations"],
            linear_heads=p.get("linear_heads", True),
            weight_decay=p.get("weight_decay", 1.0),
            unfreeze_first_head=p.get("unfreeze_first_head", False),
            iterated_shared_features=p.get("iterated_shared_features", False),
        )

    raise ValueError(
        f"Unknown algo_name '{algo_name}'. "
        "Supported: dqn, dqnrcshared, gidqnshared, idqnshared, isdqnshared, "
        "mmgidqnshared, mmdqnrcshared, gisdqnshared"
    )


def run(argvs=sys.argv[1:]):
    parser = argparse.ArgumentParser("Evaluate a trained Atari agent and record videos at 3 resolutions.")
    parser.add_argument(
        "--experiment_path",
        type=str,
        required=True,
        help="Path to the experiment folder (the one that contains parameters.json). "
        "E.g. experiments/atari/exp_output/L2_K5_WD1.../",
    )
    parser.add_argument(
        "--algo_name",
        type=str,
        required=True,
        help="Algorithm name. " "(e.g. dqn).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Seed of model to load.",
    )
    parser.add_argument(
        "--n_eval_episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save videos. " "Defaults to {experiment_path}/{algo_name}/videos/seed{seed}/.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Video frame rate.",
    )
    args = parser.parse_args(argvs)

    params_path = os.path.join(args.experiment_path, "parameters.json")
    assert os.path.exists(params_path), f"parameters.json not found at: {params_path}"
    raw = json.load(open(params_path))

    shared = raw["shared_parameters"]
    algo = raw.get(args.algo_name, {})
    p = {**shared, **algo}

    game_name = shared["experiment_name"].split("_")[-1]

    env = AtariEnvRecorder(
        name=game_name,
        state_height_width=(p["pixels"], p["pixels"]),
        n_stacked_frames=p.get("n_frame_stack", 2),
        n_skipped_frames=p.get("n_frame_skip", 4),
    )

    key = jax.random.PRNGKey(args.seed)
    agent = _build_agent(args.algo_name, p, env, key)

    model_path = os.path.join(args.experiment_path, args.algo_name, "models", str(args.seed))
    assert os.path.exists(model_path), f"Model not found at: {model_path}"
    model = pickle.load(open(model_path, "rb"))
    params = model["params"]

    output_dir = args.output_dir or os.path.join(args.experiment_path, args.algo_name, "videos", f"seed{args.seed}")

    print(f"\nEvaluating {args.algo_name} on {game_name} " f"(pixels={p['pixels']}, seed={args.seed})")
    print(f"  model : {model_path}")
    print(f"  output: {output_dir}\n")

    evaluate_and_record(agent, params, env, args.n_eval_episodes, output_dir, fps=args.fps)


if __name__ == "__main__":
    run()
