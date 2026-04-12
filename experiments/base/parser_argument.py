import argparse
from functools import wraps
from typing import Callable, List


def output_added_arguments(add_algo_arguments: Callable) -> Callable:
    @wraps(add_algo_arguments)
    def decorated(parser: argparse.ArgumentParser) -> List[str]:
        unfiltered_old_arguments = list(parser._option_string_actions.keys())

        add_algo_arguments(parser)

        unfiltered_arguments = list(parser._option_string_actions.keys())
        unfiltered_added_arguments = [
            argument for argument in unfiltered_arguments if argument not in unfiltered_old_arguments
        ]

        return [
            argument.strip("-")
            for argument in unfiltered_added_arguments
            if argument.startswith("--") and argument not in ["--help"]
        ]

    return decorated


@output_added_arguments
def add_base_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-en",
        "--experiment_name",
        help="Experiment name.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-s",
        "--seed",
        help="Seed of the experiment.",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-dw",
        "--disable_wandb",
        help="Disable wandb.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-f",
        "--features",
        nargs="*",
        help="List of features for the Q-networks.",
        type=int,
        default=[200, 200],
    )
    parser.add_argument(
        "-rbc",
        "--replay_buffer_capacity",
        help="Replay Buffer capacity.",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        help="Batch size for training.",
        type=int,
        default=32,
    )
    parser.add_argument(
        "-n",
        "--update_horizon",
        help="Value of n in n-step TD update.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-gamma",
        "--gamma",
        help="Discounting factor.",
        type=float,
        default=0.99,
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        help="Learning rate.",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "-horizon",
        "--horizon",
        help="Horizon for truncation.",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "-at",
        "--architecture_type",
        help="Type of architecture.",
        type=str,
        default="fc",
        choices=["cnn", "impala", "fc"],
    )
    parser.add_argument(
        "-ne",
        "--n_epochs",
        help="Number of epochs to perform.",
        type=int,
        default=50,
    )
    parser.add_argument(
        "-ntspe",
        "--n_training_steps_per_epoch",
        help="Number of training steps per epoch.",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "-wtp",
        "--wind_and_turbulence_power",
        nargs=2,
        help="List of wind power (0.0 - 20.0) and turbulence power (0.0 - 2.0)",
        type=float,
        default=[15.0, 1.5],
    )
    parser.add_argument(
        "-utd",
        "--update_to_data",
        help="Number of data points to collect per online Q-network update.",
        type=float,
        default=1,
    )
    parser.add_argument(
        "-nis",
        "--n_initial_samples",
        help="Number of initial samples before the training starts.",
        type=int,
        default=1_000,
    )
    parser.add_argument(
        "-ee",
        "--epsilon_end",
        help="Ending value for the linear decaying epsilon used for exploration.",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "-ed",
        "--epsilon_duration",
        help="Duration of epsilon's linear decay used for exploration.",
        type=float,
        default=1_000,
    )
    parser.add_argument(
        "-tup",
        "--target_update_period",
        help="Number of training steps before updating the target Q-network.",
        type=int,
        default=200,
    )
    parser.add_argument(
        "-ln",
        "--layer_norm",
        help="Layer norm for (conv, fc) layers.",
        nargs=2,
        type=lambda x: x.lower() in ("1", "true"),
        default=[False, False],
    )
    parser.add_argument(
        "--gap",
        help="Whether to use Global Average Pruning.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--per",
        help="Whether to use Prioritized Experience Replay.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--low_scale",
        help="Whether to use low-scale images of 42 x 42 instead of 84 x 84 pixels.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--n_conv",
        help="Number of Convolutional Layers. Min=1, Max=3. Ignored when architecture other than CNN chosen.",
        default=3,
        type=int,
        choices=range(1, 4),
    )
    parser.add_argument(
        "--n_fc",
        help="Number of FC layers after Conv layers, counting also FC mapping to actions. Min=1, Max=2.",
        type=int,
        default=2,
        choices=range(1, 3),
    )
    parser.add_argument(
        "--n_frame_stack",
        help="Number of ALE frames to stack in RB sample.",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--n_frame_skip",
        help="Number of ALE frames to skip before receiving a sample.",
        default=4,
        type=int,
    )


def add_n_bellman_iterations(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-nbi",
        "--n_bellman_iterations",
        type=int,
        help="Number of bellman iterations.",
        default=1,
    )


def add_freeze_first_head(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--unfreeze_first_head",
        help="Whether the first network should be fixed or not for the duration of a Bellman iteration",
        action="store_true",
        default=False,
    )


def add_iterated_shared_features(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--iterated_shared_features",
        help="Whether to use only one network instead of a second one for the first target.",
        action="store_true",
        default=False,
    )


def add_weight_decay(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-wd",
        "--weight_decay",
        help="Weighting of the regularization in weight decay.",
        type=float,
        default=1,
    )

def add_omega(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-omg",
        "--omega",
        help="Temperature for MellowMax function in Target.",
        type=float,
        default=1,
    )


def add_linear_heads(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--linear_heads",
        help="Whether to share only the last layer instead of the two last layers.",
        default=False,
        action="store_true",
    )


@output_added_arguments
def add_dqn_arguments(parser: argparse.ArgumentParser):
    pass


@output_added_arguments
def add_dqnrc_arguments(parser: argparse.ArgumentParser):
    add_weight_decay(parser)


@output_added_arguments
def add_dqnrcshared_arguments(parser: argparse.ArgumentParser):
    add_weight_decay(parser)
    add_linear_heads(parser)
    add_iterated_shared_features(parser)

@output_added_arguments
def add_mmdqnrcshared_arguments(parser: argparse.ArgumentParser):
    add_weight_decay(parser)
    add_linear_heads(parser)
    add_iterated_shared_features(parser)
    add_omega(parser)

@output_added_arguments
def add_idqn_arguments(parser: argparse.ArgumentParser):
    add_n_bellman_iterations(parser)
    add_iterated_shared_features(parser)


@output_added_arguments
def add_idqnshared_arguments(parser: argparse.ArgumentParser):
    add_n_bellman_iterations(parser)
    add_linear_heads(parser)


@output_added_arguments
def add_isfidqnshared_arguments(parser: argparse.ArgumentParser):
    add_n_bellman_iterations(parser)
    add_linear_heads(parser)
    add_iterated_shared_features(parser)


@output_added_arguments
def add_gidqn_arguments(parser: argparse.ArgumentParser):
    add_n_bellman_iterations(parser)
    add_weight_decay(parser)
    add_freeze_first_head(parser)


@output_added_arguments
def add_gidqnshared_arguments(parser: argparse.ArgumentParser):
    add_n_bellman_iterations(parser)
    add_weight_decay(parser)
    add_linear_heads(parser)

@output_added_arguments
def add_mmgidqnshared_arguments(parser: argparse.ArgumentParser):
    add_n_bellman_iterations(parser)
    add_weight_decay(parser)
    add_linear_heads(parser)
    add_omega(parser)


@output_added_arguments
def add_ufhisfgidqnshared_arguments(parser: argparse.ArgumentParser):
    add_n_bellman_iterations(parser)
    add_weight_decay(parser)
    add_linear_heads(parser)
    add_freeze_first_head(parser)
    add_iterated_shared_features(parser)
