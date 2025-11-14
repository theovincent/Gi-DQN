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
        "-tuf",
        "--target_update_frequency",
        help="Number of training steps before updating the target Q-network.",
        type=int,
        default=200,
    )


def add_n_bellman_iterations(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-nbi",
        "--n_bellman_iterations",
        type=int,
        help="Number of bellman iterations.",
        default=1,
    )


def add_weight_decay(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-wd",
        "--weight_decay",
        help="Weighting of the regularization in weight decay.",
        type=float,
        default=0.001,
    )


def add_freeze_first_head(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--unfreeze_first_head",
        help="Whether the first network should be fixed or not for the duration of a Bellman iteration",
        action="store_true",
        default=False,
    )


def add_target_sync_frequency(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-tsf",
        "--target_sync_frequency",
        help="Number of training steps before updating each target Q-network to its corresponding online Q-network. (D)",
        type=int,
        default=10,
    )


def add_mu(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-mu",
        "--mu",
        help="Factor that is multiplied by the alpha loss before adding it to the overall loss.",
        type=float,
        default=1,
    )


def add_distributional_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-nb",
        "--n_bins",
        help="Number of bins to use for the categorical distribution.",
        type=int,
        default=50,
    )
    parser.add_argument(
        "-minn",
        "--min_value",
        help="Value of the lowest learnable value of the target.",
        type=float,
        default=-100,
    )
    parser.add_argument(
        "-maxn",
        "--max_value",
        help="Value of the highest learnable value of the target.",
        type=float,
        default=100,
    )
    parser.add_argument(
        "-sigma",
        "--sigma",
        help="Standard deviation of each target sample. If \sigma / \eta = 0.75, then \sigma = 0.75 * (max_value - min_value) / n_bins",
        type=float,
        default=3,
    )

@output_added_arguments
def add_layer_norm(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-ln",
        "--layer_norm",
        help="Whether to use layer normalization.",
        default=False,
        action="store_true",
    )

@output_added_arguments
def add_gidqn_arguments(parser: argparse.ArgumentParser):
    add_n_bellman_iterations(parser)
    add_weight_decay(parser)
    add_freeze_first_head(parser)
    add_layer_norm(parser)


@output_added_arguments
def add_gidqnshared_arguments(parser: argparse.ArgumentParser):
    add_n_bellman_iterations(parser)
    add_weight_decay(parser)
    add_freeze_first_head(parser)
    add_mu(parser)
    add_layer_norm(parser)


@output_added_arguments
def add_hlgidqnshared_arguments(parser: argparse.ArgumentParser):
    add_n_bellman_iterations(parser)
    add_weight_decay(parser)
    add_freeze_first_head(parser)
    add_mu(parser)
    add_distributional_arguments(parser)
    add_layer_norm(parser)


@output_added_arguments
def add_fidqn_arguments(parser: argparse.ArgumentParser):
    add_n_bellman_iterations(parser)
    add_layer_norm(parser)


@output_added_arguments
def add_fidqnshared_arguments(parser: argparse.ArgumentParser):
    add_n_bellman_iterations(parser)
    add_layer_norm(parser)


@output_added_arguments
def add_hlfidqnshared_arguments(parser: argparse.ArgumentParser):
    add_n_bellman_iterations(parser)
    add_distributional_arguments(parser)
    add_layer_norm(parser)


@output_added_arguments
def add_idqn_arguments(parser: argparse.ArgumentParser):
    add_n_bellman_iterations(parser)
    add_target_sync_frequency(parser)
    add_layer_norm(parser)


@output_added_arguments
def add_idqnshared_arguments(parser: argparse.ArgumentParser):
    add_n_bellman_iterations(parser)
    add_target_sync_frequency(parser)
    add_layer_norm(parser)


@output_added_arguments
def add_hlidqnshared_arguments(parser: argparse.ArgumentParser):
    add_n_bellman_iterations(parser)
    add_target_sync_frequency(parser)
    add_distributional_arguments(parser)
    add_layer_norm(parser)


@output_added_arguments
def add_dqn_arguments(parser: argparse.ArgumentParser):
    add_layer_norm(parser)
    pass


@output_added_arguments
def add_hldqn_arguments(parser: argparse.ArgumentParser):
    add_distributional_arguments(parser)
    add_layer_norm(parser)


@output_added_arguments
def add_dqnrc_arguments(parser: argparse.ArgumentParser):
    add_weight_decay(parser)
    add_layer_norm(parser)


@output_added_arguments
def add_dqnrcshared_arguments(parser: argparse.ArgumentParser):
    add_weight_decay(parser)
    add_mu(parser)
    add_layer_norm(parser)


@output_added_arguments
def add_hldqnrcshared_arguments(parser: argparse.ArgumentParser):
    add_weight_decay(parser)
    add_mu(parser)
    add_distributional_arguments(parser)
    add_layer_norm(parser)
