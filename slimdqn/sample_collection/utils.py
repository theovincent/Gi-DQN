import jax
import jax.numpy as jnp
from functools import partial

from slimdqn.sample_collection.replay_buffer import ReplayBuffer, TransitionElement


@partial(jax.jit, static_argnames=("best_action_fn", "n_actions", "epsilon_fn", "epsilon_td"))
def select_action(best_action_fn, params, state, key, n_actions, epsilon_fn, n_training_steps, epsilon_td):
    uniform_key, action_key = jax.random.split(key)

    if epsilon_td:
        best_action, explore_action = best_action_fn(params, state)
    else:
        best_action = best_action_fn(params, state)
        explore_action = jax.random.randint(action_key, (), 0, n_actions)

    return jnp.where(
        jax.random.uniform(uniform_key) <= epsilon_fn(n_training_steps),  # if uniform < epsilon,
        explore_action,  # take an exploratory action
        best_action,  # otherwise, take a greedy action
    )


def collect_single_sample(key, env, agent, rb: ReplayBuffer, p, epsilon_schedule, n_training_steps: int):
    action = select_action(
        agent.best_action,
        agent.params,
        env.state,
        key,
        env.n_actions,
        epsilon_schedule,
        n_training_steps,
        p.get("epsilon_td", False),
    ).item()

    obs = env.observation
    reward, absorbing = env.step(action)

    episode_end = absorbing or env.n_steps >= p["horizon"]
    rb.add(
        TransitionElement(
            observation=obs,
            action=action,
            reward=reward if rb.clipping is None else rb.clipping(reward),
            is_terminal=absorbing,
            episode_end=episode_end,
        )
    )

    if episode_end:
        env.reset()

    return reward, episode_end
