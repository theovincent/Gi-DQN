import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
import imageio.v2 as imageio

_HD_DIM = 1080


def _upscale(frame: np.ndarray) -> np.ndarray:
    """Upscale a frame to ~1080p using nearest-neighbor (preserves hard pixel edges)."""
    h, w = frame.shape[:2]
    scale = max(1, _HD_DIM // max(h, w))
    return cv2.resize(frame, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)


def _to_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def _to_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    return frame


def _make_writer(path: str, fps: int, w: int, h: int):
    return imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",  
        quality=8,
        macro_block_size=2,  # allows any even dims
        ffmpeg_params=["-movflags", "+faststart"],  
    )


def evaluate_and_record(agent, params, env, n_episodes: int, output_dir: str, fps: int = 15):
    """
    Videos are upscaled to ~1080p via nearest-neighbor so individual pixels are visible:
    native ALE RGB (e.g. 210×160 -> 1050×800 at 5×)
    standard 84×84 grayscale (-> 1008×1008 at 12×)
    training-resolution grayscale (e.g. 16×16 -> 1072×1072 at 67×)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    px = env.state_height  # for quadratic img
    labels = {
        "native": f"native_{env.original_state_height}x{env.original_state_width}",
        "gray_84": "gray_84x84",
        "gray_pixels": f"gray_{px}x{px}",
    }
    env.reset()
    sample = {k: _upscale(_to_rgb(env.frames[k])) for k in labels}
    dims = {k: (sample[k].shape[1], sample[k].shape[0]) for k in labels}

    episode_returns = []

    for ep in tqdm(range(n_episodes), desc="Evaluating"):

        writers = {
            k: _make_writer(os.path.join(output_dir, f"ep{ep:03d}_{labels[k]}.mp4"), fps, *dims[k]) for k in labels
        }

        env.reset()
        for k in labels:
            writers[k].append_data(_upscale(_to_rgb(env.frames[k])))

        done = False
        ep_return = 0.0
        n_steps = 0

        while not done:
            action = agent.best_action(params, env.state)
            reward, done = env.step(action)
            ep_return += reward
            n_steps += 1
            for k in labels:
                writers[k].append_data(_upscale(_to_rgb(env.frames[k])))

        for w in writers.values():
            w.close()  # flush mem

        episode_returns.append(ep_return)
        print(f"  ep {ep:3d}: return = {ep_return:.1f}  ({n_steps} steps)")

    mean_returns = float(np.mean(episode_returns))
    print(f"\nMean return over {n_episodes} episodes: {mean_returns:.2f}")
    print(f"Videos saved to: {output_dir}")
    return episode_returns

