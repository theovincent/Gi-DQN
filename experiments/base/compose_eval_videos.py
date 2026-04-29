#!/usr/bin/env python
"""Concatenate per-episode resolution videos side-by-side for visual comparison.

for example:
# All three streams left-to-right:
python compose_eval_videos.py --videos_dir .../seed1 --include native ale agent --output_dir /out/three_way
"""

import argparse
import os
import re
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm


STREAM_PATTERNS = {
    "native": re.compile(r"^ep(\d+)_native_\d+x\d+\.mp4$"),
    "ale":    re.compile(r"^ep(\d+)_gray_84x84\.mp4$"),
    # "agent" = any PxP gray that is NOT 84x84 (negative lookahead)
    "agent":  re.compile(r"^ep(\d+)_gray_(?!84x84)(\d+)x\2\.mp4$"),
}

STREAM_LABELS = {
    "native": "Native (raw ALE)",
    "ale":    "Standard 84x84",
    "agent":  "Agent input",
}


def find_episodes(videos_dir, streams):
    """Return only episodes with all requested streams."""
    by_ep = {}
    for fname in sorted(os.listdir(videos_dir)):
        for stream in streams:
            m = STREAM_PATTERNS[stream].match(fname)
            if m:
                ep = int(m.group(1))
                by_ep.setdefault(ep, {})[stream] = os.path.join(videos_dir, fname)
                break
    return {ep: paths for ep, paths in sorted(by_ep.items())
            if len(paths) == len(streams)}


def _pad_to_height(frame, target_h):
    """Center-pad a frame vertically with black to reach target_h."""
    h = frame.shape[0]
    if h == target_h:
        return frame
    top = (target_h - h) // 2
    bot = target_h - h - top
    return cv2.copyMakeBorder(frame, top, bot, 0, 0,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))


def _ensure_even(canvas):
    """libx264 + yuv420p needs even width and height: pad by 1 if odd."""
    h, w = canvas.shape[:2]
    pad_h, pad_w = h % 2, w % 2
    if pad_h or pad_w:
        canvas = cv2.copyMakeBorder(canvas, 0, pad_h, 0, pad_w,
                                    cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return canvas


def compose_episode(stream_paths, stream_order, output_path, label=True):
    """Stream all input readers in lockstep, hstack frames, write one MP4."""
    readers = {s: imageio.get_reader(stream_paths[s]) for s in stream_order}
    try:
        fps = readers[stream_order[0]].get_meta_data().get("fps", 15)

        # Pull first frames to size the canvas
        first = {s: readers[s].get_next_data() for s in stream_order}

        labels = {**STREAM_LABELS, "agent": f"Agent input: {first['agent'].shape[1]}x{first['agent'].shape[0]}"} if "agent" in first else STREAM_LABELS

        panel_h = max(f.shape[0] for f in first.values())
        panel_widths = [first[s].shape[1] for s in stream_order]
        total_w = sum(panel_widths)
        label_h = 36 if label else 0

        writer = imageio.get_writer(
            output_path,
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            quality=8,
            macro_block_size=2,
            ffmpeg_params=["-movflags", "+faststart"],
        )
        try:
            def compose(frame_dict):
                panels = [_pad_to_height(frame_dict[s], panel_h) for s in stream_order]
                row = np.hstack(panels)
                if not label:
                    return _ensure_even(row)
                band = np.zeros((label_h, total_w, 3), dtype=np.uint8)
                x = 0
                for s, w in zip(stream_order, panel_widths):
                    cv2.putText(band, STREAM_LABELS[s], (x + 14, label_h - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240),
                                2, cv2.LINE_AA)
                    x += w
                return _ensure_even(np.vstack([band, row]))

            writer.append_data(compose(first))

            while True:
                try:
                    frames = {s: readers[s].get_next_data() for s in stream_order}
                except (StopIteration, IndexError):
                    break  # one stream ran out — stop together
                writer.append_data(compose(frames))
        finally:
            writer.close()
    finally:
        for r in readers.values():
            r.close()


def main():
    ap = argparse.ArgumentParser(
        description="Side-by-side concatenate per-episode evaluation videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--videos_dir", required=True,
                    help="Path to a seedN/ folder containing ep*_*.mp4 files.")
    ap.add_argument("--include", nargs="+", required=True,
                    choices=["native", "ale", "agent"],
                    help="Streams to combine, left-to-right. "
                         "native=raw ALE RGB, ale=84x84 gray, agent=training-resolution gray.")
    ap.add_argument("--output_dir", default=None,
                    help="Where to write the combined videos. "
                         "Default: <videos_dir>/comparison_<streams>.")
    ap.add_argument("--no_label", action="store_true",
                    help="Disable the text label band above each panel.")
    args = ap.parse_args()

    if len(args.include) < 2:
        ap.error("--include needs at least 2 streams (otherwise there's nothing to compare).")
    if len(set(args.include)) != len(args.include):
        ap.error("--include cannot contain duplicates.")

    videos_dir = os.path.abspath(args.videos_dir)
    out_dir = args.output_dir or os.path.join(
            videos_dir, "comparison_" + "_".join(args.include)
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    episodes = find_episodes(videos_dir, args.include)
    if not episodes:
        raise SystemExit(
            f"No episodes in {videos_dir} have all of: {args.include}\n"
            f"Files present: {sorted(os.listdir(videos_dir))[:8]}..."
        )

    print(f"Found {len(episodes)} episode(s) with streams {args.include}")
    print(f"Output: {out_dir}\n")

    for ep, paths in tqdm(episodes.items(), desc="Composing"):
        out_name = f"ep{ep:03d}_" + "_".join(args.include) + ".mp4"
        out_path = os.path.join(out_dir, out_name)
        compose_episode(paths, args.include, out_path, label=not args.no_label)


if __name__ == "__main__":
    main()
