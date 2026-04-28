import numpy as np
import cv2

from slimdqn.environments.atari import AtariEnv


class AtariEnvRecorder(AtariEnv):
    """AtariEnv that captures frames at 3 resolutions after every step and reset.

    Adds a `frames` dict with keys:
      "native"      — (H, W, 3) uint8 RGB at native ALE resolution (e.g. 210×160)
      "gray_84"     — (84, 84) uint8 grayscale, standard ALE downscale
      "gray_pixels" — (pixels, pixels) uint8 grayscale, training resolution
    """

    _GRAY_84 = 84

    def reset(self):
        super().reset()
        self._capture_frames()

    def step(self, action):
        reward, terminal = super().step(action)
        self._capture_frames()
        return reward, terminal

    def _capture_frames(self):
        native_rgb = np.empty(
            (self.original_state_height, self.original_state_width, 3), dtype=np.uint8
        )
        self.env.env.ale.getScreenRGB(native_rgb)

        # screen_buffer[0] holds the max-pooled native-resolution grayscale after step,
        # or the plain first frame after reset — correct in both cases for display.
        gray_84 = cv2.resize(
            self.screen_buffer[0], (self._GRAY_84, self._GRAY_84), interpolation=cv2.INTER_AREA
        )

        self.frames = {
            "native": native_rgb,
            "gray_84": gray_84,
            "gray_pixels": self.state_[:, :, -1].copy(),  # latest frame in stack
        }

