"""Always-listening wake-word detection with openWakeWord."""

from __future__ import annotations

import numpy as np


class WakeDetector:
    def __init__(self, model_name: str, threshold: float = 0.5):
        from openwakeword import utils
        from openwakeword.model import Model

        utils.download_models()  # no-op when already cached
        # A path to a custom-trained .onnx (e.g. "heyo.onnx") also works here.
        self.model = Model(wakeword_models=[model_name], inference_framework="onnx")
        self.threshold = threshold

    def feed(self, frame: np.ndarray) -> bool:
        """frame: int16 mono samples (80 ms @ 16 kHz). True when the wake word fired."""
        scores = self.model.predict(frame)
        if any(score >= self.threshold for score in scores.values()):
            self.model.reset()  # avoid double-triggering on the same utterance
            return True
        return False
