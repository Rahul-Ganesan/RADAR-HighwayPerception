import numpy as np


class ConditionClassifier:
    def __init__(self, labels=None):
        self.labels = labels or ["clear", "rain", "fog", "night", "glare"]

    def predict(self, frame):
        brightness = float(frame.mean() / 255.0)
        contrast = float(frame.std() / 128.0)
        scores = {
            "night": max(0.0, 0.7 - brightness),
            "fog": max(0.0, 0.8 - contrast),
            "glare": max(0.0, brightness - 0.75),
            "rain": max(0.0, 0.5 - contrast * 0.6),
            "clear": 0.4 + brightness * 0.2 + contrast * 0.2,
        }
        total = sum(scores.values()) + 1e-6
        return {k: v / total for k, v in scores.items()}

    def top_label(self, frame):
        probs = self.predict(frame)
        return max(probs, key=probs.get)
