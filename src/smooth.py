from collections import deque
import numpy as np

class EmotionSmoother:
    def __init__(self, size=10):
        self.buffer = deque(maxlen=size)

    def smooth(self, preds):
        self.buffer.append(preds)
        avg = np.mean(self.buffer, axis=0)
        return avg.argmax()
