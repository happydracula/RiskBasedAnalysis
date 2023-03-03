import os
import tensorflow as tf
import cv2
import numpy as np
from queue import Queue


class FightDetector:
    def __init__(self, fsize, Tx):
        self.framesQueue = Queue(maxsize=Tx)
        self.fsize = fsize
        self.Tx = Tx
        self.model = tf.keras.models.load_model('model.h5')

    def detect(self, frame):
        original_frame = np.array(frame)
        frame = cv2.resize(frame, (self.fsize, self.fsize))
        if(self.framesQueue.full()):
            frames = np.array([list(self.framesQueue.queue)])
            result = self.predict(frames)
            text = ""
            if(result == 1):
                text = "Fight Detected"
            else:
                text = "No Fight Detected"

            cv2.putText(original_frame, text, (100, 100),
                        cv2.FONT_HERSHEY_DUPLEX, color=(255, 0, 0))

        else:
            self.framesQueue.put(frame)
        return original_frame

    def predict(self, frames):
        res = self.model(frames)
        if(res >= 0.6):
            return 1
        else:
            return 0
