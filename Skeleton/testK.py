import os
import sys
import time
import pickle
from dataclasses import dataclass, field
from collections import Counter

import cv2
import keras
import mediapipe as mp
import numpy as np


# ===================== CUSTOM LAYER =====================

class TemporalAttention(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True,
        )

    def call(self, x):
        import tensorflow as tf
        score = tf.math.tanh(tf.matmul(x, self.W) + self.b)
        weight = tf.nn.softmax(score, axis=1)
        return tf.reduce_sum(x * weight, axis=1)

    def get_config(self):
        return super().get_config()


CUSTOM_OBJECTS = {"TemporalAttention": TemporalAttention}


# ===================== CONFIG =====================

@dataclass
class Config:
    model_path: str = "model.keras"
    pkl_path: str = "data.pkl"

    seq_len: int = 30
    step: int = 10
    num_joints: int = 33
    coords: int = 2
    input_dim: int = 66
    num_classes: int = 8
    min_confidence: float = 0.6

    left_hip_idx: int = 23
    right_hip_idx: int = 24
    nose_idx: int = 0

    predict_interval: float = 0.3
    max_history: int = 200

    class_names: dict = field(default_factory=lambda: {
        i: f"Person {i}" for i in range(8)
    })


# ===================== NORMALIZE =====================

def normalize_frame(kp, cfg):
    kp = kp.copy()
    hip = (kp[cfg.left_hip_idx] + kp[cfg.right_hip_idx]) / 2
    kp -= hip
    torso = np.linalg.norm(kp[cfg.nose_idx]) + 1e-6
    kp /= torso
    return kp


def normalize_sequence(seq, cfg):
    return np.array([normalize_frame(f, cfg) for f in seq])


# ===================== MODEL =====================

def load_model(cfg):
    if not os.path.exists(cfg.model_path):
        print("Không tìm thấy model")
        sys.exit()

    model = keras.models.load_model(
        cfg.model_path,
        custom_objects=CUSTOM_OBJECTS,
        compile=False
    )

    _, cfg.seq_len, cfg.input_dim = model.input_shape
    cfg.num_classes = model.output_shape[-1]
    cfg.coords = cfg.input_dim // cfg.num_joints

    return model


# ===================== POSE =====================

class PoseExtractor:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose()

    def extract(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)

        if not res.pose_landmarks:
            return None, res

        kp = np.array([[lm.x, lm.y] for lm in res.pose_landmarks.landmark])
        return kp, res


# ===================== ENGINE =====================

class Engine:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.buffer = []
        self.history = []

    def add(self, kp):
        kp = normalize_frame(kp, self.cfg)
        self.buffer.append(kp.reshape(-1))

        if len(self.buffer) > self.cfg.seq_len * 2:
            self.buffer = self.buffer[-self.cfg.seq_len:]

    def ready(self):
        return len(self.buffer) >= self.cfg.seq_len

    def predict_raw(self):
        seq = np.array(self.buffer[-self.cfg.seq_len:])
        seq = seq.reshape(1, self.cfg.seq_len, self.cfg.input_dim)

        probs = self.model.predict(seq, verbose=0)[0]
        cls = int(np.argmax(probs))
        conf = float(np.max(probs))

        self.history.append((cls, conf, probs))
        if len(self.history) > self.cfg.max_history:
            self.history.pop(0)

        return cls, conf, probs

    def predict_smooth(self, window=5):
        cls, conf, probs = self.predict_raw()

        if len(self.history) < window:
            return cls, conf, probs

        recent = self.history[-window:]
        classes = [r[0] for r in recent]

        best = Counter(classes).most_common(1)[0][0]
        avg_conf = np.mean([r[1] for r in recent if r[0] == best])

        return best, avg_conf, probs


# ===================== REALTIME =====================

def run_webcam(model, cfg):
    cap = cv2.VideoCapture(0)
    pose = PoseExtractor()
    engine = Engine(model, cfg)

    last = 0
    name = "..."
    conf = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        kp, res = pose.extract(frame)

        if kp is not None:
            engine.add(kp)

        if engine.ready() and time.time() - last > cfg.predict_interval:
            cls, conf, _ = engine.predict_smooth()
            name = cfg.class_names.get(cls, str(cls))
            last = time.time()

        cv2.putText(frame, f"{name} {conf:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Gait", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# ===================== VIDEO =====================

def run_video(model, cfg, path):
    cap = cv2.VideoCapture(path)
    pose = PoseExtractor()
    engine = Engine(model, cfg)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        kp, _ = pose.extract(frame)
        if kp is not None:
            engine.add(kp)

        if engine.ready():
            cls, conf, _ = engine.predict_smooth()
            name = cfg.class_names.get(cls, str(cls))

            cv2.putText(frame, f"{name} {conf:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# ===================== PKL =====================

def run_pkl(model, cfg, path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    total = 0
    correct = 0

    for sample in data["annotations"]:
        label = sample["label"]
        kp = sample["keypoint"][0]

        kp = normalize_sequence(kp, cfg)
        kp = kp.reshape(kp.shape[0], -1)

        for i in range(0, len(kp) - cfg.seq_len, cfg.step):
            seq = kp[i:i+cfg.seq_len]
            seq = seq.reshape(1, cfg.seq_len, cfg.input_dim)

            probs = model.predict(seq, verbose=0)[0]
            pred = np.argmax(probs)

            total += 1
            if pred == label:
                correct += 1

    print("Accuracy:", correct / total if total else 0)


# ===================== MAIN =====================

def main():
    cfg = Config()
    model = load_model(cfg)

    while True:
        print("\n1. Webcam")
        print("2. Video")
        print("3. PKL")
        print("4. Exit")

        c = input(">> ")

        if c == "1":
            run_webcam(model, cfg)

        elif c == "2":
            path = input("Video path: ")
            run_video(model, cfg, path)

        elif c == "3":
            path = input("PKL path: ")
            run_pkl(model, cfg, path)

        elif c == "4":
            break


if __name__ == "__main__":
    main()