import os
import time
import queue
import numpy as np
import librosa
import sounddevice as sd
from dataclasses import dataclass, field

import keras
from keras import Model


# ===================== CONFIG =====================

@dataclass
class Config:
    voice_model_path: str = "voice.keras"
    gait_model_path: str = "gait.keras"
    fusion_model_path: str = "fusion.keras"
    dataset_dir: str = "Train"

    sr: int = 16000
    max_len: int = 150
    n_mfcc: int = 40
    n_features: int = 120

    chunk_duration: float = 2.5
    min_conf: float = 0.7
    silence_threshold: float = 0.02
    min_speech: float = 2.0

    num_classes: int = 8


cfg = Config()


# ===================== MODEL LOADER =====================

def load_model(path):
    if not os.path.exists(path):
        return None
    return keras.models.load_model(path, compile=False)


def load_all():
    voice = load_model(cfg.voice_model_path)
    gait = load_model(cfg.gait_model_path)
    fusion = load_model(cfg.fusion_model_path)
    return voice, gait, fusion


# ===================== FEATURE =====================

def extract_mfcc(audio):
    if len(audio) < cfg.sr * 0.5:
        return None

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=cfg.sr,
        n_mfcc=cfg.n_mfcc
    ).T

    if cfg.n_features == 120:
        d1 = librosa.feature.delta(mfcc.T).T
        d2 = librosa.feature.delta(mfcc.T, order=2).T
        mfcc = np.hstack([mfcc, d1, d2])

    mfcc = (mfcc - mfcc.mean(0)) / (mfcc.std(0) + 1e-8)

    if len(mfcc) < cfg.max_len:
        mfcc = np.pad(mfcc, ((0, cfg.max_len - len(mfcc)), (0, 0)))
    else:
        mfcc = mfcc[:cfg.max_len]

    return mfcc.reshape(1, cfg.max_len, -1)


# ===================== ENGINE =====================

class Engine:
    def __init__(self, voice_model, speakers):
        self.model = voice_model
        self.speakers = speakers
        self.history = []

    def predict(self, audio):
        feat = extract_mfcc(audio)
        if feat is None:
            return None, 0, None

        probs = self.model.predict(feat, verbose=0)[0]
        idx = int(np.argmax(probs))
        conf = float(np.max(probs))

        name = self.speakers.get(idx, f"Rc{idx}")
        return name, conf, probs


# ===================== AUDIO STREAM =====================

class AudioStream:
    def __init__(self, engine):
        self.engine = engine
        self.q = queue.Queue()
        self.buffer = []
        self.running = False

    def callback(self, indata, frames, time_info, status):
        self.q.put(indata[:, 0].copy())

    def start(self):
        self.running = True
        chunk = int(cfg.sr * cfg.chunk_duration)

        stream = sd.InputStream(
            callback=self.callback,
            channels=1,
            samplerate=cfg.sr,
            blocksize=chunk
        )

        stream.start()

        try:
            while self.running:
                audio = self.q.get()

                rms = np.sqrt(np.mean(audio ** 2))

                if rms > cfg.silence_threshold:
                    self.buffer.append(audio)

                    total = len(np.concatenate(self.buffer)) / cfg.sr

                    if total >= cfg.min_speech:
                        full = np.concatenate(self.buffer)
                        name, conf, _ = self.engine.predict(full)

                        if name:
                            print(f"\n>>> {name} ({conf:.2%})")

                        self.buffer = []

                else:
                    self.buffer = []

        except KeyboardInterrupt:
            print("Stop")
        finally:
            stream.stop()
            stream.close()


# ===================== SPEAKERS =====================

def load_speakers():
    speakers = {}
    if os.path.exists(cfg.dataset_dir):
        dirs = sorted(os.listdir(cfg.dataset_dir))
        for i, d in enumerate(dirs):
            speakers[i] = d
    else:
        for i in range(cfg.num_classes):
            speakers[i] = f"Rc{i}"
    return speakers


# ===================== MAIN =====================

def main():
    voice, _, _ = load_all()

    if voice is None:
        print("No voice model")
        return

    speakers = load_speakers()
    engine = Engine(voice, speakers)

    print("Ready...")

    stream = AudioStream(engine)
    stream.start()


if __name__ == "__main__":
    main()