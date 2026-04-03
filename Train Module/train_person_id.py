import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras import layers, callbacks, Model
from keras.regularizers import l2

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE

import seaborn as sns


# ===================== CONFIG =====================

PKL_PATH = r"C:\Users\Tai\Desktop\nckh\GaitAndVoice\converted_dataset.pkl"

SEQ_LEN = 30
STEP = 10
EPOCHS = 150
BATCH = 32
LR = 5e-4

# Augmentation
AUG_MULTIPLIER = 3
AUG_SCALE_RANGE = (0.85, 1.15)
AUG_NOISE_STD = 0.005
AUG_MIRROR_PROB = 0.5
AUG_TIME_MASK = 3

LEFT_HIP = 23
RIGHT_HIP = 24
NOSE = 0


# ===================== LOAD =====================

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ===================== NORMALIZE =====================

def normalize_hip(kp):
    kp = kp.copy()

    for t in range(len(kp)):
        hip = (kp[t, LEFT_HIP] + kp[t, RIGHT_HIP]) / 2
        kp[t] -= hip

        torso = np.linalg.norm(kp[t, NOSE]) + 1e-6
        kp[t] /= torso

    return kp


# ===================== AUGMENT =====================

def augment(seq):
    T, D = seq.shape
    seq = seq.reshape(T, 33, 2)

    # scale
    scale = np.random.uniform(*AUG_SCALE_RANGE)
    seq *= scale

    # noise
    seq += np.random.normal(0, AUG_NOISE_STD, seq.shape)

    # mirror
    if np.random.rand() < AUG_MIRROR_PROB:
        seq[:, :, 0] *= -1

    # time mask
    if AUG_TIME_MASK > 0:
        length = np.random.randint(1, AUG_TIME_MASK + 1)
        start = np.random.randint(0, max(1, T - length))
        seq[start:start + length] = 0

    return seq.reshape(T, D)


def augment_data(X, y):
    X_all, y_all = [X], [y]

    for _ in range(AUG_MULTIPLIER):
        batch = np.array([augment(s) for s in X], dtype=np.float32)
        X_all.append(batch)
        y_all.append(y)

    return np.concatenate(X_all), np.concatenate(y_all)


# ===================== SEQUENCE =====================

def extract_sequences(annotations):
    X, y = [], []

    for sample in annotations:
        kp = sample["keypoint"][0]
        label = sample["label"]

        if len(kp) == 0:
            continue

        kp = normalize_hip(kp)
        kp = kp.reshape(len(kp), -1)

        for i in range(0, len(kp) - SEQ_LEN, STEP):
            X.append(kp[i:i + SEQ_LEN])
            y.append(label)

    return np.array(X, np.float32), np.array(y)


# ===================== MODEL =====================

class TemporalAttention(layers.Layer):
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1))
        self.b = self.add_weight(shape=(input_shape[1], 1))

    def call(self, x):
        score = keras.ops.tanh(keras.ops.matmul(x, self.W) + self.b)
        weight = keras.ops.softmax(score, axis=1)
        return keras.ops.sum(x * weight, axis=1)


def build_model(input_dim, num_classes):
    inp = layers.Input(shape=(SEQ_LEN, input_dim))
    x = layers.BatchNormalization()(inp)

    x = layers.Bidirectional(layers.LSTM(
        64, return_sequences=True,
        kernel_regularizer=l2(1e-4),
        dropout=0.3, recurrent_dropout=0.2
    ))(x)

    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Bidirectional(layers.LSTM(
        32, return_sequences=True,
        kernel_regularizer=l2(1e-4),
        dropout=0.2, recurrent_dropout=0.2
    ))(x)

    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = TemporalAttention()(x)

    x = layers.Dense(64, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)

    emb = layers.Dense(32, activation="relu", name="embedding")(x)
    x = layers.Dropout(0.3)(emb)

    out = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inp, out)


# ===================== MAIN =====================

if __name__ == "__main__":

    print("Loading dataset...")
    data = load_pkl(PKL_PATH)

    X_raw, y_raw = extract_sequences(data["annotations"])

    num_classes = len(set(y_raw))
    print("Classes:", num_classes)

    X_train_raw, X_val, y_train_raw, y_val = train_test_split(
        X_raw, y_raw, test_size=0.2, stratify=y_raw
    )

    print("Augmenting...")
    X_train, y_train = augment_data(X_train_raw, y_train_raw)

    # shuffle
    idx = np.random.permutation(len(X_train))
    X_train, y_train = X_train[idx], y_train[idx]

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)

    model = build_model(X_raw.shape[2], num_classes)

    model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    cb = [
        callbacks.EarlyStopping(patience=20, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(patience=8),
        callbacks.ModelCheckpoint("final_speaker_model_full.keras", save_best_only=True)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH,
        callbacks=cb
    )

    # ================= EVAL =================

    y_score = model.predict(X_val)
    y_pred = np.argmax(y_score, axis=1)
    y_true = np.argmax(y_val, axis=1)

    print(classification_report(y_true, y_pred))

    print("Saving model...")
    model.save("final_speaker_model_full.keras")