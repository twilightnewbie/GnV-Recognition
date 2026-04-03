import os, numpy as np, librosa, keras
from keras import layers
from sklearn.model_selection import train_test_split

# ===== CONFIG =====
DATASET = r"C:\Users\Tai\Desktop\nckh\GaitAndVoice\data\Train"
SR, MAX_LEN, NUM_CLASSES = 16000, 150, 8

# ===== FEATURE =====
def feat(path):
    y,_ = librosa.load(path, sr=SR)
    if len(y)==0: return None
    mfcc = librosa.feature.mfcc(y, sr=SR, n_mfcc=40).T
    d1 = librosa.feature.delta(mfcc.T).T
    d2 = librosa.feature.delta(mfcc.T, order=2).T
    x = np.hstack([mfcc,d1,d2])
    x = (x-x.mean(0))/(x.std(0)+1e-8)
    if len(x)<MAX_LEN:
        pad = np.zeros((MAX_LEN,x.shape[1])); pad[:len(x)]=x; return pad
    return x[:MAX_LEN]

# ===== LOAD =====
X,y=[],[]
classes=sorted([d for d in os.listdir(DATASET) if os.path.isdir(os.path.join(DATASET,d))])

for i,c in enumerate(classes):
    for f in os.listdir(os.path.join(DATASET,c)):
        if f.endswith(".wav"):
            x=feat(os.path.join(DATASET,c,f))
            if x is not None:
                X.append(x); y.append(i)

X,y=np.array(X),np.array(y)
y=keras.utils.to_categorical(y,NUM_CLASSES)

# ===== SPLIT =====
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,stratify=np.argmax(y,1))

# ===== MODEL =====
inp=layers.Input((MAX_LEN,X.shape[2]))
x=layers.LSTM(256,return_sequences=True,dropout=0.2)(inp)
x=layers.LSTM(256,return_sequences=True,dropout=0.2)(x)
x=layers.LSTM(128,return_sequences=True,dropout=0.2)(x)
x=layers.LSTM(128,return_sequences=True,dropout=0.2)(x)
x=layers.LSTM(64)(x)
x=layers.Dense(128,activation="relu")(x)
x=layers.Dense(64,activation="relu")(x)
out=layers.Dense(NUM_CLASSES,activation="softmax")(x)

model=keras.Model(inp,out)
model.compile(optimizer=keras.optimizers.Adam(5e-4),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# ===== TRAIN =====
model.fit(Xtr,ytr,validation_data=(Xte,yte),epochs=150,batch_size=16)

# ===== SAVE =====
model.save("voice.keras")