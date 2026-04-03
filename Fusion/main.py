import os, threading, time, numpy as np, tensorflow as tf
import librosa, sounddevice as sd, cv2, mediapipe as mp
from collections import deque

# ===================== CUSTOM LAYER =====================
try:
    reg = tf.keras.saving.register_keras_serializable
except:
    reg = tf.keras.utils.register_keras_serializable

@reg()
class TemporalAttention(tf.keras.layers.Layer):
    def build(self, s):
        self.W = self.add_weight(shape=(int(s[-1]),1))
        self.b = self.add_weight(shape=(int(s[1]),1))
    def call(self,x):
        e=tf.keras.backend.tanh(tf.keras.backend.dot(x,self.W)+self.b)
        a=tf.keras.backend.softmax(e,axis=1)
        return tf.keras.backend.sum(x*a,axis=1)

# ===================== LOAD =====================
BASE=os.path.dirname(os.path.abspath(__file__))

speaker_model=tf.keras.models.load_model(
    os.path.join(BASE,"final_speaker_model_full.keras"),
    compile=False
)

pose_model=tf.keras.models.load_model(
    os.path.join(BASE,"best_person_id_v3.keras"),
    custom_objects={'TemporalAttention':TemporalAttention},
    compile=False
)

CLASSES=["Khoa","Tai","Don","Rc4","Rc5","Rc6","Rc7","Rc8"]
SR=16000

state={
    "voice_prob":None,
    "pose_prob":None,
    "voice_label":"--",
    "pose_label":"--",
    "final_label":"UNKNOWN",
    "final_conf":0.0
}

lock=threading.Lock()
event=threading.Event()

# ===================== VOICE =====================
def clean_audio(x):
    x=x/(np.max(np.abs(x))+1e-6)
    return librosa.effects.trim(x)[0]

def extract_voice(x):
    mfcc=librosa.feature.mfcc(y=x,sr=SR,n_mfcc=40).T
    d1=librosa.feature.delta(mfcc.T).T
    d2=librosa.feature.delta(mfcc.T,order=2).T
    f=np.hstack([mfcc,d1,d2])
    f=(f-f.mean(0))/(f.std(0)+1e-8)

    if len(f)<150:
        p=np.zeros((150,120)); p[:len(f)]=f
        return p
    return f[:150]

def predict_voice(x):
    x=clean_audio(x)
    segs=np.array_split(x,3)

    probs=[
        speaker_model.predict(np.expand_dims(extract_voice(s),0),verbose=0)[0]
        for s in segs
    ]
    return np.mean(probs,0)

def audio_loop():
    while True:
        event.wait()
        event.clear()

        audio=sd.rec(int(5*SR),samplerate=SR,channels=1)
        sd.wait()
        audio=audio.flatten()

        prob=predict_voice(audio)
        idx=np.argmax(prob)

        with lock:
            state["voice_prob"]=prob
            state["voice_label"]=CLASSES[idx]

# ===================== POSE =====================
mp_pose=mp.solutions.pose
pose=mp_pose.Pose()

LEFT,RIGHT,NOSE=23,24,0

def preprocess(seq):
    seq=np.array(seq)

    for i in range(1,len(seq)):
        seq[i]=0.7*seq[i]+0.3*seq[i-1]

    for t in range(len(seq)):
        hip=(seq[t][LEFT]+seq[t][RIGHT])/2
        seq[t]-=hip
        seq[t]/=(np.linalg.norm(seq[t][NOSE])+1e-6)

    seq=seq.reshape(len(seq),-1)

    if len(seq)<30:
        p=np.zeros((30,66))
        p[:len(seq)]=seq
        return np.expand_dims(p,0)

    return np.expand_dims(seq[:30],0)

def predict_pose(seq):
    return pose_model.predict(preprocess(seq),verbose=0)[0]

# ===================== SKELETON DRAW =====================
def draw_skeleton(frame, landmarks):
    h,w=frame.shape[:2]
    VIS_TH=0.5

    # connections
    for p1,p2 in mp_pose.POSE_CONNECTIONS:
        lm1,lm2=landmarks[p1],landmarks[p2]

        if lm1.visibility<VIS_TH or lm2.visibility<VIS_TH:
            continue

        x1,y1=int(lm1.x*w),int(lm1.y*h)
        x2,y2=int(lm2.x*w),int(lm2.y*h)

        thickness=2+int(2*lm1.visibility)
        cv2.line(frame,(x1,y1),(x2,y2),(0,200,255),thickness)

    # joints
    for lm in landmarks:
        if lm.visibility<VIS_TH:
            continue

        x,y=int(lm.x*w),int(lm.y*h)
        cv2.circle(frame,(x,y),4,(0,255,100),-1)

# ===================== FUSION =====================
def fuse(v,p):
    if v is None:
        return p
    x=v*p
    return x/(np.sum(x)+1e-8)

history=deque(maxlen=10)

def smooth(label):
    history.append(label)
    return max(set(history),key=history.count)

# ===================== DRAW UI =====================
def draw(frame,state,fps):
    h,w=frame.shape[:2]

    cv2.putText(frame,f"Voice: {state['voice_label']}",(20,h-140),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

    cv2.putText(frame,f"Pose: {state['pose_label']}",(20,h-100),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,0),2)

    cv2.putText(frame,f"Final: {state['final_label']}",(20,h-60),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

    bar_w=500; x=400; y=h-80
    cv2.rectangle(frame,(x,y),(x+bar_w,y+20),(100,100,100),2)

    fill=int(bar_w*state["final_conf"])
    cv2.rectangle(frame,(x,y),(x+fill,y+20),(0,255,0),-1)

    cv2.putText(frame,f"{state['final_conf']*100:.1f}%",
                (x+bar_w+10,y+18),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

    cv2.putText(frame,f"FPS:{fps}",
                (w-150,h-20),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

    return frame

# ===================== VIDEO =====================
def video_loop():
    cap=cv2.VideoCapture(0)
    seq=[]
    prev=time.time()

    while True:
        ret,frame=cap.read()
        if not ret:
            break

        frame=cv2.flip(frame,1)

        res=pose.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

        if res.pose_landmarks:
            lm=res.pose_landmarks.landmark

            draw_skeleton(frame,lm)

            kp=np.array([(l.x,l.y) for l in lm])

            if len(seq)>0:
                kp=0.7*kp+0.3*np.array(seq[-1])

            seq.append(kp.tolist())

            if len(seq)>60:
                seq.pop(0)

            if len(seq)>=30:
                prob_p=predict_pose(seq)
                idx=np.argmax(prob_p)

                with lock:
                    state["pose_prob"]=prob_p
                    state["pose_label"]=CLASSES[idx]

        with lock:
            v,p=state["voice_prob"],state["pose_prob"]

        if p is not None:
            probs=fuse(v,p)
            idx=np.argmax(probs)

            raw_conf=probs[idx]
            conf=min(raw_conf*0.9,0.97)

            label=CLASSES[idx] if conf>=0.6 else "UNKNOWN"

            state["final_label"]=smooth(label)
            state["final_conf"]=conf

        fps=int(1/(time.time()-prev))
        prev=time.time()

        cv2.imshow("AI Fusion System",draw(frame,state,fps))

        k=cv2.waitKey(1)&0xFF
        if k==ord('v'):
            event.set()
        if k==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ===================== MAIN =====================
if __name__=="__main__":
    threading.Thread(target=audio_loop,daemon=True).start()
    video_loop()