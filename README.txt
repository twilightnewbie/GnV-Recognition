# HUIT-RTID: Real-Time Multimodal Gait and Voice Identification System

## Overview
HUIT-RTID is a lightweight, real-time multimodal biometric framework designed to identify individuals by integrating skeleton-based gait patterns and voice signatures. The system utilizes the HUIT-MIFNet architecture to adaptively combine modality outputs, overcoming common unimodal failures such as visual occlusion or acoustic noise.

* Proposed Method: HUIT-MIFNet (HUIT Multimodal Identification Fusion Network).
* Dataset: HUIT-MGV Dataset (HUIT Multimodal Gait-Voice Dataset).
* System Core: HUIT-RTID System (HUIT Real-Time Identification System).
* Performance: Achieves 91.3% real-time accuracy and maintains 10 FPS on consumer-grade CPUs.

---

##  System Architecture
The HUIT-RTID system operates through two parallel, independent pipelines integrated at the decision level:

### 1. Gait Recognition Pipeline
* Skeleton Extraction: Real-time pose estimation using MediaPipe.
* Feature Representation: 33 skeletal keypoints flattened into 66-dimensional vectors.
* Temporal Modeling: Processes sequences using a sliding window of 30 frames with a step of 10.
* Classifier: A Bidirectional LSTM enhanced with a custom Temporal Attention mechanism to focus on discriminative gait frames.

### 2. Voice Recognition Pipeline
* Audio Capture: 16 kHz sampling rate via consumer-grade microphones.
* Feature Extraction: 120-dimensional vectors comprising 40 MFCCs plus their first- and second-order delta derivatives.
* Classifier: A deep 5-layer Stacked LSTM designed to capture long-range acoustic characteristics.

---

##  HUIT-MIFNet: Adaptive Fusion Logic
The HUIT-MIFNet architecture performs decision-level fusion using an Adaptive Multiplicative Fusion strategy:

  p_"fused" =(p_v⊙p_g)/(∥p_v⊙p_g ∥_1+ϵ)

* Dynamic Gating: Acts as a gating mechanism where the confidence of one modality naturally scales the influence of the other.
* Robust Prioritization: Prioritizes the more reliable modality during inference (e.g., fallback to gait if audio is noisy).
* Temporal Smoothing: Stabilizes predictions using majority-vote smoothing over the 10 most recent results.
* Security Threshold: A confidence threshold of 0.6 is applied; predictions below this are labeled as "UNKNOWN" to prevent false positives.

---

##  HUIT-MGV Dataset
The HUIT-MGV Dataset : https://drive.google.com/drive/folders/1A5OYycscantQX-BD2GiaGgoZuB2uD3nL?usp=sharing

* Subjects: 8 enrolled identities (Rc1–Rc8).
* Environmental Variability: Includes variations in lighting, background noise, and partial body occlusion.
* Privacy Note: Due to the sensitive nature of biometric data, this dataset is not publicly released.

---

##  Experimental Results
| Method             | Accuracy (%)   | Hardware            | FPS          |
| :---               | :---           | :---                | :---         |
| Voice-only         | 95.42%         | Consumer-grade CPU  | Real-time    |
| Gait-only          | 85–92%         | Consumer-grade CPU  | Real-time    |
| HUIT-RTID (Fusion) | 91.30%         | Consumer-grade CPU  | Real-time    |

* HUIT-RTID significantly outperforms traditional score-averaging fusion (78.6%) under degraded conditions.

---

## 🛠 Installation
1.  Clone the Repository:
    ```bash
    git clone https://github.com/twilightnewbie/GnV-Recognition.git
    cd GnV-Recognition
    ```
2.  Install Dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the System:
    ```bash
    python main.py
    ```

---

## 🔗 Citation
If you use HUIT-MIFNet or the HUIT-RTID System in your research, please cite:
```text
Tô Duy Tài, Đỗ Minh Tiến, Phan Văn Khải, Vũ Đức Thịnh. (2026). A REAL-TIME MULTIMODAL PERSON IDENTIFICATION SYSTEM BASED ON VOICE AND GAIT FUSION. Faculty of Information Technology, Ho Chi Minh City University of Industry and Trade (HUIT).
```
