##Real-Time Multimodal Gait and Voice Identification

## Overview

GVI (Gait and Voice Identification) is a real-time multimodal biometric framework that integrates skeleton-based gait recognition and voice recognition to enhance robustness in real-world environments. 
 - The system is designed to overcome unimodal limitations, such as gait occlusion and acoustic noise, through an adaptive reliability-driven approach.
 - Hardware: Operates on consumer-grade hardware (standard webcam and microphone) without specialized acceleration.
 - Fusion Strategy: Employs an adaptive multiplicative fusion mechanism at the decision level to maintain stability under degraded conditions.
 - Performance: Achieves a real-time identification accuracy of 91.3%.
---

## System Architecture

The framework consists of two parallel unimodal pipelines:

### 1. Gait Recognition Pipeline

- Real-time skeleton extraction using MediaPipe
- Representation: 33 keypoints (flattened into 66-dimensional vectors)
- Modeling: Sliding window of 30 frames with a step of 10.
- Classifier: Two-layer Bidirectional LSTM with a custom Temporal Attention mechanism

---

### 2. Voice Recognition Pipeline

- Audio recording at 16 kHz
- MFCC feature extraction (40 MFCC + delta + delta-delta = 120 features)
- Temporal modeling using LSTM
- Softmax-based confidence estimation

---

## Decision-Level Fusion

The final identity prediction is determined using an **Adaptive Multiplicative Fusion** strategy:
  p_"fused" =(p_v⊙p_g)/(∥p_v⊙p_g ∥_1+ϵ)

- If voice confidence is reliable → prioritize voice prediction  
- Otherwise → fallback to gait recognition  
- Product-rule fusion is applied to combine probability distributions  
- Gating: Acts as a dynamic gating mechanism where one modality's confidence scales the other's influence.
- Stability: Applies majority-vote smoothing over a 10-frame sliding window.
- Threshold: A confidence threshold of 0.6 is required; otherwise, the identity is labeled as "UNKNOWN".
This approach improves system robustness under:
- Noisy audio environments  
- Partial body occlusion  
- Lighting variations  

---

## Dataset: GVI Dataset

A small-scale multimodal dataset collected for real-time identification research.

### Dataset Characteristics

- 8 subjects (labeled Rc1–Rc8)
- Multiple recording sessions per subject  
- Audio recorded at 16 kHz  
- Skeleton sequences extracted from video and webcam
- Includes environmental variations:
  - Background noise  
  - Lighting changes  
  - Movement variations  

> Due to privacy and biometric data considerations, the dataset is **not publicly released**.

---

## Models

### Core Identification Models

- **Gait Recognition**: Skeleton + LSTM + Attention (TensorFlow/Keras)  
- **Speaker Identification**: MFCC + LSTM (TensorFlow/Keras)  

---

## Installation

```bash
pip install -r requirements.txt
