# HUIT-GVI: Real-Time Multimodal Gait and Voice Identification

## Overview

HUIT-GVI (Gait and Voice Identification) is a real-time multimodal biometric framework that integrates **gait (skeleton-based recognition)** and **voice recognition** to enhance robustness in real-world deployment scenarios.

The system operates using consumer-grade hardware, including a standard webcam and microphone, without requiring specialized devices.

Two independently trained unimodal models are combined through **decision-level fusion**, allowing the system to maintain stable performance when one modality is degraded due to environmental conditions such as noise or occlusion.

---

## System Architecture

The framework consists of two parallel unimodal pipelines:

### 1. Gait Recognition Pipeline

- Real-time skeleton extraction using MediaPipe
- 33 keypoints representation (x, y coordinates)
- Temporal sequence modeling (30 frames)
- LSTM + Attention-based classifier
- Softmax confidence output

---

### 2. Voice Recognition Pipeline

- Audio recording at 16 kHz
- MFCC feature extraction (40 MFCC + delta + delta-delta = 120 features)
- Temporal modeling using LSTM
- Softmax-based confidence estimation

---

## Decision-Level Fusion

The final identity prediction is determined using a **confidence-based fusion strategy**:

- If voice confidence is reliable → prioritize voice prediction  
- Otherwise → fallback to gait recognition  
- Product-rule fusion is applied to combine probability distributions  

This approach improves system robustness under:
- Noisy audio environments  
- Partial body occlusion  
- Lighting variations  

---

## Dataset: HUIT-GVI Dataset

A small-scale multimodal dataset collected for real-time identification research.

### Dataset Characteristics

- 3–8 enrolled subjects  
- Multiple recording sessions per subject  
- Audio recorded at 16 kHz  
- Skeleton sequences extracted from video  
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

### Auxiliary Modules (Optional)

- Whisper (Speech-to-Text)  
- PhoBERT (Vietnamese NLP)  

> Auxiliary modules are used for extended functionality and do **not affect identity prediction**.

---

## Installation

```bash
pip install -r requirements.txt