# DS340 Final Project
# Baymax 0.1.0

**Eduardo Torres & Jacky Lin** — DS340 Final Project

A neural network system for detecting short human distress-related vocalizations (e.g., "ouch", "ahh") in audio clips, even in the presence of background noise.

---

## Overview

This project is inspired by wake-word detection systems, but instead of detecting a fixed keyword, it focuses on identifying pain-related vocalizations — sounds that are more variable and less structured than standard keywords. These vocalizations differ significantly across speakers and environments, making the task more challenging. The goal is to explore how well machine learning models can generalize across different types of human vocal expressions and noisy conditions.

---

## Task Setup

Binary audio classification:
- **Class 1:** Pain-related vocalizations
- **Class 2:** Non-pain audio (normal speech, silence, background noise)

---

## Data

**Primary dataset:**
> Landry, Touko. "ASVP-ESD (Speech & Non-Speech Emotional Sound)." Kaggle.com, 2022. [Link](https://www.kaggle.com/datasets/dejolilandry/asvpesdspeech-nonspeech-emotional-utterances)

The dataset contains audio files in mixed-voice, mixed-volume environments, categorized as:

| Emotion   | Sub-categories |
|-----------|----------------|
| Happiness | laugh, gaggle, others |
| Sadness   | cry, sigh, sniffle, suffering |
| Fear      | scream, panic |
| Angry     | rage, frustration, grunt, other |
| Surprise  | surprised, amazed, astonishment, others |
| Disgust   | disgust, rejection |
| Pain      | moaned |
| Boredom   | sigh |

Supplemental data may also be collected manually.

---

## Methods

### Preprocessing
Raw audio signals are converted into numerical representations:
- **Spectrograms**
- **MFCC** (Mel-frequency cepstral coefficients)

### Models
- **CNNs** applied to spectrogram images
- **Feedforward / recurrent neural networks** applied to MFCC features

---

## Experiments

Four aspects of the system are varied and evaluated:

1. **Feature Representation** — MFCC vs. spectrogram
2. **Noise Conditions** — Clean audio vs. audio with background noise
3. **Model Architecture** — CNN vs. simpler baseline models
4. **Audio Clip Length** — Short vs. longer audio windows

Each variation is evaluated using consistent metrics for fair comparison.

---

## Evaluation

**Metrics:**
- Accuracy
- Precision and Recall

**Demonstration:** The model processes an input audio clip and outputs the probability that it contains a pain-related vocalization.

---

## Tech Stack

- Python
- PyTorch or TensorFlow
- NumPy
- Librosa

---

## Project Timeline

| Week | Goals |
|------|-------|
| Week 1 | Collect and organize dataset; implement preprocessing pipeline |
| Week 2 | Train baseline model (CNN on spectrograms) |
| Week 3 *(Milestone)* | Compare MFCC-based and spectrogram-based models; present initial results to TA |
| Week 4 | Run experiments (noise levels, model types, clip lengths) |
| Week 5 | Final evaluation; prepare paper and presentation |

---

## Contributions

- Custom dataset design and collection of pain-related vocalizations
- Implementation and comparison of multiple audio feature extraction techniques
- Evaluation of model performance under different noise conditions
- Systematic analysis of performance across architectures and preprocessing methods