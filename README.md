# Neural Network Pain Audio Recognition and Classification

## Overview

This project builds machine learning models to detect **pain from audio recordings**. Using mel spectrograms, we classify short human vocalizations as either **pain** or **no pain**.

---

## What This Project Does

* Converts raw audio into spectrograms
* Trains CNN and Transformer-based models
* Evaluates binary (pain vs no pain) classification
* Compares clean vs noisy data performance

---

## Full Repository Structure

```
.
├── cnns/
│   ├── best_pain_cnn.pt
│   ├── Model 1 - Clean Data, No Class Balancing.ipynb
│   └── Model 2 - Noisy + Clean Data, With Class Balancing.ipynb
│
├── Data/                          # MUST BE CREATED LOCALLY (not in repo)
│   ├── Raw-Data/
│   ├── all-pain/
│   │   ├── eda/
│   │   ├── npy/
│   │   │   ├── no-pain/
│   │   │   └── pain/
│   │   └── png/
│   │       ├── no-pain/
│   │       └── pain/
│   │
│   ├── clean-pain-only/
│   │   ├── metadata/
│   │   ├── npy/
│   │   │   ├── no-pain/
│   │   │   └── pain/
│   │   └── png/
│   │       ├── no-pain/
│   │       └── pain/
│   │
│   └── Raw-Data/
│
├── Data-Conversion/              # preprocessing scripts
│
├── TRANSFER LEARNING MODEL/
│   ├── All Categories/
│   └── Binary Model/
│
├── README.md
├── LICENSE
└── .gitignore
```

---

## Setup (START HERE)

### 1. Clone Repo

```
git clone <your-repo-url>
cd <repo-name>
```

---

### 2. Create Virtual Environment

```
python -m venv venv
```

Activate it:

**Windows**

```
venv\Scripts\activate
```

---

### 3. Install Dependencies

```
pip install torch librosa numpy matplotlib scikit-learn transformers
```

---

## Data Setup (CRITICAL)

The dataset is NOT included (~2.7GB).

### Step 1 — Download Dataset

Download from Kaggle:

https://www.kaggle.com/datasets/dejolilandry/asvpesdspeech-nonspeech-emotional-utterances

---

### Step 2 — Create Data Folder

Create this EXACT structure:

```
Data/
├── Raw-Data/
├── all-pain/
│   ├── eda/
│   ├── npy/
│   │   ├── no-pain/
│   │   └── pain/
│   └── png/
│       ├── no-pain/
│       └── pain/
└── clean-pain-only/
    ├── metadata/
    ├── npy/
    │   ├── no-pain/
    │   └── pain/
    └── png/
        ├── no-pain/
        └── pain/
```

---

### Step 3 — Add Raw Data

Put ALL downloaded files into:

```
Data/Raw-Data/
```

---

### Step 4 — Run Preprocessing

```
cd Data-Conversion
python your_conversion_script.py
```

This will generate:

```
Data/clean-pain-only/npy/
Data/clean-pain-only/png/
Data/all-pain/npy/
Data/all-pain/png/
```

If these folders are empty, your preprocessing failed.

---

## How to Run Models

### Option 1 — Use Notebooks

Open:

```
cnns/
```

Run:

* Model 1 (clean data)
* Model 2 (noisy + clean)

---

### Option 2 — Load Trained Model

```
cnns/best_pain_cnn.pt
```

You can load it in PyTorch:

```
model.load_state_dict(torch.load("cnns/best_pain_cnn.pt"))
model.eval()
```

---

### Option 3 — Transfer Learning Models

Go to:

```
TRANSFER LEARNING MODEL/
```

Run:

* Binary model notebook
* All categories notebook

---

## Preprocessing Details

* Sample rate: 16,000 Hz
* Duration: 3 seconds
* FFT: 1024
* Hop length: 512
* Mel bins: 128

Output:

```
(128, 94)
```

Model input:

```
(N, 1, 128, 94)
```

---

## Models

### PainCNN (Baseline)

CNN trained from scratch:

```
Conv → BN → ReLU → Pool → Dropout
Conv → BN → ReLU → Pool → Dropout
Conv → BN → ReLU → Pool → Dropout
Flatten → FC → Output
```

---

### Transfer Learning (AST)

Model:

```
MIT/ast-finetuned-audioset-10-10-0.4593
```

Used for:

* Binary classification
* Multiclass classification

---

## Results

### Binary Model

* Accuracy: ~0.542 → ~0.738
* F1: ~0.276 → ~0.699

### Multiclass Model

* Accuracy: ~0.552
* F1: ~0.539

Conclusion:

Binary classification works significantly better.

---

## Limitations

* Only ~704 pain samples
* Strong class imbalance
* Acted dataset (not real clinical data)
* Poor real-world noise handling
* No temporal modeling
* No demographic balancing

---

## Important Notes

* Data folder is NOT included
* Do NOT commit dataset
* Folder structure MUST match exactly
* If scripts fail → check paths first

---

## Authors

Jacky Lin
Eduardo Torres

---

## Acknowledgements

Dataset: ASVP-ESD (Kaggle)
Model: MIT AST (Hugging Face)

---

## AI Usage

AI was used for:

* debugging
* code clarity
* documentation improvements

All core work was completed by the authors.
