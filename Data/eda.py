import os
import numpy as np
import matplotlib.pyplot as plt

PAIN_DIR = "Data/clean-pain-only/npy/pain"
NO_PAIN_DIR = "Data/clean-pain-only/npy/no-pain"


# =========================
# 1. COUNT FILES
# =========================
def count_files():
    pain_files = [f for f in os.listdir(PAIN_DIR) if f.endswith(".npy")]
    no_pain_files = [f for f in os.listdir(NO_PAIN_DIR) if f.endswith(".npy")]

    print("\n=== DATASET SIZE ===")
    print(f"Pain samples: {len(pain_files)}")
    print(f"No-pain samples: {len(no_pain_files)}")
    print(f"Total samples: {len(pain_files) + len(no_pain_files)}")

    return pain_files, no_pain_files


# =========================
# 2. LOAD RANDOM SAMPLES
# =========================
def load_random_samples(folder, files, n=5):
    chosen = np.random.choice(files, min(n, len(files)), replace=False)
    return [np.load(os.path.join(folder, f)) for f in chosen]


# =========================
# 3. BASIC STATS
# =========================
def basic_stats(samples, name):
    print(f"\n=== {name} STATS ===")
    for i, s in enumerate(samples):
        print(
            f"Sample {i}: shape={s.shape}, "
            f"min={s.min():.2f}, max={s.max():.2f}, "
            f"mean={s.mean():.2f}, std={s.std():.2f}"
        )


# =========================
# 4. SHAPE CONSISTENCY
# =========================
def check_shapes(folder, files):
    shapes = set()

    for f in files[:50]:  # check first 50 (fast)
        arr = np.load(os.path.join(folder, f))
        shapes.add(arr.shape)

    print(f"\nUnique shapes in {folder}: {shapes}")


# =========================
# 5. PLOT SAMPLES
# =========================
def plot_samples(samples, title):
    plt.figure(figsize=(12, 4))
    for i, s in enumerate(samples):
        plt.subplot(1, len(samples), i + 1)
        plt.imshow(s, aspect='auto', origin='lower')
        plt.title(f"{title} {i}")
        plt.axis('off')
    plt.show()


# =========================
# MAIN
# =========================
pain_files, no_pain_files = count_files()

check_shapes(PAIN_DIR, pain_files)
check_shapes(NO_PAIN_DIR, no_pain_files)

pain_samples = load_random_samples(PAIN_DIR, pain_files)
no_pain_samples = load_random_samples(NO_PAIN_DIR, no_pain_files)

basic_stats(pain_samples, "Pain")
basic_stats(no_pain_samples, "No Pain")

plot_samples(pain_samples, "Pain")
plot_samples(no_pain_samples, "No Pain")