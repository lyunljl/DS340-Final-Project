"""
EDA — Pain / No-Pain Spectrogram Dataset
=========================================
Analyses the .npy log-mel spectrograms in:
  Data/all-pain/npy/pain/
  Data/all-pain/npy/no-pain/

Run:
  python eda.py
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────
PAIN_DIR    = "Data/all-pain/npy/pain"
NO_PAIN_DIR = "Data/all-pain/npy/no-pain"
OUTPUT_DIR  = "Data/all-pain/eda"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPECTED_SHAPE = (128, 94)

print("=" * 60)
print("  EDA — Pain / No-Pain Spectrogram Dataset")
print("=" * 60)


# ── 1. Load all files ──────────────────────────────────────────────────────
def load_folder(folder, label_name):
    paths   = sorted(glob.glob(os.path.join(folder, "*.npy")))
    arrays, bad_shapes, filenames = [], [], []
    for p in paths:
        arr = np.load(p)
        if arr.shape != EXPECTED_SHAPE:
            bad_shapes.append((os.path.basename(p), arr.shape))
            continue
        arrays.append(arr.astype(np.float32))
        filenames.append(os.path.basename(p))
    print(f"\n[{label_name}]")
    print(f"  Files found   : {len(paths)}")
    print(f"  Loaded ok     : {len(arrays)}")
    print(f"  Shape errors  : {len(bad_shapes)}")
    if bad_shapes:
        for name, shape in bad_shapes[:5]:
            print(f"    {name} → {shape}")
    return np.array(arrays, dtype=np.float32), filenames

print("\n── 1. Loading data ──────────────────────────────────────────")
pain_X,    pain_files    = load_folder(PAIN_DIR,    "PAIN")
no_pain_X, no_pain_files = load_folder(NO_PAIN_DIR, "NO-PAIN")

X_all = np.concatenate([pain_X, no_pain_X], axis=0)
y_all = np.array([1]*len(pain_X) + [0]*len(no_pain_X))

print(f"\n  Total samples : {len(X_all)}")
print(f"  Pain    (1)   : {len(pain_X)}")
print(f"  No-pain (0)   : {len(no_pain_X)}")
print(f"  Imbalance ratio (no-pain:pain) : {len(no_pain_X)/max(len(pain_X),1):.2f}:1")


# ── 2. Class distribution ──────────────────────────────────────────────────
print("\n── 2. Class Distribution ────────────────────────────────────")
fig, ax = plt.subplots(figsize=(6, 4))
counts  = [len(no_pain_X), len(pain_X)]
bars    = ax.bar(["No-Pain", "Pain"], counts,
                 color=["#4a90d9", "#e05c5c"], edgecolor="white", linewidth=0.8)
for bar, v in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(v), ha="center", fontsize=12, fontweight="bold")
ax.set_title("Class Distribution", fontsize=14, fontweight="bold")
ax.set_ylabel("Sample count")
ax.set_ylim(0, max(counts) * 1.15)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_class_distribution.png"), dpi=130)
plt.show()
print(f"  Saved → 01_class_distribution.png")


# ── 3. Global value statistics ─────────────────────────────────────────────
print("\n── 3. Value Statistics ──────────────────────────────────────")
for name, X in [("Pain", pain_X), ("No-Pain", no_pain_X), ("All", X_all)]:
    print(f"\n  [{name}]")
    print(f"    min  : {X.min():.2f}")
    print(f"    max  : {X.max():.2f}")
    print(f"    mean : {X.mean():.2f}")
    print(f"    std  : {X.std():.2f}")
    print(f"    median: {np.median(X):.2f}")


# ── 4. Value histograms ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Pixel-Value Histograms (Log-Mel dB)", fontsize=13, fontweight="bold")

for ax, (name, X, color) in zip(axes, [
    ("All Samples",  X_all,    "#888888"),
    ("Pain",         pain_X,   "#e05c5c"),
    ("No-Pain",      no_pain_X,"#4a90d9"),
]):
    ax.hist(X.flatten(), bins=120, color=color, edgecolor="none", alpha=0.85)
    ax.set_title(name)
    ax.set_xlabel("Log-mel value (dB)")
    ax.set_ylabel("Count")
    ax.axvline(X.mean(), color="black", linestyle="--", linewidth=1,
               label=f"mean={X.mean():.1f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_value_histograms.png"), dpi=130)
plt.show()
print(f"  Saved → 02_value_histograms.png")


# ── 5. Silence analysis ────────────────────────────────────────────────────
print("\n── 5. Silence Analysis (values ≤ -79 dB) ───────────────────")
pain_silence    = (pain_X    <= -79).mean(axis=(1, 2))
no_pain_silence = (no_pain_X <= -79).mean(axis=(1, 2))
all_silence     = np.concatenate([pain_silence, no_pain_silence])

for name, s in [("Pain", pain_silence), ("No-Pain", no_pain_silence)]:
    print(f"\n  [{name}]")
    print(f"    Mean silence ratio : {s.mean():.2%}")
    print(f"    Samples >50% silence : {(s > 0.5).sum()}  ({(s > 0.5).mean():.1%})")
    print(f"    Samples >90% silence : {(s > 0.9).sum()}  ({(s > 0.9).mean():.1%})")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("Silence Distribution (fraction of spectrogram at -80 dB floor)",
             fontsize=13, fontweight="bold")

axes[0].hist(pain_silence,    bins=40, alpha=0.7, label="Pain",    color="#e05c5c")
axes[0].hist(no_pain_silence, bins=40, alpha=0.7, label="No-Pain", color="#4a90d9")
axes[0].set_xlabel("Silence ratio")
axes[0].set_ylabel("Count")
axes[0].set_title("Overlaid by Class")
axes[0].legend()
axes[0].grid(alpha=0.2)

axes[1].boxplot([pain_silence, no_pain_silence],
                labels=["Pain", "No-Pain"],
                patch_artist=True,
                boxprops=dict(facecolor="#f0f0f0"))
axes[1].set_ylabel("Silence ratio")
axes[1].set_title("Boxplot by Class")
axes[1].grid(alpha=0.2)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_silence_analysis.png"), dpi=130)
plt.show()
print(f"  Saved → 03_silence_analysis.png")


# ── 6. Mean spectrograms ───────────────────────────────────────────────────
print("\n── 6. Mean Spectrograms ─────────────────────────────────────")
pain_mean    = pain_X.mean(axis=0)
no_pain_mean = no_pain_X.mean(axis=0)
diff         = pain_mean - no_pain_mean

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Mean Spectrograms", fontsize=13, fontweight="bold")

vmin = min(pain_mean.min(), no_pain_mean.min())
vmax = max(pain_mean.max(), no_pain_mean.max())

im0 = axes[0].imshow(pain_mean,    aspect="auto", origin="lower",
                     cmap="magma", vmin=vmin, vmax=vmax)
axes[0].set_title("Mean — Pain")
axes[0].set_xlabel("Time frames"); axes[0].set_ylabel("Mel bins")
plt.colorbar(im0, ax=axes[0], label="dB")

im1 = axes[1].imshow(no_pain_mean, aspect="auto", origin="lower",
                     cmap="magma", vmin=vmin, vmax=vmax)
axes[1].set_title("Mean — No-Pain")
axes[1].set_xlabel("Time frames"); axes[1].set_ylabel("Mel bins")
plt.colorbar(im1, ax=axes[1], label="dB")

abs_max = np.abs(diff).max()
im2 = axes[2].imshow(diff, aspect="auto", origin="lower",
                     cmap="RdBu_r", vmin=-abs_max, vmax=abs_max)
axes[2].set_title("Difference (Pain − No-Pain)")
axes[2].set_xlabel("Time frames"); axes[2].set_ylabel("Mel bins")
plt.colorbar(im2, ax=axes[2], label="dB diff")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_mean_spectrograms.png"), dpi=130)
plt.show()
print(f"  Saved → 04_mean_spectrograms.png")


# ── 7. Per-frequency band energy ───────────────────────────────────────────
print("\n── 7. Per-Frequency Band Energy ─────────────────────────────")
pain_freq_mean    = pain_X.mean(axis=(0, 2))     # avg over samples and time → (128,)
no_pain_freq_mean = no_pain_X.mean(axis=(0, 2))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(pain_freq_mean,    color="#e05c5c", label="Pain",    linewidth=1.5)
ax.plot(no_pain_freq_mean, color="#4a90d9", label="No-Pain", linewidth=1.5)
ax.fill_between(range(128), pain_freq_mean, no_pain_freq_mean,
                alpha=0.15, color="purple", label="Difference")
ax.set_xlabel("Mel bin (low → high frequency)")
ax.set_ylabel("Mean log-mel energy (dB)")
ax.set_title("Average Energy per Frequency Band", fontsize=13, fontweight="bold")
ax.legend()
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_frequency_band_energy.png"), dpi=130)
plt.show()
print(f"  Saved → 05_frequency_band_energy.png")


# ── 8. Per-time energy ────────────────────────────────────────────────────
print("\n── 8. Per-Time Energy ───────────────────────────────────────")
pain_time_mean    = pain_X.mean(axis=(0, 1))     # avg over samples and mel → (94,)
no_pain_time_mean = no_pain_X.mean(axis=(0, 1))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(pain_time_mean,    color="#e05c5c", label="Pain",    linewidth=1.5)
ax.plot(no_pain_time_mean, color="#4a90d9", label="No-Pain", linewidth=1.5)
ax.set_xlabel("Time frame")
ax.set_ylabel("Mean log-mel energy (dB)")
ax.set_title("Average Energy over Time", fontsize=13, fontweight="bold")
ax.legend()
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "06_time_energy.png"), dpi=130)
plt.show()
print(f"  Saved → 06_time_energy.png")


# ── 9. Sample spectrograms ────────────────────────────────────────────────
print("\n── 9. Sample Spectrograms ───────────────────────────────────")
n_show = 6
fig, axes = plt.subplots(2, n_show, figsize=(16, 5))
fig.suptitle("Random Sample Spectrograms", fontsize=13, fontweight="bold")

rng = np.random.default_rng(42)
pain_idx    = rng.choice(len(pain_X),    size=n_show, replace=False)
no_pain_idx = rng.choice(len(no_pain_X), size=n_show, replace=False)

for col in range(n_show):
    axes[0, col].imshow(pain_X[pain_idx[col]],
                        aspect="auto", origin="lower", cmap="magma")
    axes[0, col].set_title(f"Pain #{pain_idx[col]}", fontsize=8)
    axes[0, col].axis("off")

    axes[1, col].imshow(no_pain_X[no_pain_idx[col]],
                        aspect="auto", origin="lower", cmap="magma")
    axes[1, col].set_title(f"No-Pain #{no_pain_idx[col]}", fontsize=8)
    axes[1, col].axis("off")

axes[0, 0].set_ylabel("Pain",    fontsize=10)
axes[1, 0].set_ylabel("No-Pain", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "07_sample_spectrograms.png"), dpi=130)
plt.show()
print(f"  Saved → 07_sample_spectrograms.png")


# ── 10. Variance map ──────────────────────────────────────────────────────
print("\n── 10. Variance Maps ────────────────────────────────────────")
pain_var    = pain_X.var(axis=0)
no_pain_var = no_pain_X.var(axis=0)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Variance Maps (high = more variation across samples)",
             fontsize=13, fontweight="bold")

vmax = max(pain_var.max(), no_pain_var.max())
im0 = axes[0].imshow(pain_var,    aspect="auto", origin="lower",
                     cmap="hot", vmin=0, vmax=vmax)
axes[0].set_title("Variance — Pain")
axes[0].set_xlabel("Time frames"); axes[0].set_ylabel("Mel bins")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(no_pain_var, aspect="auto", origin="lower",
                     cmap="hot", vmin=0, vmax=vmax)
axes[1].set_title("Variance — No-Pain")
axes[1].set_xlabel("Time frames"); axes[1].set_ylabel("Mel bins")
plt.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "08_variance_maps.png"), dpi=130)
plt.show()
print(f"  Saved → 08_variance_maps.png")


# ── 11. Summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  EDA SUMMARY")
print("=" * 60)
print(f"  Total samples     : {len(X_all)}")
print(f"  Pain              : {len(pain_X)}  ({len(pain_X)/len(X_all):.1%})")
print(f"  No-pain           : {len(no_pain_X)}  ({len(no_pain_X)/len(X_all):.1%})")
print(f"  Imbalance ratio   : {len(no_pain_X)/max(len(pain_X),1):.2f}:1")
print(f"  Global mean (dB)  : {X_all.mean():.2f}")
print(f"  Global std  (dB)  : {X_all.std():.2f}")
print(f"  Avg silence (pain): {pain_silence.mean():.2%}")
print(f"  Avg silence (no-p): {no_pain_silence.mean():.2%}")
print(f"\n  All plots saved to: {OUTPUT_DIR}/")
print("=" * 60)