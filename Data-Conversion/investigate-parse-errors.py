import os
import glob
from collections import Counter

INPUT_FOLDER = "Data/Raw-Data/Audio"

part_counts  = Counter()
samples      = {}   # part_count → example filenames

actor_folders = sorted(os.listdir(INPUT_FOLDER))

for actor_folder in actor_folders:
    actor_folder_path = os.path.join(INPUT_FOLDER, actor_folder)
    if not os.path.isdir(actor_folder_path):
        continue

    for filename in sorted(os.listdir(actor_folder_path)):
        if not filename.lower().endswith(".wav"):
            continue

        base  = os.path.splitext(filename)[0]
        parts = base.split("-")
        n     = len(parts)

        part_counts[n] += 1

        # Keep up to 3 examples per part count
        if n not in samples:
            samples[n] = []
        if len(samples[n]) < 3:
            samples[n].append(f"{actor_folder}/{filename}")

# ── Report ─────────────────────────────────────────────────────────────────
print("Filename part counts (split by '-'):")
print(f"  {'Parts':>6}  {'Count':>6}")
print(f"  {'------':>6}  {'------':>6}")
for n, count in sorted(part_counts.items()):
    print(f"  {n:>6}  {count:>6}")

print("\nExample filenames per part count:")
for n, examples in sorted(samples.items()):
    print(f"\n  [{n} parts]")
    for ex in examples:
        print(f"    {ex}")