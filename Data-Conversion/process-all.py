import os
import csv
from convert import MelConverter


# =========================
# PERMANENT PATHS (FIXED)
# =========================
INPUT_FOLDER = "Data/Raw-Data/Audio"

NPY_PAIN_PATH    = "Data/all-pain/npy/pain"
NPY_NO_PAIN_PATH = "Data/all-pain/npy/no-pain"

PNG_PAIN_PATH    = "Data/all-pain/png/pain"
PNG_NO_PAIN_PATH = "Data/all-pain/png/no-pain"

METADATA_PATH = "Data/all-pain/metadata/labels_v2.csv"


# =========================
# LABELING RULES
# =========================
VALID_MODALITY      = "03"
VALID_VOCAL_CHANNEL = "02"   # non-speech only
PAIN_EMOTION        = "11"   # pain/groan → pain folder
INVALID_ENDINGS     = {"66"} # mixed voices — excluded
                             # "77" (noisy env) → INCLUDED in both folders


def parse_filename(filename):
    """
    Handles all filename formats found in the dataset:
      8  parts → older format, no ending code
      9  parts → missing ending code
      10 parts → standard format
      11 parts → extra similarity field at end
      12 parts → noisy files (77 is the last part)

    Emotion is always at index 2 across all formats.
    """
    if not filename.lower().endswith(".wav"):
        return None

    base  = os.path.splitext(filename)[0]
    parts = base.split("-")
    n     = len(parts)

    if n not in {8, 9, 10, 11, 12}:
        return None

    # Determine ending code based on format
    if n == 8:
        ending_code = "00"      # no ending code in this format
    elif n == 9:
        ending_code = "00"      # no ending code in this format
    elif n == 10:
        ending_code = parts[9]  # standard position
    elif n == 11:
        ending_code = parts[9]  # same as 10-part, extra field at [10]
    elif n == 12:
        ending_code = parts[11] # 77 is the very last part

    return {
        "modality":      parts[0],
        "vocal_channel": parts[1],
        "emotion":       parts[2],
        "intensity":     parts[3],
        "statement":     parts[4],
        "actor":         parts[5],
        "age":           parts[6],
        "source":        parts[7],
        "language":      parts[8] if n >= 9 else "00",
        "ending_code":   ending_code,
        "base_name":     base,
    }


def should_keep_file(info):
    """
    Returns (keep, label, class_name, is_noisy)

    Inclusion rules:
      - Must be audio-only (modality 03)
      - Must be non-speech (vocal channel 02)
      - Must NOT be a mixed-voice file (ending 66)
      - Noisy files (ending 77) ARE included

    Labeling:
      - Emotion 11 (pain/groan) → pain    (label=1)
      - Everything else         → no-pain (label=0)
    """
    if info is None:
        return False, None, None, False

    if info["modality"] != VALID_MODALITY:
        return False, None, None, False

    if info["vocal_channel"] != VALID_VOCAL_CHANNEL:
        return False, None, None, False

    if info["ending_code"] in INVALID_ENDINGS:
        return False, None, None, False

    is_noisy = info["ending_code"] == "77"
    is_pain  = info["emotion"] == PAIN_EMOTION

    if is_pain:
        return True, 1, "pain", is_noisy
    else:
        return True, 0, "no-pain", is_noisy


def write_metadata(rows):
    fieldnames = [
        "original_path",
        "actor_folder",
        "filename",
        "base_name",
        "modality",
        "vocal_channel",
        "emotion",
        "intensity",
        "statement",
        "actor",
        "age",
        "source",
        "language",
        "ending_code",
        "label",
        "class_name",
        "is_noisy",
        "npy_output_path",
        "png_output_path",
    ]

    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)

    with open(METADATA_PATH, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_dataset():
    converter = MelConverter()

    metadata_rows = []

    processed_pain          = 0
    processed_no_pain       = 0
    processed_pain_noisy    = 0
    processed_no_pain_noisy = 0
    skipped_invalid         = 0
    skipped_duplicate       = 0
    parse_errors            = 0
    failed_files            = 0

    if not os.path.exists(INPUT_FOLDER):
        print(f"[ERROR] Input folder not found: {INPUT_FOLDER}")
        return

    # Create output directories
    for path in [NPY_PAIN_PATH, NPY_NO_PAIN_PATH,
                 PNG_PAIN_PATH, PNG_NO_PAIN_PATH]:
        os.makedirs(path, exist_ok=True)

    actor_folders = sorted(os.listdir(INPUT_FOLDER))

    for actor_folder in actor_folders:
        actor_folder_path = os.path.join(INPUT_FOLDER, actor_folder)

        if not os.path.isdir(actor_folder_path):
            continue

        for filename in sorted(os.listdir(actor_folder_path)):
            if not filename.lower().endswith(".wav"):
                continue

            info = parse_filename(filename)
            if info is None:
                parse_errors += 1
                continue

            keep, label, class_name, is_noisy = should_keep_file(info)
            if not keep:
                skipped_invalid += 1
                continue

            original_path = os.path.join(actor_folder_path, filename)

            npy_output_dir = NPY_PAIN_PATH if class_name == "pain" else NPY_NO_PAIN_PATH
            png_output_dir = PNG_PAIN_PATH if class_name == "pain" else PNG_NO_PAIN_PATH

            npy_output_path = os.path.join(npy_output_dir, info["base_name"] + ".npy")
            png_output_path = os.path.join(png_output_dir, info["base_name"] + ".png")

            # Skip if already processed
            if os.path.exists(npy_output_path) and os.path.exists(png_output_path):
                skipped_duplicate += 1
                continue

            try:
                converter.process_file(
                    audio_path=original_path,
                    npy_path=npy_output_path,
                    png_path=png_output_path
                )
            except Exception as e:
                failed_files += 1
                print(f"[ERROR] Failed processing {original_path}: {e}")
                continue

            metadata_rows.append({
                "original_path":   original_path,
                "actor_folder":    actor_folder,
                "filename":        filename,
                "base_name":       info["base_name"],
                "modality":        info["modality"],
                "vocal_channel":   info["vocal_channel"],
                "emotion":         info["emotion"],
                "intensity":       info["intensity"],
                "statement":       info["statement"],
                "actor":           info["actor"],
                "age":             info["age"],
                "source":          info["source"],
                "language":        info["language"],
                "ending_code":     info["ending_code"],
                "label":           label,
                "class_name":      class_name,
                "is_noisy":        is_noisy,
                "npy_output_path": npy_output_path,
                "png_output_path": png_output_path,
            })

            if class_name == "pain" and is_noisy:
                processed_pain_noisy += 1
            elif class_name == "pain":
                processed_pain += 1
            elif class_name == "no-pain" and is_noisy:
                processed_no_pain_noisy += 1
            else:
                processed_no_pain += 1

    write_metadata(metadata_rows)

    total_pain    = processed_pain + processed_pain_noisy
    total_no_pain = processed_no_pain + processed_no_pain_noisy

    print("\n[DONE]")
    print(f"Pain files")
    print(f"  Clean : {processed_pain}")
    print(f"  Noisy : {processed_pain_noisy}")
    print(f"  Total : {total_pain}")
    print(f"No-pain files")
    print(f"  Clean : {processed_no_pain}")
    print(f"  Noisy : {processed_no_pain_noisy}")
    print(f"  Total : {total_no_pain}")
    print(f"Skipped invalid   : {skipped_invalid}")
    print(f"Skipped duplicates: {skipped_duplicate}")
    print(f"Parse errors      : {parse_errors}")
    print(f"Failed files      : {failed_files}")
    print(f"Metadata saved to : {METADATA_PATH}")


if __name__ == "__main__":
    process_dataset()