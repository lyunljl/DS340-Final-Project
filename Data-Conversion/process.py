import os
import csv
from convert import MelConverter


# =========================
# PERMANENT PATHS (FIXED)
# =========================
INPUT_FOLDER = "Data/Raw-Data/Audio"

NPY_PAIN_PATH = "Data/clean-pain-only/npy/pain"
NPY_NO_PAIN_PATH = "Data/clean-pain-only/npy/no-pain"

PNG_PAIN_PATH = "Data/clean-pain-only/png/pain"
PNG_NO_PAIN_PATH = "Data/clean-pain-only/png/no-pain"

METADATA_PATH = "Data/clean-pain-only/metadata/labels_v1.csv"


# =========================
# LABELING RULES
# =========================
VALID_MODALITY = "03"
VALID_VOCAL_CHANNEL = "02"   # non-speech only
POSITIVE_EMOTION = "11"      # pain
NEGATIVE_EMOTIONS = {"02", "03", "10"}
INVALID_ENDINGS = {"66", "77"}


def parse_filename(filename):
    if not filename.lower().endswith(".wav"):
        return None

    base = os.path.splitext(filename)[0]
    parts = base.split("-")

    if len(parts) != 10:
        return None

    return {
        "modality": parts[0],
        "vocal_channel": parts[1],
        "emotion": parts[2],
        "intensity": parts[3],
        "statement": parts[4],
        "actor": parts[5],
        "age": parts[6],
        "source": parts[7],
        "language": parts[8],
        "ending_code": parts[9],
        "base_name": base,
    }


def should_keep_file(info):
    if info is None:
        return False, None, None

    if info["modality"] != VALID_MODALITY:
        return False, None, None

    if info["vocal_channel"] != VALID_VOCAL_CHANNEL:
        return False, None, None

    if info["ending_code"] in INVALID_ENDINGS:
        return False, None, None

    emotion = info["emotion"]

    if emotion == POSITIVE_EMOTION:
        return True, 1, "pain"

    if emotion in NEGATIVE_EMOTIONS:
        return True, 0, "no-pain"   # 🔥 match folder name

    return False, None, None


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

    processed_pain = 0
    processed_no_pain = 0
    skipped_invalid = 0
    skipped_duplicate = 0
    parse_errors = 0
    failed_files = 0

    if not os.path.exists(INPUT_FOLDER):
        print(f"[ERROR] Input folder not found: {INPUT_FOLDER}")
        return

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

            keep, label, class_name = should_keep_file(info)
            if not keep:
                skipped_invalid += 1
                continue

            original_path = os.path.join(actor_folder_path, filename)

            if class_name == "pain":
                npy_output_dir = NPY_PAIN_PATH
                png_output_dir = PNG_PAIN_PATH
            else:
                npy_output_dir = NPY_NO_PAIN_PATH
                png_output_dir = PNG_NO_PAIN_PATH

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
                "original_path": original_path,
                "actor_folder": actor_folder,
                "filename": filename,
                "base_name": info["base_name"],
                "modality": info["modality"],
                "vocal_channel": info["vocal_channel"],
                "emotion": info["emotion"],
                "intensity": info["intensity"],
                "statement": info["statement"],
                "actor": info["actor"],
                "age": info["age"],
                "source": info["source"],
                "language": info["language"],
                "ending_code": info["ending_code"],
                "label": label,
                "class_name": class_name,
                "npy_output_path": npy_output_path,
                "png_output_path": png_output_path,
            })

            if class_name == "pain":
                processed_pain += 1
            else:
                processed_no_pain += 1

    write_metadata(metadata_rows)

    print("\n[DONE]")
    print(f"Processed pain files: {processed_pain}")
    print(f"Processed no-pain files: {processed_no_pain}")
    print(f"Skipped invalid files: {skipped_invalid}")
    print(f"Skipped duplicates: {skipped_duplicate}")
    print(f"Filename parse errors: {parse_errors}")
    print(f"Failed files: {failed_files}")
    print(f"Metadata saved to: {METADATA_PATH}")


if __name__ == "__main__":
    process_dataset()