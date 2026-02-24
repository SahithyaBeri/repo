
import os
import json
import subprocess
import pandas as pd
from glob import glob

# =========================
# CONFIG
# =========================

ANNOTATION_ROOT = "/kaggle/input/openpack-annotation"
VIDEO_PATH = "/kaggle/working/sample.mp4"

FRAME_ROOT = "/kaggle/working/frames"
OUTPUT_ROOT = "/kaggle/working/training_data_samples"

FPS = 25
RESOLUTION = 336
WINDOW_SEC = 0.5
WINDOW_FRAMES = int(FPS * WINDOW_SEC)

os.makedirs(FRAME_ROOT, exist_ok=True)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# =========================
# Extract frames
# =========================

def extract_frames(video_path, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    if len(os.listdir(output_folder)) > 0:
        return

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"scale={RESOLUTION}:{RESOLUTION},fps={FPS}",
        f"{output_folder}/frame_%05d.jpg",
        "-hide_banner",
        "-loglevel", "error"
    ]

    subprocess.run(cmd)


# =========================
# Load annotations
# =========================

def load_annotations():

    files = glob(
        f"{ANNOTATION_ROOT}/U*/annotation/openpack-operations/*.csv"
    )

    dfs = []

    for f in files:

        df = pd.read_csv(f)

        df["session"] = os.path.basename(f).replace(".csv","")
        df["subject"] = f.split("/")[-4]

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


# =========================
# Boundary sampling
# =========================

def sample_clip(idx, total_frames):

    center = idx % total_frames

    start = max(0, center - WINDOW_FRAMES)
    end = min(total_frames - 1, center + WINDOW_FRAMES)

    return start, end


# =========================
# Get frame paths
# =========================

def get_frames(folder, start, end, stride=3):

    frames = []

    for i in range(start, end, stride):

        path = f"{folder}/frame_{i+1:05d}.jpg"

        if os.path.exists(path):
            frames.append(path)

    return frames


# =========================
# MAIN
# =========================

def run():

    frame_folder = f"{FRAME_ROOT}/sample"

    extract_frames(VIDEO_PATH, frame_folder)

    total_frames = len(os.listdir(frame_folder))

    annotations = load_annotations()

    saved = 0

    for idx, row in annotations.iterrows():

        start, end = sample_clip(idx, total_frames)

        frames = get_frames(frame_folder, start, end)

        if len(frames) == 0:
            continue

        example = {

            "frames": frames,

            "target": {

                "clip_id":
                f"{row['subject']}_{row['session']}_{idx}",

                "dominant_operation": "operation",

                "temporal_segment": {

                    "start_frame": start,
                    "end_frame": end
                },

                "anticipated_next_operation": "UNKNOWN",

                "confidence": 1.0
            }
        }

        with open(
            f"{OUTPUT_ROOT}/sample_{saved}.json",
            "w"
        ) as f:

            json.dump(example, f, indent=2)

        saved += 1

        if saved == 20:
            break

    print("Saved", saved, "examples")


if __name__ == "__main__":
    run()
