import os
import json
import re
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

############################################
# CONFIG
############################################

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_MODEL_PATH = "gpt2"
FINETUNED_MODEL_PATH = "/kaggle/working/finetuned_model"

ANNOTATION_PATH = "/kaggle/input/datasets/sahithyabr02/openpack-annotation/U0108/annotation/openpack-outliers/S0300.csv"

############################################
# LOAD GROUND TRUTH (FIRST 30 CLIPS)
############################################

df = pd.read_csv(ANNOTATION_PATH)

# convert start/end to datetime
df["start"] = pd.to_datetime(df["start"], format="mixed")
df["end"] = pd.to_datetime(df["end"], format="mixed")

# sort by start time
df = df.sort_values("start").reset_index(drop=True)

# take first 30 clips
df = df.iloc[:30]

ground_truth = []

for i in range(len(df)-1):
    ground_truth.append({
        "dominant_operation": df.loc[i, "event"],
        "anticipated_next_operation": df.loc[i+1, "event"],
        "temporal_segment": [
            df.loc[i, "start"].timestamp(),
            df.loc[i, "end"].timestamp()
        ]
    })

############################################
# TEMPORAL IOU FUNCTION
############################################

def compute_tiou(pred, gt):
    pred_start, pred_end = pred
    gt_start, gt_end = gt

    intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)

    if union <= 0:
        return 0

    return intersection / union

############################################
# MODEL EVALUATION
############################################

def evaluate_model(model_path):

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(DEVICE)

    # Fix pad token warning
    model.config.pad_token_id = tokenizer.eos_token_id

    oca_correct = 0
    aa_correct = 0
    tiou_correct = 0

    valid_predictions = 0
    total = len(ground_truth)

    for gt in ground_truth:

        prompt = (
            "Given a video clip, predict:\n"
            "1. dominant_operation\n"
            "2. anticipated_next_operation\n"
            "3. temporal_segment\n"
            "Return JSON.\n"
            "Answer:\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract FIRST valid JSON block (non-greedy)
        match = re.search(r"\{.*?\}", decoded, re.DOTALL)

        if not match:
            continue

        try:
            prediction = json.loads(match.group())
        except:
            continue

        valid_predictions += 1

        # OCA
        if prediction.get("dominant_operation") == gt["dominant_operation"]:
            oca_correct += 1

        # AA@1
        if prediction.get("anticipated_next_operation") == gt["anticipated_next_operation"]:
            aa_correct += 1

        # tIoU
        if "temporal_segment" in prediction:
            try:
                tiou = compute_tiou(
                    prediction["temporal_segment"],
                    gt["temporal_segment"]
                )
                if tiou >= 0.5:
                    tiou_correct += 1
            except:
                pass

    print(f"Valid predictions: {valid_predictions}/{total}")

    return {
        "OCA": oca_correct / total,
        "tIoU@0.5": tiou_correct / total,
        "AA@1": aa_correct / total
    }

############################################
# RUN BOTH MODELS
############################################

results = {
    "base_model": evaluate_model(BASE_MODEL_PATH),
    "finetuned_model": evaluate_model(FINETUNED_MODEL_PATH)
}

############################################
# SAVE RESULTS
############################################

with open("/kaggle/working/results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nFINAL RESULTS:\n")
print(json.dumps(results, indent=2))