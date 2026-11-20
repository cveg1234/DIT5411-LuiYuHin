"""Prepare PACKED .keras models for --resume training.

The user uploads model_A.keras / model_B.keras into Kaggle's /kaggle/working
(via the file browser's upload button). This script copies them into the
checkpoint layout train.py expects and writes the state JSON, so a run with
`--resume --epochs 2` trains exactly one MORE epoch from those weights.
"""
import json
import os
import sys

DEST_DIR = "results/checkpoints"

# Kaggle input paths (model attached as Input) first, then local/working dir.
KAGGLE_BASE = ("/kaggle/input/models/cveg1234/"
               "dit5411-chinese-character-recognition/tensorflow2/default/1")
SOURCES = [
    ("A", [f"{KAGGLE_BASE}/model_A.keras",
           "model_A.keras", "uploads/model_A.keras", "backup/model_A.keras"]),
    ("B", [f"{KAGGLE_BASE}/model_B.keras",
           "model_B.keras", "uploads/model_B.keras", "backup/model_B.keras"]),
]

# If the uploaded results file exists, use its accuracies as history seed.
hist = {"accuracy": [], "val_accuracy": []}
if os.path.isfile("results/RESULTS.md"):
    text = open("results/RESULTS.md", encoding="utf-8").read()
    for line in text.splitlines():
        if "| A |" in line or "| B |" in line:
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) >= 3 and "%" in parts[1]:
                try:
                    hist["accuracy"].append(float(parts[1].strip("%")) / 100)
                except ValueError:
                    pass

os.makedirs(DEST_DIR, exist_ok=True)

for model, candidates in SOURCES:
    src = next((p for p in candidates if os.path.isfile(p)), None)
    if src is None:
        print(f"Model {model}: no uploaded .keras found (looked for {candidates})")
        continue
    dst = os.path.join(DEST_DIR, f"model_{model}_latest.keras")
    if os.path.abspath(src) != os.path.abspath(dst):
        import shutil
        shutil.copy2(src, dst)
    state = {
        "completed_epochs": 1,
        "history": {"accuracy": list(hist["accuracy"]),
                    "val_accuracy": list(hist["val_accuracy"])},
    }
    with open(os.path.join(DEST_DIR, f"model_{model}_state.json"), "w",
              encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    print(f"Model {model}: prepared checkpoint from {src} "
          f"(completed_epochs=1)")
    print("  -> run with --resume --epochs 2 to train one more epoch")
