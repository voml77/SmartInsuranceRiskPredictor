import os
import shutil
import random
from pathlib import Path

# Quelle & Ziel
SOURCE_DIR = "data/temp_training"
TARGET_DIR = "data/images"
CLASSES = ["00-damage", "01-whole"]
SPLIT_RATIO = 0.8  # 80% Training, 20% Test

def split_data():
    for cls in CLASSES:
        source_class_dir = Path(SOURCE_DIR) / cls
        files = list(source_class_dir.glob("*"))
        random.shuffle(files)

        n_train = int(len(files) * SPLIT_RATIO)
        train_files = files[:n_train]
        test_files = files[n_train:]

        # Zielordner anlegen
        for split in ["train", "test"]:
            split_class_dir = Path(TARGET_DIR) / split / cls
            split_class_dir.mkdir(parents=True, exist_ok=True)

        # Kopieren
        for f in train_files:
            shutil.copy(f, Path(TARGET_DIR) / "train" / cls / f.name)

        for f in test_files:
            shutil.copy(f, Path(TARGET_DIR) / "test" / cls / f.name)

        print(f"✅ Klasse '{cls}': {len(train_files)} train, {len(test_files)} test")

    print("✅ Split abgeschlossen!")

if __name__ == "__main__":
    split_data()
    