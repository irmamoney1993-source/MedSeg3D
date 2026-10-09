"""Check local resources required to reproduce segmentation inference (standard library only)."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent.parent
required_files = {
    "Custom U-Net source": ROOT / "models/unet3d.py",
    "Fixed case split": ROOT / "outputs/splits/train_val_split.json",
    "U-Net Best checkpoint": ROOT / "outputs/checkpoints/unet_baseline_40_fast_best.pth",
    "SegResNet Best checkpoint": ROOT / "outputs/checkpoints/segresnet_baseline_60_fast_best.pth",
}
required_dirs = {
    "MSD image volumes": ROOT / "data/raw/Task01_BrainTumour/imagesTr",
    "MSD segmentation masks": ROOT / "data/raw/Task01_BrainTumour/labelsTr",
}


def main():
    missing = []
    for label, path in required_files.items():
        ok = path.is_file()
        print(f"{'[OK]' if ok else '[MISSING]'} {label}: {path.relative_to(ROOT)}")
        if not ok:
            missing.append(label)
    for label, path in required_dirs.items():
        ok = path.is_dir()
        print(f"{'[OK]' if ok else '[MISSING]'} {label}: {path.relative_to(ROOT)}")
        if not ok:
            missing.append(label)
    path = required_files["Fixed case split"]
    if path.is_file():
        try:
            split = json.loads(path.read_text(encoding="utf-8-sig"))
            train, val = split["train"], split["validation"]
            assert len(train) == 387 and len(val) == 97
            assert len(set(train)) == 387 and len(set(val)) == 97
            assert set(train).isdisjoint(val)
            print("[OK] Fixed split: 387 train / 97 validation, no duplicate cases")
        except (KeyError, ValueError, AssertionError, TypeError) as exc:
            print("[ERROR] Invalid split:", exc)
            missing.append("split integrity")
    if missing:
        print("\nInference prerequisites incomplete. The published result tables can still be checked with verify_results.py.")
        return 2
    print("\nLocal resource check passed. Inference still requires a compatible PyTorch/MONAI environment.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
