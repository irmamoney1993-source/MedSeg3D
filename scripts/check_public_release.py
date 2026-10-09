"""Audit the published source tree without loading MRI data or PyTorch."""
from __future__ import annotations
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIRED = (
    "README.md", "models/unet3d.py", "scripts/train_unet_baseline_40.py",
    "scripts/train_segresnet_60.py", "scripts/compare_unet60_vs_segresnet60.py",
    "scripts/visualize_unet_segresnet_cases.py", "scripts/verify_results.py",
    "docs/REPRODUCIBILITY.md", "docs/CASE_STUDIES.md",
    "outputs/splits/train_val_split.json", "results/metrics/summary_97.csv",
    "results/metrics/unet60_vs_segresnet60_per_case.csv",
    "results/figures/ATTRIBUTION_AND_LICENSE.md",
)
BLOCKED_NAMES = {"id_rsa", "id_ed25519", ".env", ".env.local", ".env.production"}
BLOCKED_SUFFIXES = (".nii", ".nii.gz", ".pth", ".pt", ".ckpt", ".pkl", ".pickle", ".pem", ".key")
BLOCKED_PATH_PREFIXES = ("data/raw/", "data/cache/", "outputs/checkpoints/", "outputs/backups/")


def main():
    violations = []
    for item in REQUIRED:
        if not (ROOT / item).is_file():
            violations.append("Missing required file: " + item)
    # In a git checkout, audit the actual publishable files rather than local,
    # correctly ignored caches/checkpoints. Tracked files are always included.
    candidates = None
    if (ROOT / ".git").exists():
        try:
            proc = subprocess.run(
                ["git", "-C", str(ROOT), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                capture_output=True, check=True,
            )
            candidates = [ROOT / x.decode("utf-8") for x in proc.stdout.split(b"\0") if x]
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("[WARN] Git not accessible; scanning the complete working folder instead")
    if candidates is None:
        candidates = list(ROOT.rglob("*"))
    files = []
    for p in candidates:
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT).as_posix()
        if ".git" in p.relative_to(ROOT).parts or "__pycache__" in p.parts or ".pytest_cache" in p.parts:
            continue
        files.append(p)
        if p.name.lower() in BLOCKED_NAMES or p.name.lower().endswith(BLOCKED_SUFFIXES) or rel.startswith(BLOCKED_PATH_PREFIXES):
            violations.append("Private or large file in release tree: " + rel)
        if rel.endswith(".zip") or rel.startswith("MedSeg3D/"):
            violations.append("Nested archive/project directory: " + rel)
    split = ROOT / "outputs/splits/train_val_split.json"
    if split.exists():
        try:
            obj = json.loads(split.read_text(encoding="utf-8-sig"))
            tr, va = obj["train"], obj["validation"]
            assert len(tr) == 387 and len(set(tr)) == 387
            assert len(va) == 97 and len(set(va)) == 97
            assert set(tr).isdisjoint(va)
        except (KeyError, ValueError, AssertionError, TypeError) as exc:
            violations.append("Invalid 387/97 split JSON: " + str(exc))
    # Validate local links in public-facing Markdown; external URLs remain unaffected.
    for f in (ROOT / "README.md", ROOT / "docs/CASE_STUDIES.md", ROOT / "docs/REPRODUCIBILITY.md"):
        if not f.exists():
            continue
        s = f.read_text(encoding="utf-8-sig")
        for link in re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)|!\[[^\]]*\]\(([^)]+)\)", s):
            url = (link[0] or link[1]).split("#", 1)[0]
            if not url or "://" in url or url.startswith(("#", "mailto:")):
                continue
            if not (f.parent / url).exists():
                violations.append(f"Broken local link in {f.relative_to(ROOT)}: {url}")
    if violations:
        for v in violations:
            print("[FAIL]", v)
        return 2
    print("[OK] Required source, split, results and documentation are present")
    print("[OK] 387/97 fixed case split validated")
    print("[OK] No NIfTI volumes, checkpoints, nested archive or known secret files in publishable files")
    print("[OK] Main Markdown local links resolve")
    print(f"[OK] Release tree has {len(files)} source and result files")
    print("Note: this is an automated screening, not a full privacy/credential or medical-image-license audit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
