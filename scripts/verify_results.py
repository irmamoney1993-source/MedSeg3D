"""Independent standard-library verification of the published 97-case Dice tables."""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results/metrics"
SPLIT = ROOT / "outputs/splits/train_val_split.json"
CHECK_METRICS = ("Class1", "Class2", "Class3", "RawMean", "WT", "TC", "ET", "RegionMean")


def finite_values(rows, key):
    arr = []
    for row in rows:
        value = float(row[key])
        if math.isfinite(value):
            arr.append(value)
    return arr


def near(a, b, description, tolerance=1e-8):
    if not math.isclose(a, b, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"{description}: {a:.12f} != {b:.12f}")


def load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    split = json.loads(SPLIT.read_text(encoding="utf-8-sig"))
    train = split["train"]
    val = split["validation"]
    assert len(train) == 387 and len(val) == 97, "split size must be 387/97"
    assert len(set(train)) == 387 and len(set(val)) == 97, "duplicate cases in split"
    assert set(train).isdisjoint(val), "train/validation overlap"
    assert all(x.endswith(".nii.gz") for x in (*train, *val))
    rows = load_csv(DATA / "unet60_vs_segresnet60_per_case.csv")
    assert len(rows) == 97, "per-case table should have 97 records"
    assert len({r["case"] for r in rows}) == 97, "duplicate case in CSV"
    assert {r["case"] for r in rows} == set(val), "case IDs differ from fixed split"
    summ = {r["Metric"]: r for r in load_csv(DATA / "summary_97.csv")}
    assert set(summ) == set(CHECK_METRICS)

    et_empty = [r for r in rows if r["ET_GT_empty"].strip().lower() == "true"]
    assert len(et_empty) == 4, "expected four GT-negative ET cases"
    print(f"[OK] Case split: {len(train)} train, {len(val)} validation, no overlap")
    print(f"[OK] Per-case table: {len(rows)} unique validation cases; ET GT-negative: {len(et_empty)}")
    for metric in CHECK_METRICS:
        nvalid = 93 if metric in {"Class3", "ET"} else 97
        reference = summ[metric]
        if metric in {"RawMean", "RegionMean"}:
            assert reference["ValidCases"] == "macro", f"expected macro summary for {metric}"
        else:
            assert int(reference["ValidCases"]) == nvalid, f"unexpected valid case count for {metric}"
        means = {}
        for name in ("UNet60", "SegResNet60"):
            col = f"{name}_{metric}"
            if metric in {"RawMean", "RegionMean"}:
                col += "_case"
            # Per-case RegionMean includes NaN exclusion per case. Published macro
            # RegionMean is instead the average of 3 population region means.
            if metric == "RegionMean":
                vals = [sum(finite_values(rows, f"{name}_{r}")) / len(finite_values(rows, f"{name}_{r}")) for r in ("WT", "TC", "ET")]
                means[name] = sum(vals) / 3
            elif metric == "RawMean":
                vals = [sum(finite_values(rows, f"{name}_{r}")) / len(finite_values(rows, f"{name}_{r}")) for r in ("Class1", "Class2", "Class3")]
                means[name] = sum(vals) / 3
            else:
                vals = finite_values(rows, col)
                assert len(vals) == nvalid, f"wrong valid-case count for {col}"
                means[name] = sum(vals) / len(vals)
            near(means[name], float(reference[name]), f"{name} {metric}")
        near(means["SegResNet60"]-means["UNet60"], float(reference["Delta_SegResNet_minus_UNet"]), f"delta {metric}")
        print(f"[OK] {metric:10} U-Net={means['UNet60']:.4f} SegResNet={means['SegResNet60']:.4f}")

    # Confirm TC preference counts and GT-negative ET false positives.
    delta = [float(r["Delta_SegResNet_minus_UNet_TC"]) for r in rows]
    n_better = sum(d > 0 for d in delta)
    n_worse = sum(d < 0 for d in delta)
    assert (n_better, n_worse) == (65, 32), "TC paired outcome changed"
    fp_unet = sum(int(r["UNet60_ET_pred_voxels"]) for r in et_empty)
    fp_seg = sum(int(r["SegResNet60_ET_pred_voxels"]) for r in et_empty)
    assert (fp_unet, fp_seg) == (3836, 3732), "GT-negative ET counts do not match"
    print(f"[OK] TC improved / declined: {n_better} / {n_worse}")
    print(f"[OK] ET false positives on GT-negative cases: U-Net={fp_unet}, SegResNet={fp_seg} voxels")
    print("PASS: published results are internally consistent with case-level records.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
