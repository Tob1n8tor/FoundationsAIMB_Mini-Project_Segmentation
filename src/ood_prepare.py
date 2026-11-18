"""Bonus task (OOD), data preparation for the TotalSegmentator dataset (v300).

1. Merge the 13 TotalSegmentator masks that correspond to BTCV organs into one BTCV-indexed label map per case
   (`ts_btcv/<case>_label.nii.gz`; uint8, same grid/affine as the CT). Existing labels are not overwritten.
2. Survey the test split: scan shape, spacing, orientation and voxel count per organ (`survey.csv`).
3. Split the test cases into `case_lists/abdominal.txt` (all of OOD_CORE_ORGANS in the field of view) and
   `case_lists/non_abdominal.txt` (the rest: head, neck, chest-only, leg scans, ...).

Input: the TotalSegmentator dataset (`Totalsegmentator_dataset_v300/` with meta.csv), at least the test-split
cases. Usage: python -m src.ood_prepare [--workers 6]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np
import pandas as pd

from src.config import (
    OOD_CASE_LIST_DIR,
    OOD_CORE_ORGANS,
    OOD_DATASET_DIR,
    OOD_LABEL_DIR,
    OOD_SURVEY_PATH,
    TS_TO_BTCV,
)


def build_label(case: str) -> None:
    """Merge the 13 BTCV-compatible TotalSegmentator masks into one label map with BTCV indices 1..13."""
    out = OOD_LABEL_DIR / f"{case}_label.nii.gz"
    if out.exists():
        return
    ct = nib.load(OOD_DATASET_DIR / case / "ct.nii.gz")  # header only
    lab = np.zeros(ct.shape, np.uint8)
    for idx, name in TS_TO_BTCV.items():
        lab[np.asanyarray(nib.load(OOD_DATASET_DIR / case / "segmentations" / f"{name}.nii.gz").dataobj) > 0] = idx
    hdr = ct.header.copy()
    hdr.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(lab, ct.affine, hdr), str(out))


def survey_case(case: str) -> dict:
    ct = nib.load(OOD_DATASET_DIR / case / "ct.nii.gz")
    counts = np.bincount(np.asarray(nib.load(OOD_LABEL_DIR / f"{case}_label.nii.gz").dataobj).ravel(), minlength=14)
    return {
        "case": case,
        "shape": tuple(int(s) for s in ct.shape),
        "zooms": tuple(round(float(z), 2) for z in ct.header.get_zooms()),
        "axcodes": "".join(nib.aff2axcodes(ct.affine)),
        **{name: int(counts[idx]) for idx, name in TS_TO_BTCV.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    meta = pd.read_csv(OOD_DATASET_DIR / "meta.csv", sep=";")
    cases = sorted(meta.loc[meta.split == "test", "image_id"])
    OOD_LABEL_DIR.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.workers) as ex:
        list(ex.map(build_label, cases))
        survey = pd.DataFrame(list(ex.map(survey_case, cases)))

    survey = survey.merge(meta, left_on="case", right_on="image_id").drop(columns="image_id")
    survey["n_organs"] = (survey[list(TS_TO_BTCV.values())] > 0).sum(axis=1)
    survey["abdominal"] = (survey[OOD_CORE_ORGANS] > 0).all(axis=1)
    survey.to_csv(OOD_SURVEY_PATH, index=False)

    OOD_CASE_LIST_DIR.mkdir(parents=True, exist_ok=True)
    for name, sel in (("abdominal", survey.abdominal), ("non_abdominal", ~survey.abdominal)):
        (OOD_CASE_LIST_DIR / f"{name}.txt").write_text("\n".join(survey.case[sel]) + "\n")
    print(f"test cases: {len(survey)}  abdominal: {survey.abdominal.sum()}  "
          f"all 13 organs: {(survey.n_organs == 13).sum()}  no BTCV organ: {(survey.n_organs == 0).sum()}")


if __name__ == "__main__":
    main()
