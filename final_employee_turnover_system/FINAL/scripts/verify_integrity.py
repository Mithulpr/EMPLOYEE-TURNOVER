#!/usr/bin/env python3
"""
Cryptographic SHA-256 Immutability Verifier
============================================
Computes and verifies SHA-256 hashes for all protected artifacts
against the baseline manifest to guarantee zero drift or alteration.
"""

import sys
import hashlib
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
FINAL_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
PROJECT_ROOT = FINAL_DIR.parent if FINAL_DIR.name == "FINAL" else FINAL_DIR

# Official cryptographic baseline manifest
MANIFEST = {
    "data/holdout_594.csv": "69c809da4c0f1b6314cb8a0f19f6f36b7a398e754c0f8540c983aa50f189c59b",
    "FINAL/models/anchor_catboost.joblib": "80cbb01ff4c9950aef99c7b97cad3d9ed2b8598738bae74a6d084b66a1ce51a9",
    "FINAL/models/anchor_pipeline.joblib": "d0c0120ed340c2552c61a4e91c322658e70a46eb5664d0e9a192f868f789d5c1",
    "FINAL/models/base_catboost.joblib": "409fcb1adef0f6f84340744ff88f2a03dade0a7ec078b3aeae530706c3f40315",
    "FINAL/models/base_lightgbm.joblib": "0ab6411a44c90d27a580fdac4314ff17a6257831addf74302fd2108e6ed5abc9",
    "FINAL/models/meta_logistic_regression.joblib": "f3f615b3c925633745c8cfee7f273e8974cde891cc44192ccd0e34461137084b",
    "FINAL/models/false_negative_recovery_catboost.joblib": "39ff32975b8ff5e369a81a0a4a653a44a3038c1d58ac325710e68b5b24e0c4d5",
    "FINAL/models/false_positive_recovery_catboost.joblib": "cce32be26924117eb278458259afa4a32a724ec8020597a690cea3fb63b38089",
    "FINAL/models/reference_pool.npz": "a1bfb14a72cbac22a5c7f4a0a02bd8b127e1aee2e0852f5e937278b2c69a1928",
    "FINAL/models/specialist_catboost.joblib": "c9fead0570607c6d4bd39639f3aa607985652009c7be180eac9385a85a277544",
    "FINAL/configs/preprocessing_schema.json": "424302fa567bcc63142b3a61b2e560b2259bea336c107f12f806b1fe598b37dd",
    "FINAL/models/preprocessor.joblib": "25175dc3ff8b18fd02a101153a87e1676ec5bd7f2ecc89bcd72aeb44dc8ec34f",
    "FINAL/models/hr_feature_engineer.joblib": "0772868eb01575415ad8ab2ee1a40d6a2cf6ea6775db03b11d4ddecae5b23fe9",
    "FINAL/verification/retest_step32_594_results.csv": "f5b18533bc2af7f290eb387f6a9c534275f7b1762f97cab894d4ce4c22bc1d0e",
}

def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 80)
    print("  CRYPTOGRAPHIC SHA-256 IMMUTABILITY VERIFICATION")
    print("=" * 80)

    all_matched = True
    for rel_path, expected_hash in MANIFEST.items():
        fpath = PROJECT_ROOT / rel_path
        if not fpath.exists() and rel_path.startswith("FINAL/"):
            fpath = FINAL_DIR / rel_path[len("FINAL/"):]
        if not fpath.exists() and rel_path == "data/holdout_594.csv":
            fpath = FINAL_DIR / "data" / "holdout_594.csv"
        if not fpath.exists() and "retest_step32_594_results.csv" in rel_path:
            fpath = FINAL_DIR / "results" / "retest_step32_594_results.csv"
        if not fpath.exists():
            print(f"  [MISSING] {rel_path}")
            all_matched = False
            continue

        actual_hash = hash_file(fpath)
        if actual_hash.lower() == expected_hash.lower():
            print(f"  [VERIFIED] {rel_path}")
        else:
            print(f"  [MISMATCH] {rel_path}")
            print(f"    Expected: {expected_hash}")
            print(f"    Actual:   {actual_hash}")
            all_matched = False

    print("=" * 80)
    if all_matched:
        print("  RESULT: SUCCESS -- ALL PROTECTED ARTIFACTS ARE CRYPTOGRAPHICALLY INTACT.")
    else:
        print("  RESULT: FAILED -- ONE OR MORE ARTIFACTS FAILED VERIFICATION.")
    print("=" * 80 + "\n")
    return all_matched

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
