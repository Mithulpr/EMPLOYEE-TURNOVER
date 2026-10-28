# Archive Directory

This directory stores archived historical models and unselected experiment outputs preserved for auditability and provenance.

---

## Contents

- `legacy/`: Preserves the 12 model weight files under their legacy file aliases (`step37_catboost_base.joblib`, `step27_anchor_catboost.joblib`, etc.). These files have been verified via SHA-256 to be exact byte-for-byte duplicates of the active production models in `FINAL/models/`. They are retained here strictly to guarantee backward compatibility with historical experiment scripts.
- `unselected_experiments/`: Reserved location for unselected experimental scripts or notes that do not belong to the 5 official versions.

*Notice: The archive is strictly historical and does not serve as an active model collection.*
