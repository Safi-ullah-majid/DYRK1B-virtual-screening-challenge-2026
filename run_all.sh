#!/usr/bin/env bash
# Best-performing route (ML only, ~5-8 min). Run from this folder.
set -e
[ -d Drug-Design-workshop-challenge-2026 ] || git clone https://github.com/avnikonenko/Drug-Design-workshop-challenge-2026
python scripts/ml_ensemble.py            # -> submission_v3.txt
# Optional extras:
# python scripts/download_extra_data.py && python scripts/ml_extra_data.py                      # -> submission_v4b.txt
# python scripts/ml_v2_nonazaindole.py && python scripts/docking.py && python scripts/blend_docking.py   # -> submission_v2.txt
# python scripts/shape_screen.py         # -> submission_mix.txt (needs submission_v3.txt)
echo "Done. Submit submission_v3.txt (check: 100 lines, 100 unique IDs)."
