# 9th Advanced Drug Design workshop challenge - Olomouc 2026

## DYRK1B Virtual Screening Challenge

Ligand-based virtual screening pipeline (Google Colab) that picks 100 likely DYRK1B inhibitors from a blind set of 3,365 compounds. The final list is built from an ML ensemble trained on ChEMBL data, with explicit handling of an azaindole bias found in the training set.

Challenge info: https://shiny.imtm.cz/apps/polishp/9add/

Team: _<Team 65 / Safi Ullah Majid>_

## Key finding

The training set (1,891 compounds, 734 active at IC50/Ki/Kd <= 1 uM) is dominated by 7-azaindoles: 101 of its 118 azaindoles are active. A plain ML model therefore ranks every azaindole highest. The blind set contains 244 azaindoles, and a list with 89 of them in the top 100 scored **0**.

Fix that gave the first real result (score 0 -> 0.33):

- rank non-azaindoles with a model trained **without** azaindoles
- allow at most 10 azaindoles in the final 100
- allow at most 3 compounds per structural cluster

## Requirements

- Google Colab (CPU is enough for all ML notebooks)
- Python 3.10+, RDKit, pandas, numpy, scipy, scikit-learn
- `requests` (only for the ChEMBL / PDB download steps)
- Optional, for docking: AutoDock Vina 1.2.5 binary, Open Babel
- Data: `chembl_stereo_classes_fixed.smi` (train) and `blind_stereo.smi` (blind, 3,365 compounds) from the public repo `avnikonenko/Drug-Design-workshop-challenge-2026`

## Inputs

- Training: tab-separated `.smi`, no header: `smiles<TAB>id<TAB>act` (`1` = active)
- Blind: tab-separated `.smi`, no header: `smiles<TAB>id` (IDs `MOL_1` ... `MOL_3365`)
- Note: the `dataset.smi` file from the challenge download had only 3,315 compounds (`MOL_3316`-`MOL_3365` missing). Use the full 3,365-compound blind set.

## Notebooks

| File | What it does | Output |
|---|---|---|
| `dyrk1b_full_colab.ipynb` | v1: ML ranking + Vina docking into 8C2Z + blend | `submission_final.txt` |
| `dyrk1b_v2_colab.ipynb` | v2: non-azaindole model, cluster-split CV, docking, cluster cap | `submission_v2.txt` |
| `dyrk1b_v3_colab.ipynb` | v3: ML-only ensemble (RF, ET, HGB, LR, similarity) | `submission_v3.txt` |
| `dyrk1b_v4_colab.ipynb` | v4: v3 + extra DYRK1A / DYRK2 / CLK1 data from ChEMBL | `submission_v4.txt` |
| `dyrk1b_shape_colab.ipynb` | Shape / pharmacophore similarity to known DYRK ligands | `submission_mix.txt`, `submission_shape_only.txt` |

Run the cells of a notebook from top to bottom. Each notebook starts with its own install cell.

## Method (v3, best-performing family)

1. **Standardise** SMILES with RDKit; flag azaindoles with the SMARTS `c1cnc2[nH]ccc2c1`.
2. **Features:** Morgan counts (radius 2 and 3), feature-Morgan counts, atom-pair counts; binary Morgan bits for linear/boosted models.
3. **Models (trained on non-azaindoles):** Random Forest, Extra Trees, HistGradientBoosting, Logistic Regression, and a kNN-style similarity score (mean top-3 Tanimoto to actives minus 0.3 x mean top-3 to inactives).
4. **Ensemble:** average of per-model ranks.
5. **Validation:** Butina-cluster (Tanimoto distance 0.6) split, 4 folds, reporting ROC-AUC and recall in the top 5%.
6. **Azaindoles:** ranked separately and placed mid-high; capped at 10 in the final list.
7. **Selection:** top 600 by score, then at most 3 per cluster, at most 10 azaindoles, 100 unique IDs.

### Cross-validation (non-azaindoles, cluster split)

| Model | ROC-AUC | Recall @ top 5% |
|---|---|---|
| Random Forest | 0.777 | 0.131 |
| Extra Trees | 0.783 | 0.126 |
| HistGradientBoosting | 0.743 | 0.104 |
| Logistic Regression | 0.762 | 0.126 |
| Similarity (kNN) | 0.727 | 0.132 |
| Ensemble | 0.782 | 0.129 |

These come from ChEMBL-only data; the blind set is a different chemistry, so they do not predict the leaderboard score.

## Leaderboard history

| Approach | Score |
|---|---|
| ML on all ChEMBL data (89 azaindoles in top 100) | 0 |
| ML with azaindoles limited + diversity + some docking | 0.33 |
| Heavier docking weight (0.6) + hinge-contact bonus | 0.27 |
| 70 ML picks + 30 docked "novel" hinge-binder picks | 0.2 |
| v2 / v3 ML ensemble (cluster cap) | 0.33 |
| v4 / v4b: + DYRK1A, DYRK2, CLK1 data | 0.33 |
| 70 ML picks + 30 shape / pharmacophore picks | 0.27 |

The scores are consistent with about 15 hidden actives (0.33 = 5 found, 0.27 = 4, 0.2 = 3), so one compound is about 0.067 and small differences are noise. This is an inference from the scores, not a confirmed number.

## What did not help

- **Docking** (Vina into 8C2Z, grid centred on AZ191): separated known actives from inactives in a 20 vs 20 test (AUC 0.79) but did not improve the blind-set score.
- **Hinge-contact bonus:** 8 of the first 10 docked compounds made a hinge contact, so the filter was not selective.
- **Extra kinase data:** raised cluster-CV AUC slightly (0.78 -> 0.80) but lowered recall in the top 5%; leaderboard unchanged.
- **Shape / pharmacophore screen** (ErG, Gobbi 2D pharmacophore, USRCAT): swapped-in picks found no extra actives.
- **Chemprop D-MPNN:** the repo's runner targets Chemprop v1 and needs a GPU; not completed.

## Outputs

- `submission_*.txt`: 100 compound IDs, one per line, no header (this is the file to submit)
- `ranked_blind_set*.csv`: full blind-set ranking (`smiles, id, score`)
- `dock_results.csv`: Vina affinity and hinge flag per docked compound
- `shape_scores.csv`: ErG / pharmacophore / USRCAT scores per blind compound
- `extra_train.csv`: extra ChEMBL training rows (v4 notebook)

## Reproducing the best result

1. Open `dyrk1b_v3_colab.ipynb` in Colab and run both cells (about 5-8 min).
2. Download `submission_v3.txt`.
3. Check: 100 lines, 100 unique IDs.
4. Upload to the challenge page with your team token.

## Limitations

- Models are trained only on public ChEMBL data and miss actives that look unlike known kinase chemistry.
- With a small number of hidden actives, leaderboard differences of one compound are within noise.
- The docking, PDB download and Chemprop steps depend on external services and package versions; they may need adjustment as those change.

## Acknowledgements

Training and blind data and the baseline pipeline structure come from the public repository `avnikonenko/Drug-Design-workshop-challenge-2026`. Challenge organised by the 9th Advanced Drug Design workshop (Olomouc 2026), with ELIXIR and RowanSci.
