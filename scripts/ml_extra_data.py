"""
ML v4b: v3 ensemble + extra kinase data (weight 0.15). Run download_extra_data.py first.
Writes ranked_blind_set_v4b.csv and submission_v4b.txt.
"""
# ===== V4: ML ENSEMBLE + EXTRA KINASE DATA -> submission_v4b.txt =====
import numpy as np, pandas as pd, warnings
from scipy.stats import rankdata
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as rfg
from rdkit.ML.Cluster import Butina
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
RDLogger.DisableLog("rdApp.*"); warnings.filterwarnings("ignore")

MAX_AZ, MAX_PER_CLUSTER = 10, 3

D = "Drug-Design-workshop-challenge-2026/data/"
tr = pd.read_csv(D + "chembl_stereo_classes_fixed.smi", sep="\t", header=None, names=["smiles", "id", "act"])
bl = pd.read_csv(D + "blind_stereo.smi", sep="\t", header=None, names=["smiles", "id"])
tr["y"] = tr.act.astype(str).str.lower().isin(["1", "1.0", "active"]).astype(int)

import os
USE_EXTRA, EXTRA_W = os.path.exists("extra_train.csv"), 0.15   # was 0.5
N_ORIG = len(tr)
if USE_EXTRA:
    ex = pd.read_csv("extra_train.csv")
    known = set(Chem.MolToSmiles(m) for m in (Chem.MolFromSmiles(s_) for s_ in tr.smiles) if m)
    ex = ex[[ (m is not None) and (Chem.MolToSmiles(m) not in known)
              for m in (Chem.MolFromSmiles(s_) for s_ in ex.smiles)]]
    tr = pd.concat([tr, ex.assign(act=ex.y)[["smiles", "id", "act", "y"]]], ignore_index=True)
    print("extra rows added:", len(ex), "| extra actives:", int(ex.y.sum()))
else:
    print("no extra_train.csv -> training on the challenge ChEMBL set only")

g2 = rfg.GetMorganGenerator(radius=2, fpSize=2048)
g3 = rfg.GetMorganGenerator(radius=3, fpSize=2048)
gf = rfg.GetMorganGenerator(radius=2, fpSize=2048, atomInvariantsGenerator=rfg.GetMorganFeatureAtomInvGen())
gp = rfg.GetAtomPairGenerator(fpSize=2048)
AZ = Chem.MolFromSmarts("c1cnc2[nH]ccc2c1")

def feats(smiles):
    bits, A, B, az, keep = [], [], [], [], []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None: continue
        keep.append(i); bits.append(g2.GetFingerprint(m)); az.append(m.HasSubstructMatch(AZ))
        c2 = g2.GetCountFingerprintAsNumPy(m)
        A.append(np.concatenate([c2, gf.GetCountFingerprintAsNumPy(m), gp.GetCountFingerprintAsNumPy(m),
                                 g3.GetCountFingerprintAsNumPy(m)]).astype(np.float32))
        B.append((c2 > 0).astype(np.float32))
    return bits, np.vstack(A), np.vstack(B), np.array(az), keep

tb, At, Bt, taz, kt = feats(tr.smiles); y = tr.y.values[kt]
is_extra = np.array(kt) >= N_ORIG
bb, Ab, Bb, baz, kb = feats(bl.smiles); bl = bl.iloc[kb].reset_index(drop=True)
na = ~taz
W = np.where(is_extra, EXTRA_W, 1.0)
print(f"train {len(y)} | blind {len(bl)} | train azaindoles {taz.sum()} | blind azaindoles {baz.sum()}")

# cluster labels for honest CV
n0 = int((~is_extra).sum())          # cluster only the original rows (extras never appear in test folds)
d = []
for i in range(1, n0): d += [1 - s_ for s_ in DataStructs.BulkTanimotoSimilarity(tb[i], tb[:i])]
groups = np.full(len(tb), -1, int)
for ci, c in enumerate(Butina.ClusterData(d, n0, 0.6, isDistData=True)): groups[list(c)] = ci

def knn_sim(q_bits, a_bits, i_bits, k=3):
    out = []
    for fp in q_bits:
        a = np.sort(DataStructs.BulkTanimotoSimilarity(fp, a_bits))[::-1][:k].mean()
        i = np.sort(DataStructs.BulkTanimotoSimilarity(fp, i_bits))[::-1][:k].mean()
        out.append(a - 0.3 * i)
    return np.array(out)

def run_models(tr_idx, q_A, q_B, q_bits):
    """train on tr_idx (non-azaindole training rows), return dict of model -> scores for queries"""
    yy = y[tr_idx]; out = {}
    out["rf"]  = RandomForestClassifier(500, class_weight="balanced_subsample", n_jobs=-1, random_state=0).fit(At[tr_idx], yy, sample_weight=W[tr_idx]).predict_proba(q_A)[:, 1]
    out["et"]  = ExtraTreesClassifier(500, class_weight="balanced_subsample", n_jobs=-1, random_state=0).fit(At[tr_idx], yy, sample_weight=W[tr_idx]).predict_proba(q_A)[:, 1]
    out["hgb"] = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_depth=6, random_state=0).fit(Bt[tr_idx], yy, sample_weight=W[tr_idx]).predict_proba(q_B)[:, 1]
    out["lr"]  = LogisticRegression(C=0.05, max_iter=500, class_weight="balanced").fit(Bt[tr_idx], yy, sample_weight=W[tr_idx]).predict_proba(q_B)[:, 1]
    a_bits = [tb[i] for i in tr_idx if y[i] == 1]; i_bits = [tb[i] for i in tr_idx if y[i] == 0]
    out["knn"] = knn_sim(q_bits, a_bits, i_bits)
    return out

def rank_avg(scores, names): return np.mean([rankdata(scores[n]) / len(scores[n]) for n in names], axis=0)
def recall_top(y_, s, frac=0.05):
    k = max(1, int(frac * len(s))); return y_[np.argsort(-s)[:k]].sum() / max(1, y_.sum())

# ---- cluster-split CV on non-azaindoles: which models / ensemble work best ----
idx_na = np.where(na & ~is_extra)[0]
idx_extra = np.where(na & is_extra)[0]
res = {n: [] for n in ["rf", "et", "hgb", "lr", "knn", "ENSEMBLE"]}; rec = {n: [] for n in res}
for a, b in GroupKFold(4).split(idx_na, y[idx_na], groups[idx_na]):
    tr_i, te_i = np.concatenate([idx_na[a], idx_extra]), idx_na[b]
    sc = run_models(tr_i, At[te_i], Bt[te_i], [tb[i] for i in te_i])
    sc["ENSEMBLE"] = rank_avg(sc, ["rf", "et", "hgb", "lr", "knn"])
    for n in res:
        res[n].append(roc_auc_score(y[te_i], sc[n])); rec[n].append(recall_top(y[te_i], sc[n]))
print("cluster-CV (non-azaindoles):")
for n in res: print(f"  {n:9s} AUC {np.mean(res[n]):.3f} | recall@top5% {np.mean(rec[n]):.3f}")

# ---- final scoring of the blind set ----
sc = run_models(np.concatenate([idx_na, idx_extra]), Ab, Bb, bb)
ens = rank_avg(sc, ["rf", "et", "hgb", "lr", "knn"])
bl["score"] = ens; bl["az"] = baz
# azaindoles: ranked among themselves, placed mid-high (they are capped at MAX_AZ below)
a_idx = np.where(baz)[0]
bl.loc[a_idx, "score"] = 0.5 + 0.5 * rankdata(ens[a_idx]) / len(a_idx) * 0.8
bl = bl.sort_values("score", ascending=False).reset_index(drop=True)
bl.drop(columns="az").to_csv("ranked_blind_set_v4b.csv", index=False)

# ---- select 100: cluster cap + azaindole cap ----
top = bl.head(600).reset_index(drop=True)
fps = [g2.GetFingerprint(Chem.MolFromSmiles(s)) for s in top.smiles]
d = []
for i in range(1, len(fps)): d += [1 - s for s in DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])]
cl_of = np.zeros(len(top), int)
for ci, c in enumerate(Butina.ClusterData(d, len(top), 0.6, isDistData=True)): cl_of[list(c)] = ci
ids, n_az, per = [], 0, {}
for r, cid in zip(top.itertuples(), cl_of):
    if r.az and n_az >= MAX_AZ: continue
    if per.get(cid, 0) >= MAX_PER_CLUSTER: continue
    ids.append(r.id); n_az += int(r.az); per[cid] = per.get(cid, 0) + 1
    if len(ids) == 100: break
open("submission_v4b.txt", "w").write("\n".join(ids) + "\n")
print("saved submission_v4b.txt |", len(ids), "unique", len(set(ids)), "| azaindoles", n_az, "| clusters", len(per))

# overlap with your previous list, if present
import os
if os.path.exists("submission_v2.txt"):
    print("overlap with submission_v2.txt:", len(set(ids) & set(open("submission_v2.txt").read().split())))
