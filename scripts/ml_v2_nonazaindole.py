"""
ML v2: non-azaindole ranking + cluster-split CV. Writes ranked_blind_set_v2.csv.
Run from a folder containing Drug-Design-workshop-challenge-2026/data (git clone the public repo).
"""
# ===== V2-A: ML v2 (azaindole down-weighting + cluster-split CV) =====
import numpy as np, pandas as pd
from scipy.stats import rankdata
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator as rfg
from rdkit.ML.Cluster import Butina
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
RDLogger.DisableLog("rdApp.*")

D = "Drug-Design-workshop-challenge-2026/data/"
tr = pd.read_csv(D + "chembl_stereo_classes_fixed.smi", sep="\t", header=None, names=["smiles", "id", "act"])
bl = pd.read_csv(D + "blind_stereo.smi", sep="\t", header=None, names=["smiles", "id"])
tr["y"] = tr.act.astype(str).str.lower().isin(["1", "1.0", "active"]).astype(int)

morgan  = rfg.GetMorganGenerator(radius=2, fpSize=2048)
fmorgan = rfg.GetMorganGenerator(radius=2, fpSize=2048, atomInvariantsGenerator=rfg.GetMorganFeatureAtomInvGen())
apair   = rfg.GetAtomPairGenerator(fpSize=2048)
AZ = Chem.MolFromSmarts("c1cnc2[nH]ccc2c1")

def feats(smiles):
    bits, X, az, keep = [], [], [], []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None: continue
        bits.append(morgan.GetFingerprint(m)); keep.append(i); az.append(m.HasSubstructMatch(AZ))
        X.append(np.concatenate([morgan.GetCountFingerprintAsNumPy(m),
                                 fmorgan.GetCountFingerprintAsNumPy(m),
                                 apair.GetCountFingerprintAsNumPy(m)]).astype(np.float32))
    return bits, np.vstack(X), np.array(az), keep

tb, Xt, taz, kt = feats(tr.smiles); y = tr.y.values[kt]
bb, Xb, baz, kb = feats(bl.smiles)
bl = bl.iloc[kb].reset_index(drop=True)
print(f"train {len(y)} (actives {y.sum()}, azaindoles {taz.sum()}, azaindole actives {(taz & (y==1)).sum()})")

# cluster labels for honest CV (compounds in the same cluster never split across folds)
d = []
for i in range(1, len(tb)):
    d += [1 - s for s in DataStructs.BulkTanimotoSimilarity(tb[i], tb[:i])]
groups = np.zeros(len(tb), int)
for ci, c in enumerate(Butina.ClusterData(d, len(tb), 0.6, isDistData=True)):
    groups[list(c)] = ci

AZ_W = 0.3                                   # azaindole training weight
w = np.where(taz, AZ_W, 1.0)

def fit(X, y, w, seed=0):
    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    et = ExtraTreesClassifier(n_estimators=500, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    return [rf.fit(X, y, sample_weight=w), et.fit(X, y, sample_weight=w)]

def pred(models, X): return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)

for name, ww in [("equal weights", np.ones(len(y))), (f"azaindole weight {AZ_W}", w)]:
    aucs, aucs_nonaz = [], []
    for a, b in GroupKFold(4).split(Xt, y, groups):
        p = pred(fit(Xt[a], y[a], ww[a]), Xt[b])
        if len(set(y[b])) > 1: aucs.append(roc_auc_score(y[b], p))
        m_ = ~taz[b]
        if len(set(y[b][m_])) > 1: aucs_nonaz.append(roc_auc_score(y[b][m_], p[m_]))
    print(f"cluster-CV | {name:22s} AUC all {np.mean(aucs):.3f} | AUC non-azaindoles {np.mean(aucs_nonaz):.3f}")

# --- final scoring: non-azaindoles ranked by a model trained WITHOUT azaindoles ---
na = ~taz
models_na = fit(Xt[na], y[na], np.ones(na.sum()))
act_na   = [b for b, l, z in zip(tb, y, taz) if l == 1 and not z]
inact_na = [b for b, l, z in zip(tb, y, taz) if l == 0 and not z]
def sim_na(fp, k=3):
    a = np.sort(DataStructs.BulkTanimotoSimilarity(fp, act_na))[::-1][:k].mean()
    i = np.sort(DataStructs.BulkTanimotoSimilarity(fp, inact_na))[::-1][:k].mean()
    return a - 0.3 * i
ml_na = pred(models_na, Xb)
sm_na = np.array([sim_na(f) for f in bb])
s_na = (rankdata(ml_na) + rankdata(sm_na)) / (2 * len(bl))

# azaindoles keep the full model (they are capped later anyway)
models_all = fit(Xt, y, w)
ml_all = pred(models_all, Xb)
act = [b for b, l in zip(tb, y) if l == 1]; inact = [b for b, l in zip(tb, y) if l == 0]
def sim_all(fp, k=3):
    a = np.sort(DataStructs.BulkTanimotoSimilarity(fp, act))[::-1][:k].mean()
    i = np.sort(DataStructs.BulkTanimotoSimilarity(fp, inact))[::-1][:k].mean()
    return a - 0.3 * i
s_all = (rankdata(ml_all) + rankdata(np.array([sim_all(f) for f in bb]))) / (2 * len(bl))

score = pd.Series(np.nan, index=range(len(bl)))
idx_n, idx_a = np.where(~baz)[0], np.where(baz)[0]
score.iloc[idx_n] = rankdata(s_na[idx_n]) / len(idx_n)               # 0..1 among non-azaindoles
score.iloc[idx_a] = 0.5 + 0.5 * rankdata(s_all[idx_a]) / len(idx_a)  # 0.5..1 among azaindoles
bl["score"] = score.values
bl["az"] = baz
bl = bl.sort_values("score", ascending=False).reset_index(drop=True)
bl.drop(columns="az").to_csv("ranked_blind_set_v2.csv", index=False)
print("saved ranked_blind_set_v2.csv | blind non-azaindoles:", len(idx_n), "| azaindoles:", len(idx_a))
print("azaindoles in top 100:", int(bl.head(100).az.sum()), "| top 300:", int(bl.head(300).az.sum()))
