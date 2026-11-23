"""
Blend ML ranking + docking results with a cluster cap -> submission_v2.txt.
"""
# ===== V2-C: BLEND + CLUSTER CAP -> submission_v2.txt =====
import numpy as np, pandas as pd, os
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator as rfg
from rdkit.ML.Cluster import Butina

W_ML, W_DOCK, W_HINGE = 0.5, 0.4, 0.1
MAX_AZ, MAX_PER_CLUSTER, CLUSTER_DIST = 10, 2, 0.6

AZ = Chem.MolFromSmarts("c1cnc2[nH]ccc2c1")
gen = rfg.GetMorganGenerator(radius=2, fpSize=2048)
ml = pd.read_csv("ranked_blind_set_v2.csv")                       # smiles, id, score
dk = pd.read_csv("dock_results.csv") if os.path.exists("dock_results.csv") else pd.DataFrame(columns=["id","aff","hinge"])
pool = pd.read_csv("dock_pool.csv")[["id"]]
cand = ml.merge(pool, on="id")                                    # only compounds in the docking pool
cand = cand.merge(dk, on="id", how="left")
has = cand.aff.notna()
print("pool:", len(cand), "| docked:", int(has.sum()))

# docked compounds: blend; undocked: ML only (scaled to the same 0-1 range, small penalty for no docking evidence)
cand["ml_r"] = cand.score.rank(pct=True)
cand["dock_r"] = np.nan
cand.loc[has, "dock_r"] = (-cand.loc[has, "aff"]).rank(pct=True)
cand["hinge"] = cand.hinge.fillna(0)
cand["final"] = np.where(has, W_ML * cand.ml_r + W_DOCK * cand.dock_r + W_HINGE * cand.hinge,
                              (W_ML + W_DOCK * 0.5) * cand.ml_r)   # undocked: assume average docking rank
cand["az"] = [Chem.MolFromSmiles(s).HasSubstructMatch(AZ) for s in cand.smiles]
cand["fp"] = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in cand.smiles]
cand = cand.sort_values("final", ascending=False).reset_index(drop=True)

# cluster the candidates; allow at most MAX_PER_CLUSTER picks per cluster
d = []
for i in range(1, len(cand)):
    d += [1 - x for x in DataStructs.BulkTanimotoSimilarity(cand.fp[i], list(cand.fp[:i]))]
cl_of = np.zeros(len(cand), int)
for ci, c in enumerate(Butina.ClusterData(d, len(cand), CLUSTER_DIST, isDistData=True)):
    cl_of[list(c)] = ci
cand["cluster"] = cl_of

ids, n_az, per = [], 0, {}
for r in cand.itertuples():
    if r.az and n_az >= MAX_AZ: continue
    if per.get(r.cluster, 0) >= MAX_PER_CLUSTER: continue
    ids.append(r.id); n_az += int(r.az); per[r.cluster] = per.get(r.cluster, 0) + 1
    if len(ids) == 100: break
if len(ids) < 100:                                                # relax the cluster cap if needed
    for r in cand.itertuples():
        if r.id in ids: continue
        if r.az and n_az >= MAX_AZ: continue
        ids.append(r.id); n_az += int(r.az)
        if len(ids) == 100: break

open("submission_v2.txt", "w").write("\n".join(ids) + "\n")
sel = cand[cand.id.isin(ids)]
print("IDs:", len(ids), "unique:", len(set(ids)), "| azaindoles:", n_az,
      "| clusters used:", sel.cluster.nunique(), "| docked among picks:", int(sel.aff.notna().sum()))
