"""
Shape / pharmacophore similarity to known DYRK ligands (ErG, Gobbi 2D pharmacophore, USRCAT).
Writes shape_scores.csv, submission_mix.txt, submission_shape_only.txt.
"""
# ===== SHAPE / PHARMACOPHORE SCREEN vs known DYRK ligands (scaffold-hopping signal) =====
import os, json, time, requests, numpy as np, pandas as pd
from scipy.stats import rankdata
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors, rdReducedGraphs, Descriptors
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate
from rdkit.Chem import rdFingerprintGenerator as rfg
from rdkit.ML.Cluster import Butina
RDLogger.DisableLog("rdApp.*")

BASE_FILE   = "submission_v3.txt"     # your best ML list (100 IDs); the mix keeps most of it
N_KEEP, N_NEW = 70, 30                # mix = N_KEEP from BASE_FILE + N_NEW from this screen
UNIPROT     = ["Q13627", "Q9Y463"]    # DYRK1A, DYRK1B
D = "Drug-Design-workshop-challenge-2026/data/"

# ---------- 1. reference ligands: co-crystal ligands of DYRK1A/1B from the PDB ----------
def post(url, payload):
    try:
        r = requests.post(url, json=payload, timeout=60); return r.json() if r.status_code == 200 else None
    except Exception: return None
def get(url):
    try:
        r = requests.get(url, timeout=60); return r.json() if r.status_code == 200 else None
    except Exception: return None

def pdb_ligands():
    refs = {}
    for acc in UNIPROT:
        q = {"query": {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                "operator": "exact_match", "value": acc}},
             "return_type": "entry", "request_options": {"paginate": {"start": 0, "rows": 300}}}
        j = post("https://search.rcsb.org/rcsbsearch/v2/query", q)
        ids = [x["identifier"] for x in (j or {}).get("result_set", [])]
        print(acc, "PDB entries:", len(ids))
        for pid in ids:
            e = get(f"https://data.rcsb.org/rest/v1/core/entry/{pid}")
            for ent in ((e or {}).get("rcsb_entry_container_identifiers", {}).get("non_polymer_entity_ids") or []):
                n = get(f"https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{pid}/{ent}")
                comp = ((n or {}).get("pdbx_entity_nonpoly") or {}).get("comp_id")
                if not comp or comp in refs: continue
                c = get(f"https://data.rcsb.org/rest/v1/core/chemcomp/{comp}")
                smi = ((c or {}).get("rcsb_chem_comp_descriptor") or {}).get("smiles")
                if smi: refs[comp] = smi
    out = {}
    for comp, smi in refs.items():
        m = Chem.MolFromSmiles(smi)
        if m and 15 <= m.GetNumHeavyAtoms() <= 45 and 200 < Descriptors.MolWt(m) < 650:   # drop ions, solvents, big cofactors
            out[comp] = Chem.MolToSmiles(m)
    return out

try:
    refs = pdb_ligands()
except Exception as ex:
    print("PDB download failed:", ex); refs = {}
refs["HARMINE"] = "COc1ccc2c(c1)[nH]c1c(C)nccc12"
print("reference ligands:", len(refs))

# fallback / supplement: a few diverse known actives from the training set (non-azaindole)
tr = pd.read_csv(D + "chembl_stereo_classes_fixed.smi", sep="\t", header=None, names=["smiles", "id", "act"])
tr["y"] = tr.act.astype(str).str.lower().isin(["1", "1.0", "active"]).astype(int)
AZ = Chem.MolFromSmarts("c1cnc2[nH]ccc2c1")
ta = tr[tr.y == 1].copy()
ta = ta[[not Chem.MolFromSmiles(s).HasSubstructMatch(AZ) for s in ta.smiles]]
g2 = rfg.GetMorganGenerator(radius=2, fpSize=2048)
fp_ta = [g2.GetFingerprint(Chem.MolFromSmiles(s)) for s in ta.smiles]
d = []
for i in range(1, len(fp_ta)): d += [1 - x for x in DataStructs.BulkTanimotoSimilarity(fp_ta[i], fp_ta[:i])]
cl = Butina.ClusterData(d, len(fp_ta), 0.7, isDistData=True)
for k, c in enumerate(sorted(cl, key=len, reverse=True)[:25]):
    refs[f"TRAIN{k}"] = ta.smiles.iloc[c[0]]
print("total references (PDB + harmine + 25 diverse training actives):", len(refs))

# ---------- 2. descriptors for refs and blind set ----------
def embed(m, n=1):
    mh = Chem.AddHs(m)
    cids = AllChem.EmbedMultipleConfs(mh, numConfs=n, randomSeed=11)
    if len(cids) == 0: return None
    try: AllChem.MMFFOptimizeMoleculeConfs(mh, maxIters=100)
    except Exception: pass
    return mh, list(cids)

def erg(m): return rdReducedGraphs.GetErGFingerprint(m)
def erg_sim(a, b): return np.minimum(a, b).sum() / max(np.maximum(a, b).sum(), 1e-9)
def pharm(m): return Generate.Gen2DFingerprint(m, Gobbi_Pharm2D.factory)
def usr_list(mh, cids): return [rdMolDescriptors.GetUSRCAT(mh, confId=c) for c in cids]

R = {}
for name, smi in refs.items():
    m = Chem.MolFromSmiles(smi); e = embed(m, 5)
    if e: R[name] = dict(erg=erg(m), ph=pharm(m), usr=usr_list(*e))
print("references usable:", len(R))

bl = pd.read_csv(D + "blind_stereo.smi", sep="\t", header=None, names=["smiles", "id"])
t0 = time.time(); rows = []
for i, (s, mid) in enumerate(zip(bl.smiles, bl.id)):
    m = Chem.MolFromSmiles(s)
    if m is None: rows.append((mid, 0, 0, 0)); continue
    ge, gp = erg(m), pharm(m)
    e = embed(m, 1)
    u = usr_list(*e)[0] if e else None
    es = max(erg_sim(ge, r["erg"]) for r in R.values())
    ps = max(DataStructs.TanimotoSimilarity(gp, r["ph"]) for r in R.values())
    us = max((rdMolDescriptors.GetUSRScore(u, ru) for r in R.values() for ru in r["usr"]), default=0) if u is not None else 0
    rows.append((mid, es, ps, us))
    if i % 500 == 0: print(i, f"{time.time()-t0:.0f}s")
sc = pd.DataFrame(rows, columns=["id", "erg", "pharm", "usr"]).merge(bl, on="id")
sc["shape_score"] = (rankdata(sc.erg) + rankdata(sc.pharm) + rankdata(sc.usr)) / (3 * len(sc))
sc["az"] = [Chem.MolFromSmiles(s).HasSubstructMatch(AZ) if Chem.MolFromSmiles(s) else False for s in sc.smiles]
sc = sc.sort_values("shape_score", ascending=False).reset_index(drop=True)
sc.to_csv("shape_scores.csv", index=False)
print("saved shape_scores.csv")

# ---------- 3. lists ----------
only = sc[~sc.az].head(100)
open("submission_shape_only.txt", "w").write("\n".join(only.id) + "\n")

if os.path.exists(BASE_FILE):
    base = open(BASE_FILE).read().split()[:N_KEEP]
    new = [i for i in sc[~sc.az].id if i not in set(base)]
    # cluster-dedupe the new picks so they cover different chemotypes
    smap = dict(zip(sc.id, sc.smiles)); chosen, fps = [], []
    for i in new:
        fp = g2.GetFingerprint(Chem.MolFromSmiles(smap[i]))
        if fps and max(DataStructs.BulkTanimotoSimilarity(fp, fps)) > 0.6: continue
        chosen.append(i); fps.append(fp)
        if len(chosen) == N_NEW: break
    mix = base + chosen
    assert len(mix) == len(set(mix)) == 100, (len(mix), len(set(mix)))
    open("submission_mix.txt", "w").write("\n".join(mix) + "\n")
    print(f"submission_mix.txt: {N_KEEP} from {BASE_FILE} + {N_NEW} from shape/pharmacophore screen")
else:
    print(BASE_FILE, "not found - only submission_shape_only.txt written")

try:
    from google.colab import files
    for f in ["submission_mix.txt", "submission_shape_only.txt"]:
        if os.path.exists(f): files.download(f)
except Exception: pass
