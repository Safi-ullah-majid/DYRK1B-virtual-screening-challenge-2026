"""
Receptor prep (8C2Z), docking box, hinge atoms, docking pool, then AutoDock Vina docking.
Needs: ./vina binary (v1.2.5), Open Babel, 8C2Z.pdb, ranked_blind_set_v2.csv.
Writes dock_pool.csv and dock_results.csv. Edit N_DOCK / EXHAUST / TIME_LIMIT in the docking section.
"""
# ===== 3. RECEPTOR, DOCKING BOX, HINGE ATOMS, DOCKING POOL =====
import numpy as np, pandas as pd, subprocess
from collections import defaultdict
from rdkit import Chem

lines = [l for l in open("8C2Z.pdb") if l.startswith(("ATOM", "HETATM"))]
xyz = lambda l: np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])

prot = [l for l in lines if l.startswith("ATOM") and l[21] == "A" and l[16] in " A"]
open("rec_clean.pdb", "w").write("".join(prot) + "END\n")
subprocess.run("obabel rec_clean.pdb -O rec.pdbqt -xr -p 7.4", shell=True, check=True)

skip = {"HOH","DMS","GOL","EDO","SO4","PO4","CL","NA","MG","ZN","K","CA","PEG","ACT"}
groups = defaultdict(list)
for l in lines:
    if l.startswith("HETATM") and l[17:20].strip() not in skip:
        groups[(l[17:20], l[21], l[22:26])].append(l)
key = max(groups, key=lambda k: len(groups[k]))
lig = groups[key]
lc = np.array([xyz(l) for l in lig])
center = lc.mean(0)
size = np.minimum(np.maximum(lc.max(0) - lc.min(0) + 12, 20), 22)   # smaller box = faster

lig_no = np.array([xyz(l) for l in lig if l[76:78].strip() in ("N", "O")])
hinge = np.array([xyz(l) for l in prot if l[12:16].strip() in ("N", "O")
                  and np.min(np.linalg.norm(lig_no - xyz(l), axis=1)) < 3.4])
print("reference ligand:", key, len(lig), "atoms | center", center.round(1), "box", size.round(0))
print("hinge atoms found:", len(hinge), "(need >= 2)")

# docking pool (v2): top 500 non-azaindoles + best 20 azaindoles from the v2 ranking
r = pd.read_csv("ranked_blind_set_v2.csv")
azq = Chem.MolFromSmarts("c1cnc2[nH]ccc2c1")
r["az"] = [Chem.MolFromSmiles(s).HasSubstructMatch(azq) for s in r.smiles]
pool = pd.concat([r[~r.az].head(500), r[r.az].head(20)]).sort_values("score", ascending=False)
pool[["smiles", "id", "score"]].to_csv("dock_pool.csv", index=False)
print("pool:", len(pool), "| azaindoles:", int(pool.az.sum()))

# ===== V2-B2. DOCKING (reuses dock_results.csv; skips docked ids) (stops itself at TIME_LIMIT; re-run to continue) =====
import os, time, subprocess, numpy as np, pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFingerprintGenerator as rfg
from rdkit.ML.Cluster import Butina

N_DOCK     = 150        # how many compounds to dock in total
EXHAUST    = 1          # 1 = fast, 4 = more accurate but ~4x slower
TIME_LIMIT = 15 * 60    # seconds per run of this cell

pool = pd.read_csv("dock_pool.csv").reset_index(drop=True)
gen = rfg.GetMorganGenerator(radius=2, fpSize=2048)
fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in pool.smiles]
d = []
for i in range(1, len(fps)):
    d += [1 - x for x in DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])]
cl = Butina.ClusterData(d, len(fps), 0.65, isDistData=True)
reps = sorted(min(c) for c in cl)                       # best member of each cluster first
order = reps + [i for i in range(len(pool)) if i not in set(reps)]
todo = pool.iloc[order[:N_DOCK]]

def prep(smi):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    if AllChem.EmbedMolecule(m, randomSeed=7) != 0: return False
    try: AllChem.MMFFOptimizeMolecule(m, maxIters=200)
    except Exception: pass
    Chem.MolToMolFile(m, "lig.mol")
    if os.path.exists("lig.pdbqt"): os.remove("lig.pdbqt")
    subprocess.run("obabel lig.mol -O lig.pdbqt", shell=True, capture_output=True)
    return os.path.exists("lig.pdbqt") and os.path.getsize("lig.pdbqt") > 0

def dock():
    if os.path.exists("out.pdbqt"): os.remove("out.pdbqt")
    cmd = ["./vina", "--receptor", "rec.pdbqt", "--ligand", "lig.pdbqt",
           "--center_x", str(center[0]), "--center_y", str(center[1]), "--center_z", str(center[2]),
           "--size_x", str(size[0]), "--size_y", str(size[1]), "--size_z", str(size[2]),
           "--exhaustiveness", str(EXHAUST), "--num_modes", "3", "--cpu", "2",
           "--seed", "7", "--out", "out.pdbqt"]
    subprocess.run(cmd, capture_output=True, timeout=600)
    if not os.path.exists("out.pdbqt"): return None
    txt = open("out.pdbqt").read()
    res = [l for l in txt.splitlines() if l.startswith("REMARK VINA RESULT")]
    return (float(res[0].split()[3]), txt) if res else None

def hinge_hit(poses):
    for model in poses.split("ENDMDL"):
        pts = [xyz(l) for l in model.splitlines()
               if l.startswith(("ATOM", "HETATM")) and l.split()[-1] in ("N","NA","NS","OA","OS")]
        if pts and len(hinge) and np.linalg.norm(hinge[:, None] - np.array(pts)[None], axis=2).min() < 3.5:
            return 1
    return 0

done = pd.read_csv("dock_results.csv") if os.path.exists("dock_results.csv") else pd.DataFrame(columns=["id","aff","hinge"])
rows, seen = done.to_dict("records"), set(done.id)
print("already docked:", len(seen))

t0 = time.time()
for k, r_ in enumerate(todo.itertuples()):
    if r_.id in seen: continue
    if time.time() - t0 > TIME_LIMIT:
        print("time limit reached - re-run this cell to continue"); break
    try:
        if not prep(r_.smiles): continue
        out = dock()
        if out is None: continue
        rows.append({"id": r_.id, "aff": out[0], "hinge": hinge_hit(out[1])})
    except Exception as e:
        print(r_.id, "failed:", str(e)[:60])
    if k % 5 == 0:
        pd.DataFrame(rows).to_csv("dock_results.csv", index=False)
        print(f"{k+1}/{len(todo)}  {time.time()-t0:.0f}s  docked so far: {len(rows)}")
pd.DataFrame(rows).to_csv("dock_results.csv", index=False)
print("docked total:", len(rows))
