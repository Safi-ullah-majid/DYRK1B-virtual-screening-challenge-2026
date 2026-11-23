"""
Download DYRK1A / DYRK2 / CLK1 activities from ChEMBL -> extra_train.csv (needs internet).
"""
# ===== EXTRA DATA: DYRK1A (+ related kinases) actives/inactives from ChEMBL =====
import requests, pandas as pd, numpy as np, time
from rdkit import Chem

BASE = "https://www.ebi.ac.uk/chembl/api/data"
GENES = ["DYRK1A", "DYRK2", "CLK1"]          # DYRK1A = closest to DYRK1B

def get(url, params=None, tries=3):
    for t in range(tries):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 200: return r.json()
        except Exception as e:
            err = e
        time.sleep(2)
    return None

def target_ids(gene):
    j = get(f"{BASE}/target/search.json", {"q": gene, "limit": 20})
    if not j: return []
    out = []
    for t in j.get("targets", []):
        if t.get("organism") == "Homo sapiens" and t.get("target_type") == "SINGLE PROTEIN" \
           and any(gene.lower() == (c.get("component_description") or "").lower() or
                   gene.lower() in str(t.get("pref_name", "")).lower() or
                   any(gene.lower() == (s.get("component_synonym") or "").lower()
                       for s in c.get("target_component_synonyms", []))
                   for c in t.get("target_components", [])):
            out.append(t["target_chembl_id"])
    return out[:1]

rows = []
for gene in GENES:
    tids = target_ids(gene)
    print(gene, "->", tids)
    for tid in tids:
        url, params = f"{BASE}/activity.json", {"target_chembl_id": tid, "standard_type__in": "IC50,Ki,Kd",
                                                "standard_units": "nM", "limit": 1000}
        n = 0
        while url:
            j = get(url, params); params = None
            if not j: break
            for a in j["activities"]:
                if a.get("canonical_smiles") and a.get("standard_value") and a.get("standard_relation") in ("=", "<", "<="):
                    rows.append((gene, a["molecule_chembl_id"], a["canonical_smiles"], float(a["standard_value"])))
                    n += 1
            nxt = j["page_meta"].get("next")
            url = ("https://www.ebi.ac.uk" + nxt) if nxt else None
        print("  ", tid, "activities:", n)

ex = pd.DataFrame(rows, columns=["gene", "id", "smiles", "nM"])
if len(ex):
    g = ex.groupby(["gene", "id", "smiles"], as_index=False).nM.median()
    g["y"] = np.where(g.nM <= 1000, 1, np.where(g.nM >= 10000, 0, -1))      # drop 1-10 uM grey zone
    g = g[g.y >= 0].copy()
    g["can"] = [Chem.MolToSmiles(m) if (m := Chem.MolFromSmiles(s)) else None for s in g.smiles]
    g = g.dropna(subset=["can"])
    # one row per structure; if a structure is active on any of the genes keep active
    g = g.sort_values("y", ascending=False).drop_duplicates("can")
    g[["smiles", "id", "y", "gene"]].to_csv("extra_train.csv", index=False)
    print("saved extra_train.csv:", len(g), "| actives:", int(g.y.sum()), "|", g.gene.value_counts().to_dict())
else:
    print("NO extra data downloaded - continue without it (v3 works without extra_train.csv)")
