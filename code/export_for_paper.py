# export_for_paper.py -- collects every result file the paper needs into ONE folder and ONE zip. Read-only: trains nothing, changes nothing.
import os, sys, glob, json, shutil, zipfile
import numpy as np, pandas as pd
OUT = os.environ["SEEG_OUT_DIR"]                 # set SEEG_OUT_DIR before running (folder with the extracted features)
ROOT = os.environ.get("SEEG_ROOT", os.path.dirname(os.path.abspath(OUT)))   # where paper_export/ and paper_export.zip are written
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))               # code folder
import stageA_report as R

DEST = os.path.join(ROOT, "paper_export")
os.makedirs(DEST, exist_ok=True)

# 1) Stage A: one row per model x lr x seed x participant
dfA = R.load_runs(OUT)
dfA.to_csv(os.path.join(DEST, "stageA_long.csv"), index=False)
print("Stage A rows:", len(dfA), "| models:", sorted(dfA["model"].unique()), "| seeds:", sorted(dfA["seed"].unique()))

# 2) Stage B: one row per model x participant x fold
rows = []
for p in sorted(glob.glob(os.path.join(OUT, "stageB", "*.json"))):
    j = json.load(open(p))
    if "per_subject" not in j or "fold" not in j: continue
    for sub, r in j["per_subject"].items():
        rows.append(dict(model=j["model"], lr=j.get("lr", np.nan), seed=j.get("seed", -1), subject=sub, fold=j["fold"], **r))
dfB = pd.DataFrame(rows)
dfB.to_csv(os.path.join(DEST, "stageB_long.csv"), index=False)
print("Stage B rows:", len(dfB), "| models:", sorted(dfB["model"].unique()) if len(dfB) else "NONE")
if len(dfB): print(dfB.groupby("model").size().to_string())

# 3) audio-quality result files (STOI / eSTOI / MCD): copied as they are, top-level keys printed
for rel in [r"stageA_stoi\stoi_results.json", r"quality\mcd_results.json", r"quality\stoiB_results.json"]:
    src = os.path.join(OUT, rel)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(DEST, os.path.basename(src)))
        j = json.load(open(src)); print("copied", rel, "| top-level keys:", list(j)[:6], "...")
    else:
        print("NOT FOUND:", rel)

# 4) final figures (PDF + PNG) from paper_figures
fd = os.path.join(DEST, "figures"); os.makedirs(fd, exist_ok=True)
for ext in ("pdf", "png"):
    for f in glob.glob(os.path.join(OUT, "paper_figures", "*." + ext)) + glob.glob(os.path.join(ROOT, "*." + ext)):
        if os.path.basename(f).startswith("fig_"): shutil.copy(f, fd)
print("figures:", sorted(os.listdir(fd)))

# 5) split definition + a manifest of what exists
man = {s: dict(channels=int(len(np.load(os.path.join(OUT, f"{s}_channels.npy"), allow_pickle=True))),
               frames=int(len(np.load(os.path.join(OUT, f"{s}_labels.npy"), allow_pickle=True)))) for s in [f"sub-{i:02d}" for i in range(1, 11)]}
json.dump(man, open(os.path.join(DEST, "subjects_manifest.json"), "w"), indent=1)

# 6) zip
zp = os.path.join(ROOT, "paper_export.zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for dp, _, fs in os.walk(DEST):
        for f in fs: z.write(os.path.join(dp, f), os.path.relpath(os.path.join(dp, f), DEST))
print("\nDONE ->", zp, round(os.path.getsize(zp) / 1e6, 2), "MB")
