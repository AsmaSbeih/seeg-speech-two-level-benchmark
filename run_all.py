"""
run_all.py -- the whole pipeline in order, from the downloaded dataset to numbers.csv.

    python run_all.py list                  # show the stages and their commands
    python run_all.py <stage> [<stage> ...] # run these stages (in the order given)
    python run_all.py all                   # every stage, in order  (days on a CPU; every step is resumable)
    python run_all.py <stage> --dry-run     # print the commands only

Set before running:   SEEG_DATA_ROOT  (downloaded dataset)   SEEG_OUT_DIR  (folder for features and results)
Every training script is resumable: finished participant-folds are skipped, so an interrupted run can simply be repeated.
Seeds are fixed in the scripts (42, 123, 456). Bit-identical results on a different machine or GPU are not guaranteed.
Stages 'supplement' use the arguments printed in each script's docstring; check them against Supplementary S2/S3/S7 before relying on them.
"""
import os, sys, subprocess

HERE = os.path.dirname(os.path.abspath(__file__)); CODE = os.path.join(HERE, "code")
PY = sys.executable
NEURAL = ["hybrid", "baseline", "neuroincept", "convedsr"]

STAGES = {
    "features": [["extract_continuous_80.py", "hifigan"], ["validate_extraction.py"]],
    "pooled": [["stageA_pooled.py", "run", m] for m in ("baseline", "hybrid", "linear", "baseline_null", "hybrid_null")] +
              [["stageA_extra.py", "run", m] for m in ("neuroincept", "convedsr", "neuroincept_null", "convedsr_null")],
    "perparticipant": [["stageB_verwoert_linear.py"]] +
              [["stageB_subject.py", "run", m] for m in ("linear", "hybrid", "baseline", "neuroincept", "convedsr")],
    "seeds": [["stageB_extra_seeds_savepreds.py", "run", m, s] for s in ("123", "456") for m in NEURAL],
    "audio": [["stageA_stoi.py"], ["quality_metrics.py", "mcd"], ["quality_metrics.py", "stoiB", "folds=0"],
              ["cell_stageA_constant_control.py"]],
    "inword": [["inword_stageB.py"], ["speech_per_word.py"]],
    "supplement": [["shift_scan.py"], ["shift_scan_all.py"], ["supp_template_sensitivity.py"],
                   ["stageB_purge.py", "run", "hybrid", "0"], ["stageB_purge.py", "run", "hybrid", "86", "256"],
                   ["stageB_lr.py", "run", "baseline", "3e-4", "1e-3", "--folds", "0", "5", "--subs", "sub-02", "sub-06", "sub-10"],
                   ["stageB_lr.py", "run", "hybrid", "3e-4", "1e-3", "--folds", "0", "5", "--subs", "sub-02", "sub-06", "sub-10"],
                   ["stageB_lr.py", "run", "neuroincept", "1e-3"],
                   ["stageB_convedsr_1024.py", "run"], ["supp_bootstrap_channels.py", os.path.join(HERE, "results")]],
    "numbers": [["export_for_paper.py"], ["cell_verify_numbers.py"]],
    "macros": [["analysis/make_v5_macros.py", os.path.join(HERE, "results", "stageB_long.csv"), os.path.join(HERE, "results", "inword_stageB_long.csv"),
                os.path.join(HERE, "results", "numbers_v5_macros.csv"), os.path.join(HERE, "results", "numbers_v5_macros.tex")]],
}
ENV_EXTRA = {"STAGEB_LR_NEUROINCEPT": "1e-4"}          # the declared NeuroIncept learning-rate override (per-participant models)


def main(argv):
    dry = "--dry-run" in argv; argv = [a for a in argv if a != "--dry-run"]
    if not argv or argv == ["list"]:
        for k, v in STAGES.items():
            print(f"[{k}]"); [print("   python", " ".join(c)) for c in v]
        print("\nafter the 'numbers' stage:  cd results && python make_numbers.py   (numbers.csv / numbers.tex)\n"
              "the 'macros' stage adds the remaining macros of the paper (numbers_v5_macros.csv / .tex); together they define every \\N... macro")
        return
    names = list(STAGES) if argv == ["all"] else argv
    for n in names:
        if n not in STAGES: sys.exit(f"unknown stage '{n}'; try: python run_all.py list")
    for n in names:
        for c in STAGES[n]:
            print(f"\n=== [{n}] python {' '.join(c)}", flush=True)
            if dry: continue
            for v in ("SEEG_DATA_ROOT", "SEEG_OUT_DIR"):
                if v not in os.environ: sys.exit(f"set the environment variable {v} first")
            env = dict(os.environ, **ENV_EXTRA)
            r = subprocess.run([PY, os.path.join(CODE, c[0])] + c[1:], cwd=CODE, env=env)
            if r.returncode: sys.exit(f"stopped: {c[0]} returned {r.returncode}")


if __name__ == "__main__":
    main(sys.argv[1:])
