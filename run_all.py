"""
run_all.py -- the whole pipeline in order, from the downloaded dataset to numbers.csv.

    python run_all.py list                  # show the stages and their commands
    python run_all.py <stage> [<stage> ...] # run these stages (in the order given)
    python run_all.py all                   # every stage, in order  (days on a CPU; every step is resumable)
    python run_all.py <stage> --dry-run     # print the commands only

Set before running:   SEEG_DATA_ROOT  (downloaded dataset)   SEEG_OUT_DIR  (folder for features and results)
Every training script is resumable: finished participant-folds are skipped, so an interrupted run can simply be repeated.
Seeds are fixed in the scripts (42, 123, 456). Bit-identical results on a different machine or GPU are not guaranteed.
The 'perparticipant' run of convedsr gives the HG-frame variant (Supplementary S5); 'convedsr1024' trains the main ConvED-SR (1024 Hz input) and
installs it in place of the HG-frame runs (moved to SEEG_OUT_DIR/archive_convedsr_hgframes). Set CONVED_HG_DIR to that folder for the 'numbers' stage.
"""
import os, sys, subprocess

HERE = os.path.dirname(os.path.abspath(__file__)); CODE = os.path.join(HERE, "code")
PY = sys.executable
NEURAL = ["hybrid", "baseline", "neuroincept", "convedsr"]

STAGES = {
    "features": [["extract_continuous_80.py", "hifigan"], ["validate_extraction.py"]],
    "perparticipant": [["stageB_subject.py", "run", m] for m in ("linear", "hybrid", "baseline", "neuroincept", "convedsr")],
    "seeds": [["stageB_extra_seeds_savepreds.py", "run", m, s] for s in ("123", "456") for m in NEURAL],
    "convedsr1024": [["stageB_convedsr_1024_full.py", "run"], ["stageB_convedsr_1024_full.py", "install"]],
    "audio": [["quality_metrics.py", "mcd"], ["quality_metrics.py", "stoiB", "folds=0"], ["cell_stageA_constant_control.py"]],
    "inword": [["inword_stageB.py"], ["speech_per_word.py"]],
    "supplement": [["shift_scan_all.py"], ["supp_template_sensitivity.py"]],
    "numbers": [["analysis/make_numbers_v12.py", os.environ.get("SEEG_OUT_DIR", ""), os.path.join(HERE, "results", "check")]],
}
ENV_EXTRA = {}                                            # learning rates are fixed a priori in code/stageB_subject.py (FIXED_LR)


def main(argv):
    dry = "--dry-run" in argv; argv = [a for a in argv if a != "--dry-run"]
    if not argv or argv == ["list"]:
        for k, v in STAGES.items():
            print(f"[{k}]"); [print("   python", " ".join(c)) for c in v]
        print("\nthe 'numbers' stage writes results/check_macros.tex / .csv; they reproduce results/macros.tex exactly on the released files.\n"
              "results/numbers.tex (the file used by the paper) = the static macros of results/make_numbers.py + results/macros.tex")
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
