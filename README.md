# sEEG-to-speech decoding beyond task timing: a two-level benchmark

Code and result files for the manuscript **"sEEG-to-Speech Decoding Beyond Task Timing: A Two-Level Benchmark"** (A. Sbaih, J. García-Gutiérrez, D. Benavides; University of Seville).

Five decoders (a linear decoder following Verwoert et al., our re-implementations of NeuroIncept and ConvED-SR, a CNN-BiLSTM, and the MR-CNN-BiLSTM, i.e. the CNN-BiLSTM with multi-receptive residual blocks) are compared on the public SingleWordProductionDutch-iBIDS dataset against a timing template without neural input, at two levels: the whole spectrogram (level 1) and the speech frames of each word (level 2). The paper reports the per-participant models; the pooled model (one model for all participants) is kept in this repository as additional code and results not used in the paper. In the code, MR-CNN-BiLSTM is named `hybrid` and CNN-BiLSTM `baseline`.

* **Pooled model** (files named `stageA_*`; not reported in the paper; it also provides the learning-rate selection, validation blocks 1 and 6): one model for all 10 participants (test blocks 3 and 8; seeds 42, 123, 456).
* **Per-participant models** (files named `stageB_*`): one model per participant, 10 contiguous folds (seed 42 in the main text; seeds 123 and 456 in the supplement).

## Data (not included)
Download the dataset from OSF: https://doi.org/10.17605/OSF.IO/NRGX6 (Verwoert et al., Sci. Data 2022, https://doi.org/10.1038/s41597-022-01542-9). The raw recordings, the extracted feature arrays (`*_feat.npy`, `*_spec.npy`) and the saved predictions (`*_preds.npz`) are not in this repository.

## Vocoder
The audio metrics use the pre-trained SpeechBrain HiFi-GAN checkpoint `speechbrain/tts-hifigan-ljspeech` (22.05 kHz, 80 mel bins, n_fft 1024, hop 256, fmax 8000), downloaded on first use.

## Layout
```
code/        all Python scripts (flat: they import each other, keep them in one folder); code/analysis/ = statistics of the paper (with expected_outputs/)
notebooks/   the notebooks used to run and inspect the experiments (audio players removed from the saved outputs)
results/     result files behind every table and figure of the paper and of the supplement, and make_numbers.py
supplement/  fig_model_structures.pdf (Fig. S1); the final Supplementary Material PDF
```

## Environment
Python 3, with `pip install -r requirements.txt`. Set two environment variables before running the scripts:
```
SEEG_DATA_ROOT   folder with the downloaded dataset
SEEG_OUT_DIR     folder for extracted features and results
```
No path is hard-coded: every script reads `SEEG_OUT_DIR` (the features folder). `export_for_paper.py` writes `paper_export/` next to it (or to `SEEG_ROOT` if set).
Exact package versions are in `requirements.txt` (Python 3.12). Seeds are fixed in the scripts (42, 123, 456); bit-identical results on another machine or GPU are not guaranteed.

## One command per stage
`python run_all.py list` shows the whole pipeline in order; `python run_all.py <stage>` runs a stage (`features`, `pooled`, `perparticipant`, `seeds`, `audio`, `inword`, `supplement`, `numbers`, `macros`), `python run_all.py all` runs everything. All training steps are resumable. The predictions of every per-participant run (all seeds) are released as `predictions_perparticipant.zip` on Zenodo, so the word-level analysis can be recomputed without retraining (unzip it into `SEEG_OUT_DIR`).

## Order of execution (what each stage does; scripts are in `code/`)
1. **Features** (HG envelope 70-170 Hz, 80-bin log-mel on the HiFi-GAN grid):
   `python extract_continuous_80.py hifigan` (self-test: `python extract_continuous_80.py selftest`); checks: `python validate_extraction.py`.
2. **Pooled model** (`stageA_*`; learning-rate search and 3 seeds):
   `python stageA_pooled.py run baseline` / `hybrid` / `linear`, and the noise-input controls `baseline_null` / `hybrid_null`;
   `python stageA_extra.py run neuroincept` / `convedsr` / `neuroincept_null` / `convedsr_null`.
3. **Per-participant models** (`stageB_*`, 10 folds): `python stageB_verwoert_linear.py`, then `python stageB_subject.py run <linear|hybrid|baseline|neuroincept|convedsr>`.
   NeuroIncept uses the declared learning-rate override: set `STAGEB_LR_NEUROINCEPT=1e-4`.
4. **Template and residual PCC**: computed inside the scripts above with `template_floor_continuous.py` (time since the last word onset, up to 300 frames).
5. **Audio metrics**: `python stageA_stoi.py` (pooled-model STOI/eSTOI), `python quality_metrics.py mcd`, `python quality_metrics.py stoiB folds=0` (per-participant STOI/eSTOI, fold 1 of the paper), and the constant-spectrum control `cell_stageA_constant_control.py`.
6. **Shift check** (preliminary, three participants): `python shift_scan.py` writes `shift_scan.csv`.
7. **Word-level (level 2)**: `python inword_stageB.py` writes `inword_stageB_long.csv` (one row per word, system and seed; needs the saved `*_preds.npz`; seeds 123/456 come from `stageB_extra_seeds_savepreds.py`). `python speech_per_word.py` gives the speech duration per word trial (Results, Table S12).
8. **Supplement**: `stageB_extra_seeds_savepreds.py` (S1), `stageB_purge.py` and `stageB_verwoert_linear.py` (S2), `stageB_lr.py` (S3), `supp_template_sensitivity.py` and `shift_scan_all.py` (S4), `supp_bootstrap_channels.py` (S5, channel counts), `stageB_convedsr_1024.py` (S7), `quality_metrics.py` (S10, S14).
9. **Figures** (each script prints its usage): `fig_overview.py` (Fig. 1), `fig_protocol.py` (protocol figure), `fig_eval_matrix.py <results/stageB_long.csv> <out.pdf>` (Supplementary Fig. S2), `fig_recon_pair.py` (supplementary reconstruction example, per-participant model), `fig_l2_participants.py <results/inword_stageB_long.csv> <out.pdf>` (level-2 figure), `fig_model_structures.py` (architectures, Fig. S1), `make_graphical_abstract_flow.py <results/numbers.csv> <results/inword_stageB_long.csv> <fig_overview.pdf> <out.png>` (graphical abstract).
10. **Numbers of the paper**: from `results/`, run `python make_numbers.py` (writes `numbers.csv`, `numbers.tex`). Then `python code/analysis/make_v5_macros.py results/stageB_long.csv results/inword_stageB_long.csv results/numbers_v5_macros.csv results/numbers_v5_macros.tex` adds the remaining macros (level-1 Holm family and bootstrap, level 2 for every seed, learning-rate fold exclusion). Together the two files define every `\N...` macro of the paper. The other scripts in `code/analysis/` print the same numbers as tables (`bootstrap_l1.py`: Tables S8-S9; `inword_stats.py`: Table III; `inword_seeds.py`: Table S16; `lr_fold_exclusion.py`: S3); `code/analysis/expected_outputs/` holds their output on the released files, which a rerun reproduces exactly.

## Notes
* Fold numbering: the code is 0-indexed (per-participant fold k tests block k); the paper numbers blocks and folds from 1. Pooled-model test blocks (2, 7) in the code are blocks 3 and 8 in the paper.
* Per-participant STOI, eSTOI and the constant-spectrum control use fold 1 of each participant; MCD uses all folds.
* NeuroIncept and ConvED-SR are our PyTorch re-implementations based on the authors' public code (github owaismujtaba/NeuroInceptDecoder; github yuanyu1016/ConvED-SR), adapted to our frames; deviations are listed in the docstring of `code/stageA_extra.py` and in Supplementary Section S8. Their published numbers are not reproduced.

## Citation
See `CITATION.cff`.
