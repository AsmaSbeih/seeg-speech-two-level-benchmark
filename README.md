# sEEG-to-speech decoding beyond task timing: a two-level benchmark

Code and result files of the paper *sEEG-to-Speech Decoding Beyond Task Timing: A Two-Level Benchmark* (A. Sbaih, J. García-Gutiérrez, D. Benavides; University of Seville; submitted to IEEE Sensors Journal).

Five decoders (a linear decoder following Verwoert et al., our re-implementations of NeuroIncept and ConvED-SR, a CNN-BiLSTM, and the MR-CNN-BiLSTM, i.e. the CNN-BiLSTM with multi-receptive residual blocks) are compared on the public SingleWordProductionDutch-iBIDS dataset against a timing template without neural input, at two levels: the whole spectrogram (level 1, metrics per test block) and the speech frames of each word (level 2, word-level in-speech PCC). One model is trained per participant, fold (10 contiguous blocks) and seed (42, 123, 456); every number of the paper is the mean over the three seeds. In the code, MR-CNN-BiLSTM is named `hybrid` and CNN-BiLSTM `baseline`.

## What changed in version 1.1.0
* Learning rates are fixed a priori, with no selection on any data (`FIXED_LR` in `code/stageB_subject.py`): 1e-4 for the CNN-BiLSTM, the MR-CNN-BiLSTM and NeuroIncept, 5e-4 for ConvED-SR (the rates of the original code). The learning-rate selection, the NeuroIncept override and the partial guard-gap control of version 1.0 are removed.
* ConvED-SR uses its original input resolution, the HG envelope at 1024 Hz (`code/stageB_convedsr_1024_full.py`), for all participants, folds and seeds; the HG-frame variant is kept for Supplementary Section S5.
* All statistics, intervals and macros of the paper are computed by one script, `code/analysis/make_numbers_v12.py`, from the participant values averaged over the three seeds.
* Audio metrics: STOI only.

## Data (not included)
Download the dataset from OSF: https://doi.org/10.17605/OSF.IO/NRGX6 (Verwoert et al., Sci. Data 2022, https://doi.org/10.1038/s41597-022-01542-9). The raw recordings and the extracted feature arrays (`*_feat.npy`, `*_spec.npy`) are not in this repository.

## Vocoder
The audio metrics use the pre-trained SpeechBrain HiFi-GAN checkpoint `speechbrain/tts-hifigan-ljspeech` (22.05 kHz, 80 mel bins, n_fft 1024, hop 256, fmax 8000), downloaded on first use and then loaded from `SEEG_OUT_DIR/hifigan_ckpt`.

## Layout
```
code/            all Python scripts (flat: they import each other, keep them in one folder)
code/analysis/   make_numbers_v12.py: every statistic, bootstrap interval and macro of the paper and of the supplement
notebooks/       the notebooks used to run and inspect the experiments
results/         numbers.tex (the macro file used by the paper), macros.tex/.csv (output of make_numbers_v12.py),
                 stageB_long_allseeds.csv (level 1, every participant-fold-seed), inword_stageB_long.csv (level 2, every word),
                 supp_template_sensitivity.csv, shift_scan_all.csv, stoiB_*.json, mcd_results.json,
                 runs/ (the result file of every run: stageB/ seed 42, stageB_seeds_preds/ seeds 123 and 456,
                 stageB_1024full/ ConvED-SR at 1024 Hz, convedsr_hgframes/ ConvED-SR with HG frames)
supplement/      the Supplementary Material (LaTeX source and PDF)
```

## Environment
Python 3, with `pip install -r requirements.txt` (exact versions, Python 3.12). Set two environment variables before running the scripts:
```
SEEG_DATA_ROOT   folder with the downloaded dataset
SEEG_OUT_DIR     folder for extracted features and results
```
No path is hard-coded. Seeds are fixed in the scripts (42, 123, 456). Training was run on CPU and is not bit-reproducible; the three seeds describe the run-to-run variation.

## One command per stage
`python run_all.py list` shows the whole pipeline in order; `python run_all.py <stage>` runs a stage (`features`, `perparticipant`, `seeds`, `convedsr1024`, `audio`, `inword`, `supplement`, `numbers`), `python run_all.py all` runs everything. All training steps are resumable: finished participant-folds are skipped.

## Reproducing the numbers without retraining
```
CONVED_HG_DIR=results/runs/convedsr_hgframes  python code/analysis/make_numbers_v12.py results check
```
writes `check_macros.tex` and `check_macros.csv`, which reproduce `results/macros.tex` exactly. `results/numbers.tex` is the static part (dataset facts, parameter counts, published values; `results/make_numbers.py`) followed by these macros.

## Notes
* Fold numbering: the code is 0-indexed (fold k tests block k); the paper numbers blocks and folds from 1.
* STOI and the constant-spectrum control use fold 1 of each participant (seed 42); MCD uses all folds (seed 42).
* NeuroIncept and ConvED-SR are our PyTorch re-implementations based on the authors' public code (github owaismujtaba/NeuroInceptDecoder; github yuanyu1016/ConvED-SR); deviations are listed in Supplementary Section S6. Their published numbers are not reproduced.
* Files named `stageA_*` contain shared functions and model definitions used by the per-participant scripts.

## Citation
See `CITATION.cff`. Archived on Zenodo: https://doi.org/10.5281/zenodo.24467745 (all versions).
