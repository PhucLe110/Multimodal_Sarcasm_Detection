# Empirical study — Phase 4 freeze: finalists and final-benchmark protocol

Written and committed on 13 November 2026 **before** any private-test prediction for these models was produced (plan §4.1, §4.3, §11, §13). Nothing below may be changed after the final evaluation is run; results are reported whatever they show.

## Protocol amendment (B4/B5)

The augmentation treatments B4/B5 (plan §8.1–8.2) were **not run**. On 13 November the project decided to skip them for now; the generated candidates and review sheets remain in `data/processed/empirical/augmentation/` for possible later work. The §8.4 selection was therefore made among B2 and B3 only; neither beat B1 (`reports/empirical_phase2.md` §3), so **B1 is the selected imbalance treatment**. This amendment is disclosed in the final report.

## Finalists (fixed)

All are existing checkpoints, trained on the full official train with seeds 1, 2 and 42 and selected on val macro-F1 excluding `text-sarcasm`; no retraining, recalibration, threshold or prior adjustment is applied.

| ID | runs | role |
|---|---|---|
| T | `empA_T_s{1,2,42}` | caption-only baseline, plain CE |
| C | `empA_C_s{1,2,42}` | main backbone (caption + CLIP, regularized concat), plain CE |
| C + B1 | `empB1_C_s{1,2,42}` | selected candidate (C with square-root inverse class weights) |
| T + B1 | `empE_T_B1_s{1,2,42}` | caption-only with B1; tests the Phase 3 hypothesis that B1, not the image, explains C's advantage |

Other Experiment A models (I, O, D, OD, M) and the Phase 3 tag-free and M + B1 runs are **not** evaluated on private test.

## Benchmark

Private test, 1,504 samples (not 458 · text 7 · image 40 · multi 999), reference labels `vimmsd-private-test-ref.json`, sha256 `b70bc3979b8c…` of the annotation file (`reports/empirical_phase0.md` §1). It is a **fixed final benchmark, not a pristine holdout**: private-test scores were viewed earlier for other models in this project (Phase 4 Kaggle baselines, CoT stage runs, the denoising comparison). None of the four finalists' private-test predictions has been viewed.

## Declared contrasts

One family, Holm-adjusted, four-class macro-F1, paired by seed, paired bootstrap over test samples (2,000 replicates, bootstrap seed 0) plus a group-resampled sensitivity interval (caption/image groups within private test):

1. **C + B1 − C**: does the selected treatment improve the backbone?
2. **C − T**: does the image help under plain CE?
3. **T + B1 − T**: does B1 help the caption-only model?
4. **C + B1 − T + B1**: does the image help once imbalance is handled? (Hypothesis from Phase 3, declared here before testing.)

Also reported, without hypothesis tests: three-class macro-F1 (excluding `text-sarcasm`, 7 test samples), weighted-F1, accuracy, per-class precision/recall/F1 with support, confusion matrices, and the fixed subsets of `scripts/empirical_stats.py subsets --split test`.

## Decision rules (plan §11)

- An **overall improvement** is claimed only if the mean difference is positive, its 95% interval excludes zero after accounting for the family, the direction agrees across seeds where possible, and per-class regressions are disclosed.
- The practical target is **+1 point** of four-class macro-F1; smaller reliable gains may be reported as such.
- If a model improves minority recall while lowering macro-F1, it is reported as a **minority-focused tradeoff**, not an improvement.
- Results that do not meet these rules are reported as **inconclusive** or **no reliable improvement**. The four-class 0.402 of the Phase 4 Kaggle baseline is historical and selected differently; it is not compared directly.

## Execution

```bash
VIMMSD_ENV=server python scripts/run_experiments.py --plan configs/experiments/empirical_final.yaml --gpu-ids 0 --extra-eval test
python scripts/empirical_stats.py contrasts --split test --family final --group-bootstrap \
    --pairs empB1_C:empA_C empA_C:empA_T empE_T_B1:empA_T empB1_C:empE_T_B1
python scripts/empirical_stats.py contrasts --split test --family final_3class --exclude text-sarcasm \
    --pairs empB1_C:empA_C empA_C:empA_T empE_T_B1:empA_T empB1_C:empE_T_B1
python scripts/empirical_stats.py subsets --split test --models empA_T empA_C empB1_C empE_T_B1
```

`empirical_final.yaml` lists exactly the twelve runs above with their training overrides, so `run_experiments.py` only runs `evaluate.py --split test` for them (a dry run must show 12 evaluation-only jobs and no training).
