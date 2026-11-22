# Phase 11 controlled comparison: does Logit Adjustment help cross-attention?

Date: 2026-11-22 · Code commit: `de71102` (clean tree at launch) · Machine: local server, 1× RTX 5070
Artifacts: `data/server/outputs/p11_controlled/` (symlink to `/data/Phat/vimmsd_outputs/p11_controlled/`)

This is a controlled comparison on a benchmark that Phase 11 already used. The private test was looked at
in Phase 11, and it is looked at again here. These numbers measure the effect of the loss when everything
else is held fixed. They are not a fresh held-out estimate.

## 1. Audit of the original Phase 11 runs

| Item | Finding |
|---|---|
| Word report `Phase 11_ Cross-Attention + Imbalance Techniques.docx` | **Not found** in the repo or anywhere on this machine. Its numbers (CE 0.373 ± 0.011, LA 0.419 ± 0.011) are known only from the task statement. |
| Original Kaggle or server artifacts (resolved configs, logs, checkpoints, metrics, predictions) | **None found.** There are no `fusion_cross_attn_frozen*` output dirs and no Phase 11 rows in `reports/results/results.md`. Notebook 11 has no saved outputs. |
| Commit used originally | Unverifiable. Notebook 11 clones `REF = "main"`, not a pinned hash. |
| Split and training seeds | Notebook 11 uses `SEEDS = [42, 1, 2]` and `SPLIT_SEED = 42`, with the official split (train 10,805, val = public test 1,413, test = private test 1,504). |
| Checkpoint selection | `train.select_metric` keeps its default, `macro_f1` (4-class val macro-F1, no excluded labels), with early stopping on the same score. |
| Effective hyperparameters | Notebook 11 passes only `seed`, `data.split_seed` and `name` as overrides, so each arm used its own YAML defaults (table below). |

What the configs resolve to when run the way Notebook 11 runs them:

| Setting | CA-base `fusion_cross_attn_frozen` | CA-LA `fusion_cross_attn_frozen_logit_adjustment` |
|---|---|---|
| `train.lr_head` | 1e-3 | 5e-4 |
| `train.weight_decay` | 0.01 | 0.05 |
| `model.dropout` | 0.2 | 0.3 |
| `train.early_stop_patience` | 3 | 5 |
| `loss.name` | ce | logit_adjustment (τ = 1.0) |

The notebook's own markdown lists the four regularisation changes as applying "so với baseline" (relative
to the baseline), but the code never applies them to CA-base. The Word report says both arms used the same
training settings. **That claim is contradicted by the repository code.** If the original runs used
Notebook 11 as committed, they differed in five settings, not one. Since no original artifact survives,
none of the six original runs could be reused, and all six were retrained here.

## 2. Matched setup

Both arms: frozen PhoBERT-base + frozen CLIP ViT-B/32, bidirectional cross-attention (1 layer, 8 heads,
hidden 512). Only the cross-attention blocks and the flat classifier head are trained.

The common overrides on top of each arm's YAML are listed below (stored verbatim in `resolved_configs/*.overrides`):

```
train.lr_head=5e-4 train.weight_decay=0.05 model.dropout=0.3 train.early_stop_patience=5
loss.class_weight=null loss.mixup_alpha=0.0 data.image_augment=false data.augment.file=null
data.sampler=random eval.prior_correction=false train.select_metric=macro_f1 train.select_exclude_labels=[]
train.num_workers=0 data.split_seed=42 seed={42,1,2}
CE: loss.name=ce        LA: loss.name=logit_adjustment loss.tau=1.0
```

The shared settings are AdamW, a linear schedule with 10% warmup, batch size 16, at most 20 epochs, AMP, a
random sampler and pad image resize. The checkpoint is chosen by best 4-class val macro-F1 (Notebook 11's
predetermined criterion). After that, the private test is scored once with raw logits (`scripts/evaluate.py`).
`train.num_workers=0` is needed on this machine because `/dev/shm` is 64 MB, and it does not change results.

**Config diff check (before training):** within each seed the resolved configs differ only in `name`,
`config_path` and `loss.name`. `loss.tau` is present in both, but CE ignores it.

**Implemented LA loss** (`src/vimmsd/training/losses.py:40`): `CE(logits + τ·log(prior), targets)` with
τ = 1.0. The prior is the training-class frequencies 6062 / 77 / 442 / 4224 (not / text / image / multi).
Inference uses the raw logits with no shift.

## 3. Results (private test, n = 1,504; class counts 458 / 7 / 40 / 999)

### Per seed

| Arm | Seed | Selected epoch / run | Val macro-F1 | Macro-F1 (4) | Macro-F1 (3) | Accuracy | Weighted F1 |
|---|---|---|---|---|---|---|---|
| CE | 42 | 13 / 18 | 0.4026 | 0.3748 | 0.4997 | 0.6882 | 0.7014 |
| CE | 1 | 3 / 8 | 0.3724 | 0.3784 | 0.5045 | 0.7812 | 0.7663 |
| CE | 2 | 14 / 19 | 0.4004 | 0.4167 | 0.5556 | 0.7287 | 0.7305 |
| LA | 42 | 4 / 9 | 0.3908 | 0.4141 | 0.5521 | 0.7487 | 0.7578 |
| LA | 1 | 6 / 11 | 0.3815 | 0.4096 | 0.5462 | 0.7507 | 0.7620 |
| LA | 2 | 2 / 7 | 0.4042 | 0.3890 | 0.5187 | 0.7653 | 0.7551 |

Macro-F1 (3) is the mean F1 of `not`, `image` and `multi`.

### Mean ± SD over 3 seeds

| Metric | CE | LA | LA − CE |
|---|---|---|---|
| Macro-F1 (4-class) | 0.390 ± 0.023 | 0.404 ± 0.013 | +0.014 |
| Macro-F1 (3-class) | 0.520 ± 0.031 | 0.539 ± 0.018 | +0.019 |
| Accuracy | 0.733 ± 0.047 | 0.755 ± 0.009 | +0.022 |
| Weighted F1 | 0.733 ± 0.033 | 0.758 ± 0.004 | +0.026 |
| F1 not / multi | 0.641 / 0.805 | 0.654 / 0.837 | +0.013 / +0.032 |
| Recall not / multi | 0.697 / 0.779 | 0.617 / 0.846 | −0.080 / +0.067 |

### Rare classes

| Arm | Seed | text P / R / F1 | text correct (of 7) | image P / R / F1 | image correct (of 40) |
|---|---|---|---|---|---|
| CE | 42 | 0 / 0 / 0 | 0 | 0.085 / 0.150 / 0.108 | 6 |
| CE | 1 | 0 / 0 / 0 | 0 | 0 / 0 / 0 | 0 |
| CE | 2 | 0 / 0 / 0 | 0 | 0.286 / 0.200 / 0.235 | 8 |
| LA | 42 | 0 / 0 / 0 | 0 | 0.120 / 0.275 / 0.167 | 11 |
| LA | 1 | 0 / 0 / 0 | 0 | 0.101 / 0.200 / 0.135 | 8 |
| LA | 2 | 0 / 0 / 0 | 0 | 0.081 / 0.075 / 0.078 | 3 |
| **CE mean** | | 0 / 0 / 0 | 0 | 0.123 / 0.117 / 0.115 ± 0.118 | 4.7 |
| **LA mean** | | 0 / 0 / 0 | 0 | 0.101 / 0.183 / 0.126 ± 0.045 | 7.3 |

No run in either arm predicts a single `text-sarcasm` test sample correctly.

### Paired difference in 4-class macro-F1

| Seed | LA − CE |
|---|---|
| 42 | +0.039 |
| 1 | +0.031 |
| 2 | −0.028 |
| **Mean** | **+0.014** |

- **Paired bootstrap** (B = 10,000). Each replicate resamples the 1,504 test IDs once, with replacement, and
  applies the same IDs to both arms and all three seeds. 95% CI for the mean difference: **[+0.001, +0.028]**.
  Share of replicates ≤ 0: 1.9%.
- **Group bootstrap**. Groups are joined by exact normalised caption (lower-cased, whitespace-collapsed,
  ≥ 10 characters) or by identical image bytes (MD5). This gives 1,236 groups, of which 92 have more than one
  member, all from shared captions; no byte-identical images were found. 95% CI: **[+0.000, +0.028]**.
  Caption equality is a reasonable proxy but not a verified near-duplicate grouping, and perceptual image
  duplicates were not searched for.
- **Both intervals reflect only test-sample variability, conditional on these three trained seeds.**
  They do not include training-seed variance, which is large here: the per-seed differences range from
  −0.028 to +0.039 (SD 0.037). One of the three seeds goes the other way.

### Where the two arms disagree (true class, summed over 3 seeds)

| True class | n per seed | LA right, CE wrong | CE right, LA wrong |
|---|---|---|---|
| not-sarcasm | 458 | 54 | 164 |
| text-sarcasm | 7 | 0 | 0 |
| image-sarcasm | 40 | 15 | 7 |
| multi-sarcasm | 999 | 344 | 142 |
| **Total** | | **413** | **313** |

Per-seed counts are in `paired_flips_by_class.csv`. The pattern differs by seed. In seeds 42 and 2, LA wins
about 165 multi samples and loses 57–84 not samples. In seed 1 the direction reverses on multi: LA wins 12
and loses 71.

## 4. Comparison with the Phase 11 Word report

| Original claim | Status |
|---|---|
| CE 0.373 ± 0.011 | Not reproducible or verifiable: no artifacts. Matched CE gives 0.390 ± 0.023. The original CE probably ran with lr 1e-3 / wd 0.01 / dropout 0.2 / patience 3. |
| LA 0.419 ± 0.011 | Not verifiable. Matched LA gives 0.404 ± 0.013, which uses the same settings the LA YAML specifies, so the gap (−0.015) is not explained by config. It may come from a different commit, environment or cache. |
| LA improves macro-F1 by about +0.046 | **Partly supported.** The matched mean gain is +0.014, about a third of the reported gain. Up to roughly two-thirds of the original gap is plausibly due to the hyperparameter confound and seed noise rather than LA. |
| "Both runs used the same training settings" | **Contradicted** by Notebook 11 and the YAMLs as committed. |

## 5. Summary

- **Macro-F1:** with training settings matched, LA raises 4-class private-test macro-F1 by **+0.014** on
  average (0.390 → 0.404) and 3-class macro-F1 by +0.019. Two seeds favour LA and one favours CE. The
  test-resampling CI just excludes zero, but given the seed spread and only three seeds, the effect is small
  and not robust.
- **Accuracy trade-off:** none on average. Accuracy rises by +0.022 and weighted F1 by +0.026, and LA is
  much more stable across seeds (accuracy SD 0.009 vs 0.047). LA shifts predictions from `not` towards
  `multi`: not recall −0.080, multi recall +0.067.
- **Rare classes:** `text-sarcasm` stays at 0 correct in all six runs, so LA does not rescue it. On
  `image-sarcasm`, LA gets 7.3 correct on average against 4.7 for CE (recall 0.18 vs 0.12), but precision
  stays around 0.10 and F1 moves only from 0.115 to 0.126, which is within seed noise.
- LA selects its checkpoint much earlier (epoch 2–6 against 3–14 for CE). Validation macro-F1 at the
  selected checkpoint is the same for both arms (0.392).

## 6. Artifacts

| Run | Resolved config | Overrides | Predictions (with `id`) | Log |
|---|---|---|---|---|
| `ca_ce_s42` | `resolved_configs/ca_ce_s42.json`, `ca_ce_s42/config.json` | `resolved_configs/ca_ce_s42.overrides` | `ca_ce_s42/test_predictions.csv`, `ca_ce_s42/val_predictions.csv` | `ca_ce_s42/train_eval.log` |
| `ca_ce_s1` | `resolved_configs/ca_ce_s1.json`, `ca_ce_s1/config.json` | `…/ca_ce_s1.overrides` | `ca_ce_s1/test_predictions.csv` | `ca_ce_s1/train_eval.log` |
| `ca_ce_s2` | `resolved_configs/ca_ce_s2.json`, `ca_ce_s2/config.json` | `…/ca_ce_s2.overrides` | `ca_ce_s2/test_predictions.csv` | `ca_ce_s2/train_eval.log` |
| `ca_la_s42` | `resolved_configs/ca_la_s42.json`, `ca_la_s42/config.json` | `…/ca_la_s42.overrides` | `ca_la_s42/test_predictions.csv` | `ca_la_s42/train_eval.log` |
| `ca_la_s1` | `resolved_configs/ca_la_s1.json`, `ca_la_s1/config.json` | `…/ca_la_s1.overrides` | `ca_la_s1/test_predictions.csv` | `ca_la_s1/train_eval.log` |
| `ca_la_s2` | `resolved_configs/ca_la_s2.json`, `ca_la_s2/config.json` | `…/ca_la_s2.overrides` | `ca_la_s2/test_predictions.csv` | `ca_la_s2/train_eval.log` |

All paths are relative to `data/server/outputs/p11_controlled/`. Each run dir also contains `best.pt`,
`best_val_metrics.json`, `history.json`, `test_metrics.json` and confusion matrices. The commit is stored in
each `config.json` and in `commit.txt`.

- Machine-readable per-seed metrics: `per_seed_metrics.csv`
- Paired statistics: `paired_analysis.json`; disagreements by class: `paired_flips_by_class.csv`
- Reproduce: `run_lane.sh {ce|la}` (training) and `python data/server/outputs/p11_controlled/analyze.py` (analysis, run from the repo root)

## 7. Follow-up: caption-only + LA vs cross-attention + LA (full private test)

Caption-only runs: `data/server/outputs/p11_caption_la_s{42,1,2}` (plan `configs/experiments/phase11_image_contribution.yaml`,
commit `f8f25ab`). They resolve to the same config as `ca_la_s*` except `model.modality=text`: the same LA loss
(τ = 1.0), lr, weight decay, dropout, patience, sampler and selection metric. Predictions join on all 1,504 IDs
with identical labels. Analysis: `analyze_caption_vs_xattn.py` → `caption_vs_xattn_per_seed.csv`,
`caption_vs_xattn_stats.json`, `caption_vs_xattn_flips_by_class.csv`.

| Mean ± SD (3 seeds) | Caption-only + LA | Cross-attn + LA | Cross-attn + CE |
|---|---|---|---|
| Macro-F1 (4) | 0.390 ± 0.006 | 0.404 ± 0.013 | 0.390 ± 0.023 |
| Macro-F1 (3) | 0.519 ± 0.008 | 0.539 ± 0.018 | 0.520 ± 0.031 |
| Accuracy | 0.722 ± 0.033 | 0.755 ± 0.009 | 0.733 ± 0.047 |
| Weighted F1 | 0.736 ± 0.018 | 0.758 ± 0.004 | 0.733 ± 0.033 |
| F1 not / multi | 0.658 / 0.803 | 0.654 / 0.837 | 0.641 / 0.805 |
| image P / R / F1 | 0.081 / 0.150 / 0.097 | 0.101 / 0.183 / 0.126 | 0.123 / 0.117 / 0.115 |
| image correct (of 40) / predicted | 6.0 / 83 | 7.3 / 69 | 4.7 / 33 |
| text correct (of 7) | 0 | 0 | 0 |

Paired differences use the same bootstrap as above (B = 10,000, the same resampled IDs for all models and seeds,
conditional on these seeds):

| Comparison | Per seed (42 / 1 / 2) | Mean | 95% CI |
|---|---|---|---|
| Cross-attn LA − caption LA, macro-F1 (4) | +0.030 / +0.014 / −0.000 | +0.015 | [−0.004, +0.032] |
| Cross-attn LA − caption LA, accuracy | +0.047 / −0.009 / +0.061 | +0.033 | [+0.019, +0.047] |
| Cross-attn CE − caption LA, macro-F1 (4) | −0.009 / −0.017 / +0.027 | +0.000 | [−0.020, +0.020] |

Disagreements between cross-attn LA and caption LA, summed over seeds: cross-attn wins 356 and loses 135 on `multi`,
wins 68 and loses 145 on `not`, wins 15 and loses 11 on `image`, and neither ever gets a `text` sample right.

Reading: adding the image through cross-attention, under LA, gives a reliable accuracy gain (+0.033) but a
macro-F1 gain (+0.015) whose interval includes zero. The gain comes almost entirely from `multi-sarcasm`. On
`image-sarcasm`, the class where the image should matter most, cross-attention gets 1.3 more of 40 correct per
seed, which is within noise. Caption-only predicts `image-sarcasm` *more* often than cross-attention (83 vs 69
per seed) and gets 6 of 40 right without ever seeing an image, so captions carry cues for that label.
Cross-attention without LA matches caption-only + LA on macro-F1 exactly.

## 8. Still unavailable

- The Word report itself, and every artifact from the original Kaggle runs. The original numbers and the
  settings those runs actually used remain unverified.
- Training-seed uncertainty beyond three seeds. A between-seed interval would need more seeds.
