# Understanding and Improving Vietnamese Multimodal Sarcasm Detection: Modality Contributions, Class Imbalance, and Surface Cues

Final report of the study planned in `reports/plan/empirical_analysis_plan.md`, run 12–13 November 2026 on the project GPU server (RTX 5070). Phase documents: `reports/results/empirical_phase0.md` (audit, protocol, splits), `empirical_phase1.md` (infrastructure), `empirical_phase2.md` (modality matrix, input diagnostics, imbalance screen), `empirical_phase3.md` (confirmation, cues, transfer), `empirical_phase4_freeze.md` (finalists and contrasts, committed before the final benchmark). 60 training runs in total, all with frozen PhoBERT and CLIP encoders.

## 1. Summary

- **No intervention gives a reliable improvement on the final benchmark.** The selected model, C + B1 (caption + CLIP, square-root inverse class weights), is the best of the four finalists on private test (macro-F1 0.380 vs 0.369 for C, 0.366 for T + B1, 0.359 for T) and improves on C for every seed (+0.011, at the +1-point practical target), but its 95% interval includes zero after Holm adjustment. By the frozen decision rules this is **inconclusive**, not an established improvement.
- **The ordering is consistent across val and private test** (C + B1 > C > T), but every predeclared contrast on private test has an interval that touches or includes zero.
- **The image contributes modestly**: C − T is +0.019 on val and +0.010 on private test. The Phase 3 hypothesis that class weighting alone explains this was **not supported**: on private test C + B1 − T + B1 is +0.015, positive for every seed, though still not significant.
- **Image-derived text (OCR, VLM descriptions) adds nothing detectable** over the caption under matched settings.
- **Class weighting (B1)** raised `image-sarcasm` recall on val, but on private test `image-sarcasm` F1 stays near zero for every model (best 0.057); C + B1's private-test gain comes from `multi-sarcasm` and `not-sarcasm`. **`text-sarcasm` is never predicted** by any model on any split.
- **Source tags are a real cue** (removing them flips about a quarter of predictions on tagged captions) but cost little overall; retraining without them does not help, and OCR still carries the source for 17–30% of tagged samples.

## 2. Data, labels and protocol

Official splits: train 10,805 (not 6,062 · text 77 · image 442 · multi 4,224), val = public test 1,413 (442 · 1 · 28 · 942), private test 1,504 (458 · 7 · 40 · 999). Checksums and cross-split overlap in `empirical_phase0.md` §1: 4.2% of public and 5.5% of private test samples share a caption or image with train; none shares both.

Label uncertainty: the official guideline defines only sarcastic vs not, and the dataset's labels contradict 3 of its 5 findable examples (`reports/guides/annotation_guideline*.md`). The four-way boundaries, especially `not-sarcasm` vs `multi-sarcasm`, are therefore uncertain; poor minority performance should not be attributed to imbalance alone.

Protocol (frozen in `empirical_phase0.md` §3): frozen encoders, flat MLP head, same budget for every model (max 20 epochs, patience 3), seeds 1, 2, 42, checkpoint selection on val macro-F1 excluding `text-sarcasm`, no private-test scoring during development. Imbalance treatments were screened on a 3-fold split inside train, grouped by shared caption or image md5, with a fixed budget of 451 optimizer steps per epoch.

**Test exposure.** Private test is a fixed final benchmark, not a pristine holdout: it was viewed earlier for other models in this project. The four finalists were frozen in a commit (`a712067`) before any of their private-test predictions existed.

**Amendment.** Augmentation treatments B4/B5 (reviewed back-translation) were not run; candidates and review sheets were produced (966 candidates to review) but the project skipped the two-person review. Selection was made among B1–B3.

## 3. Modality contributions (Experiment A, val)

| model | inputs | macro-F1 | 3-class | F1 image |
|---|---|---|---|---|
| T | caption | 0.381 ± 0.007 | 0.507 | 0.070 |
| I | pixels | 0.346 ± 0.008 | 0.461 | 0.000 |
| C | caption + CLIP | 0.399 ± 0.008 | 0.532 | 0.146 |
| O | caption + OCR | 0.383 ± 0.014 | 0.511 | 0.035 |
| D | caption + description | 0.373 ± 0.002 | 0.497 | 0.000 |
| OD | caption + OCR + description | 0.387 ± 0.013 | 0.515 | 0.113 |
| M | all | 0.391 ± 0.011 | 0.521 | 0.077 |

Predeclared contrasts: C − T +0.019 (all seeds positive, 95% CI [−0.006, +0.042], Holm p 0.79); O − T +0.003, D − T −0.008, OD − O +0.003, OD − D +0.014, M − OD +0.005, all with intervals around zero. Details: `empirical_phase2.md` §1.

## 4. Input dependence (Experiment B, val, no retraining)

Replacing the image with another sample's flips 20% of C's predictions (most for `image-sarcasm`, 36%); C + B1 behaves the same (22%). M spreads reliance over three streams: replacing pixels, OCR or description alone flips 9%, 10% and 7%; all three together 18%. Sensitivity varies two- to fourfold across seeds, so single-seed diagnostics would mislead. These are disruption measures under arbitrary substitutions, not accuracy on relabeled data.

## 5. Imbalance (Experiment C)

Internal pooled out-of-fold macro-F1 on C (seed 42): B0 CE 0.381, **B1** square-root class weights 0.389, B2 square-root sampling 0.385, B3 focal 0.371; none differs reliably from B0. B1 was selected by the predeclared rule. On full train it gave +0.017 on val for C (inconclusive) and +0.031 for T (the only val contrast surviving Holm), +0.009 for M. On private test the T + B1 gain shrinks to +0.007: the val estimate was optimistic.

## 6. Source and surface cues (Experiment D)

Removing source tags at inference flips 19–26% of predictions on the 193 tagged val captions for every model, costing about −0.01 macro-F1 overall; removing emoji and repeated punctuation costs −0.02 to −0.035. Retraining T and C without tags changes val macro-F1 by −0.008 and −0.014 (not distinguishable from zero): the tag is a useful in-distribution cue, and removing it does not improve official-split scores. Removing caption tags does not make a model source-free: OCR contains a source marker for 30% (train) and 17% (val) of tagged samples.

## 7. Final benchmark (private test, 1,504 samples)

Finalists frozen in `empirical_phase4_freeze.md`; mean ± SD over seeds 1, 2, 42.

| model | macro-F1 (4 cl.) | 3-class | weighted-F1 | accuracy | F1 not | F1 image (40) | F1 multi |
|---|---|---|---|---|---|---|---|
| T | 0.359 ± 0.021 | 0.478 | 0.721 | 0.726 | 0.644 | 0.000 | 0.791 |
| T + B1 | 0.366 ± 0.014 | 0.487 | 0.729 | 0.732 | 0.642 | 0.017 | 0.803 |
| C | 0.369 ± 0.011 | 0.492 | 0.733 | 0.738 | 0.641 | 0.027 | 0.809 |
| **C + B1** | **0.380 ± 0.007** | **0.507** | **0.741** | **0.746** | **0.647** | **0.057** | **0.817** |

`text-sarcasm` F1 is 0 for all (7 samples).

Declared family (four-class macro-F1, paired bootstrap over test samples, Holm):

| contrast | Δ | per seed (1, 2, 42) | 95% CI | p (Holm) | group 95% CI | verdict |
|---|---|---|---|---|---|---|
| C + B1 − C | +0.011 | +0.012 +0.015 +0.007 | [−0.002, +0.025] | 0.16 | [−0.003, +0.027] | inconclusive (consistent direction, at practical target) |
| C − T | +0.010 | +0.034 −0.013 +0.010 | [−0.001, +0.022] | 0.16 | [−0.001, +0.023] | inconclusive |
| T + B1 − T | +0.007 | +0.015 +0.001 +0.004 | [+0.000, +0.014] | 0.14 | [−0.000, +0.016] | inconclusive (below practical target) |
| C + B1 − T + B1 | +0.015 | +0.031 +0.001 +0.012 | [−0.000, +0.030] | 0.16 | [−0.000, +0.031] | inconclusive; does not support the Phase 3 hypothesis |

No contrast meets the frozen rule for an overall improvement. C + B1 has no per-class regression against C on private test. Unlike val, its gain on private test is not a minority-recall gain (`image-sarcasm` recall 0.04).

Context only: the Phase 4 Kaggle run of the same architecture with class weights (`fusion_concat_frozen_reg_cw`) reported 0.402 ± 0.015 private macro-F1, selected on four-class macro-F1 with a different environment and its private test viewed during development; it is not directly comparable.

## 8. Qualitative analysis

A fixed sample of 44 val examples (seed 42, majority vote over seeds) is in `data/processed/empirical/qualitative/examples_val.csv`, in six categories: C fixes T (pool 94), C harms T (pool 80), B1 fixes a minority sample of C (pool 4), B1 harms C (pool 48), tagged caption with C correct (117) or wrong (76). The near balance between fixes and harms (94 vs 80) shows that C's small net gain hides much larger churn. The manual review columns (OCR correct, caption alone sufficient, image needed, label ambiguous, failure mechanism) are **not yet filled**; the plan asks for a second reviewer on a subset.

## 9. Limitations

- Three training seeds; intervals cover sampling of evaluation examples only, not training randomness.
- Validation has one `text-sarcasm` and 28 `image-sarcasm` samples; minority conclusions rest on internal folds and are fragile, as the val-to-test drop in `image-sarcasm` F1 shows.
- Private test was not pristine at the project level (see §2).
- Frozen encoders only; fine-tuning and cross-attention were out of scope.
- B4/B5 not run; the qualitative review, the logo audit (`data/processed/empirical/diagnostics/source_logo_audit.csv`) and the second-reviewer checks are pending.
- Label definitions are uncertain (§2), which caps what any model can reach on the minority classes.

## 10. Conclusion

Under matched, frozen-encoder settings, the caption carries most of the signal; adding CLIP features and square-root class weighting each move macro-F1 by about one point, in a consistent direction across val and private test, but neither effect is established at the predeclared level. Image-derived text (OCR, descriptions) adds nothing detectable, `text-sarcasm` is never recognized, and `image-sarcasm` gains seen on val do not carry to private test. The best-supported candidate is **C + B1** (private test 0.380 ± 0.007), reported as an inconclusive improvement over its plain-CE control.
